"""Quasi-DEER CUDA replay solver."""  # sightline-ok: 27 - one solver (see file-length-ok)

# file-length-ok: one solver's passes over one FlatReplay — flatten, the day map F, the
# chain scans, the forward solve, the implicit backward; each reads the last one's shape.

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

import torch

from league_predictor.predraft.ratings.skill_filter.corpus_schema import _move
from league_predictor.predraft.ratings.skill_filter.model import (
    BatchedSkillFilter,
    SkillFilter,
    hist_features,
)
from league_predictor.predraft.ratings.skill_filter.replay_inputs import (
    precompute_replay_statics,
    to_slots_k,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from league_predictor.predraft.ratings.skill_filter.corpus_schema import (
        Corpus,
        NormStats,
    )

# Frozen solver recipe (measured on the real corpus, benchmarks.md § deer-cuda): refresh the
# own-delta jvp diagonal every 5th iteration (any diagonal is a valid splitting — only the rate
# changes); Anderson depth 5 on both loops; 16k-game F chunks bound K=5 on the 16 GB card.
_JVP_EVERY = 5
_ANDERSON_M = 5
_GAME_CHUNK = 16384

# Same-day occurrences must carry a bit-identical recency gap; anything wider is a replay bug.
_SAME_DAY_GAP_TOL = 1e-9
# An Anderson step this large is extrapolating wildly; fall back to the plain fixed-point step.
_MAX_MIXING_GAMMA = 50.0


# ── corpus flattening ──────────────────────────────────────────────────────── #


@dataclass
class FlatReplay:
    """The whole corpus as one batch, in chronological (day, in-day) order.

    Attributes:
        ctx_n/x_player_static/x_team_static/team_perf/outcome/unknown/margin_*: game-level
            (N games), from the replay statics; gaps/debut/roster_cont/ctx_raw/a_won come
            straight from the DayBatches.
        slot_app/side_oapp: each game slot/side mapped to its appearance row — an appearance is
            one (entity, day) with >=1 game; ``-1`` marks an unknown slot.
        app_prev/oapp_prev: the same entity's previous appearance, ``-1`` at debut.
        app_gap/oapp_gap: the raw evolve gaps, identical across a day's games by the
            start-of-day invariant (asserted at build).
        app_ent/app_rank: each appearance's scan coordinates (entity row, appearance index).
    """

    n_games: int
    day_sizes: list  # games per day, chronological
    day_dates: list

    # game-level statics
    ctx_n: torch.Tensor
    x_player_static: torch.Tensor
    x_team_static: torch.Tensor
    team_perf: torch.Tensor
    outcome: torch.Tensor
    unknown: torch.Tensor
    gaps: torch.Tensor
    debut: torch.Tensor
    roster_cont: torch.Tensor
    ctx_raw: torch.Tensor
    a_won: torch.Tensor
    margin_target: torch.Tensor  # [N] auxiliary dominance label (0 where masked)
    margin_mask: torch.Tensor  # [N] 1 where any dominance channel present

    # player appearances
    n_app: int
    app_ent: torch.Tensor  # [M_p] player id
    app_day: torch.Tensor  # [M_p] day index (into day_dates)
    app_rank: torch.Tensor  # [M_p] 0-based appearance index of that player
    app_prev: torch.Tensor  # [M_p] previous appearance id or -1
    app_gap: torch.Tensor  # [M_p] raw days
    slot_app: torch.Tensor  # [N,2,5] appearance id per known slot, -1 unknown
    n_players: int
    max_rank: int

    # org appearances
    n_oapp: int
    oapp_ent: torch.Tensor
    oapp_day: torch.Tensor
    oapp_rank: torch.Tensor
    oapp_prev: torch.Tensor
    oapp_gap: torch.Tensor
    side_oapp: torch.Tensor  # [N,2]
    n_orgs: int
    max_orank: int

    debut_league: torch.Tensor  # [n_players]

    def to(self, device: str) -> FlatReplay:
        return _move(self, device)


# sightline-ok: 14 - one build's inputs threaded to its passes; a wrapper re-unpacks
def flatten_corpus(  # noqa: PLR0915 - linear precompute of flat columns + appearance chains
    corpus: Corpus, model: SkillFilter, *, norm: NormStats | None = None
) -> FlatReplay:
    """Build the FlatReplay: a one-time precompute, pure in (corpus, model config, norm).

    Args:
        model: supplies config, dtype and device — build with a CPU model and ``.to(device)``
            the result.
    """
    statics = precompute_replay_statics(corpus, norm=norm)
    device = model.global_prior.device

    day_sizes = [db.a_won.shape[0] for db in corpus.days]
    day_dates = [db.date for db in corpus.days]

    cat = torch.cat
    team_perf = cat([st.team_perf for st in statics])
    gaps = cat([db.gaps for db in corpus.days]).to(device)
    debut = cat([db.debut for db in corpus.days]).to(device)
    roster_cont = cat([db.roster_cont for db in corpus.days]).to(device)
    ctx_raw = cat([db.ctx for db in corpus.days]).to(device)
    a_won = cat([db.a_won for db in corpus.days]).to(device)
    n_games = int(a_won.shape[0])

    # ── appearance graph (python per-day loop; one-time) ── #
    n_players = max(1, corpus.vocab.n_players)
    n_orgs = max(1, corpus.vocab.n_orgs)

    app_ent: list[int] = []
    app_prev: list[int] = []
    app_gap: list[float] = []
    app_rank: list[int] = []
    app_day: list[int] = []
    oapp_ent: list[int] = []
    oapp_prev: list[int] = []
    oapp_gap: list[float] = []
    oapp_rank: list[int] = []
    oapp_day: list[int] = []
    last_app = [-1] * n_players
    rank_ctr = [0] * n_players
    olast = [-1] * n_orgs
    orank_ctr = [0] * n_orgs
    slot_app_days, side_oapp_days = [], []

    for day_i, db in enumerate(corpus.days):
        pids = db.players  # [G,2,5] cpu int64
        flat_p = pids.reshape(-1).tolist()
        flat_g = db.gaps.reshape(-1).tolist()
        day_app: dict[int, int] = {}
        slot_ids = []
        for p, gp in zip(flat_p, flat_g, strict=True):
            if p < 0:
                slot_ids.append(-1)
                continue
            m = day_app.get(p)
            if m is None:
                # sightline-ok: 11 - the org axis below dedups another entity, shape, padding rule
                m = len(app_ent)
                day_app[p] = m
                # sightline-ok: 11 - the org axis below dedups another entity, shape, padding rule
                app_ent.append(p)
                app_prev.append(last_app[p])
                app_gap.append(gp)
                app_rank.append(rank_ctr[p])
                app_day.append(day_i)
            else:
                # start-of-day invariant: every same-day occurrence carries the same gap
                assert abs(app_gap[m] - gp) < _SAME_DAY_GAP_TOL, "same-day gap mismatch (player)"
            slot_ids.append(m)
        for p, m in day_app.items():
            last_app[p] = m
            rank_ctr[p] += 1
        slot_app_days.append(torch.tensor(slot_ids, dtype=torch.int64).reshape(pids.shape))

        orgs_d = db.orgs.reshape(-1).tolist()
        tgaps_d = db.team_gaps.reshape(-1).tolist()
        day_oapp: dict[int, int] = {}
        side_ids = []
        for o, gp in zip(orgs_d, tgaps_d, strict=True):
            j = day_oapp.get(o)
            if j is None:
                # sightline-ok: 11 - the player axis above dedups another entity/shape/padding
                j = len(oapp_ent)
                day_oapp[o] = j
                # sightline-ok: 11 - the player axis above dedups another entity/shape/padding
                oapp_ent.append(o)
                oapp_prev.append(olast[o])
                oapp_gap.append(gp)
                oapp_rank.append(orank_ctr[o])
                oapp_day.append(day_i)
            else:
                assert abs(oapp_gap[j] - gp) < _SAME_DAY_GAP_TOL, "same-day gap mismatch (org)"
            side_ids.append(j)
        for o, j in day_oapp.items():
            olast[o] = j
            orank_ctr[o] += 1
        side_oapp_days.append(torch.tensor(side_ids, dtype=torch.int64).reshape(db.orgs.shape))

    t64 = partial(torch.tensor, dtype=torch.int64, device=device)
    tf = partial(torch.tensor, dtype=model.global_prior.dtype, device=device)

    return FlatReplay(
        n_games=n_games,
        day_sizes=day_sizes,
        day_dates=day_dates,
        ctx_n=cat([st.ctx_n for st in statics]),
        x_player_static=cat([st.x_player_static for st in statics]),
        x_team_static=cat([st.x_team_static for st in statics]),
        team_perf=team_perf,
        outcome=cat([st.outcome for st in statics]),
        unknown=cat([st.unknown for st in statics]),
        gaps=gaps,
        debut=debut,
        roster_cont=roster_cont,
        ctx_raw=ctx_raw,
        a_won=a_won,
        margin_target=cat([st.margin_target for st in statics]),
        margin_mask=cat([st.margin_mask for st in statics]),
        n_app=len(app_ent),
        app_ent=t64(app_ent),
        app_day=t64(app_day),
        app_rank=t64(app_rank),
        app_prev=t64(app_prev),
        app_gap=tf(app_gap),
        slot_app=torch.cat(slot_app_days).to(device),
        n_players=n_players,
        max_rank=max(rank_ctr) if rank_ctr else 1,
        n_oapp=len(oapp_ent),
        oapp_ent=t64(oapp_ent),
        oapp_day=t64(oapp_day),
        oapp_rank=t64(oapp_rank),
        oapp_prev=t64(oapp_prev),
        oapp_gap=tf(oapp_gap),
        side_oapp=torch.cat(side_oapp_days).to(device),
        n_orgs=n_orgs,
        max_orank=max(orank_ctr) if orank_ctr else 1,
        debut_league=t64(corpus.vocab.debut_league or [0]),
    )


# ── K-aware chunked scatter plans (built once per (flat, K, chunk)) ────────── #


@dataclass
class BagPlans:
    """Precomputed K-broadcast index tuples: per-chunk scatters plus the adjoint's chain."""

    K: int
    game_chunk: int
    chunks: list  # [(lo, hi, known_pos, p_krow, p_ecol, o_krow, o_ecol)]
    hp_pos: torch.Tensor  # appearance ids with a prev link
    hp_krow: torch.Tensor
    hp_prev: torch.Tensor
    ho_pos: torch.Tensor
    ho_krow: torch.Tensor
    ho_prev: torch.Tensor


def build_bag_plans(flat: FlatReplay, K: int, game_chunk: int = _GAME_CHUNK) -> BagPlans:
    """Precompute the scatter index plans for a K-member bag over ``flat`` (one-time)."""
    dev = flat.slot_app.device

    def kbroadcast(cols):
        m = cols.numel()
        krow = torch.arange(K, device=dev).view(K, 1).expand(K, m).reshape(-1)
        ecol = cols.view(1, m).expand(K, m).reshape(-1)
        return krow, ecol

    chunks = []
    for lo in range(0, flat.n_games, game_chunk):
        hi = min(lo + game_chunk, flat.n_games)
        sl = flat.slot_app[lo:hi].reshape(-1)
        known_pos = torch.nonzero(sl >= 0, as_tuple=True)[0]
        p_krow, p_ecol = kbroadcast(sl[known_pos])
        o_krow, o_ecol = kbroadcast(flat.side_oapp[lo:hi].reshape(-1))
        chunks.append((lo, hi, known_pos, p_krow, p_ecol, o_krow, o_ecol))

    hp_pos = torch.nonzero(flat.app_prev >= 0, as_tuple=True)[0]
    hp_krow, hp_prev = kbroadcast(flat.app_prev[hp_pos])
    ho_pos = torch.nonzero(flat.oapp_prev >= 0, as_tuple=True)[0]
    ho_krow, ho_prev = kbroadcast(flat.oapp_prev[ho_pos])
    return BagPlans(K, game_chunk, chunks, hp_pos, hp_krow, hp_prev, ho_pos, ho_krow, ho_prev)


# ── the K-batched, game-chunked day map F ──────────────────────────────────── #


def _hist_snapped_bag(bag: BatchedSkillFilter, flat: FlatReplay, h_slots, *, lo: int, hi: int):
    """``bag.matchup_hist`` with the dispersion channel eps-snapped: ``std < eps -> exactly 0``."""
    # ``std``'s kink at zero variance (five identical latents sit exactly on it) hands the
    # solver's converged-to-tol trajectory an O(1) garbage subgradient. eps = 10x solver tol.
    base = bag._expand_k(hist_features(flat.roster_cont[lo:hi], flat.debut[lo:hi]))
    eps = 1e-8 if h_slots.dtype == torch.float64 else 1e-4
    disp = bag.roster_dispersion(h_slots)
    disp = torch.where(disp > eps, disp, torch.zeros_like(disp))
    return torch.cat([base, disp.unsqueeze(-1)], dim=-1)


def evaluate_F_bag(  # noqa: N802, PLR0915 - F is the DEER map's casing; linear tensor assembly
    bag: BatchedSkillFilter,
    flat: FlatReplay,
    W: torch.Tensor,
    U: torch.Tensor,
    prior: torch.Tensor,
    plans: BagPlans,
    *,
    with_margin: bool,
    with_diag: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """one K-batched parallel application of every day map, game-chunked.

    Args:
        W: ``[K,M,d]`` appearance states; ``U`` ``[K,Mo,d_org]``; ``prior`` ``[K,n_players,d]``.
        with_diag: also compute the own-state delta diagonals, jvp'd inside the chunk loop
            while activations are live.

    Returns:
        ``(W_new, U_new, logits [K,N], m_pred [K,N]|None, dd_p [K,M,d], dd_o [K,Mo,d_org])``;
            the diagonals are ``None`` unless ``with_diag``. Differentiable end-to-end under
            grad.
    """
    cfg = bag.config
    K, d, d_org = bag.K, cfg.d, cfg.d_org
    dtype = W.dtype

    app_prior = prior[:, flat.app_ent]  # [K,M,d]
    prev_ok = (flat.app_prev >= 0).unsqueeze(-1)
    prev_state = torch.where(prev_ok, W[:, flat.app_prev.clamp(min=0)], app_prior)
    app_evolved = bag.evolve(prev_state, app_prior, flat.app_gap)  # [K,M,d]

    op = bag.org_prior[:, None, :]  # [K,1,d_org]
    oprev_ok = (flat.oapp_prev >= 0).unsqueeze(-1)
    oprev = torch.where(oprev_ok, U[:, flat.oapp_prev.clamp(min=0)], op)
    oapp_evolved = bag.evolve_org(oprev, op, flat.oapp_gap)  # [K,Mo,d_org]

    glob = bag.global_prior.view(K, 1, 1, 1, d)
    W_new = app_evolved
    U_new = oapp_evolved
    logits_parts, m_parts = [], []
    dd_p = torch.zeros(K, flat.n_app, d, dtype=dtype, device=W.device) if with_diag else None
    dd_o = torch.zeros(K, flat.n_oapp, d_org, dtype=dtype, device=W.device) if with_diag else None

    for lo, hi, known_pos, p_krow, p_ecol, o_krow, o_ecol in plans.chunks:
        h_slots = app_evolved[:, flat.slot_app[lo:hi].clamp(min=0)]  # [K,G,2,5,d]
        unk = flat.unknown[lo:hi]
        h_slots = torch.where(unk, glob, h_slots)
        o_states = oapp_evolved[:, flat.side_oapp[lo:hi]]  # [K,G,2,d_org]

        ctx_n = flat.ctx_n[lo:hi]
        gaps, debut = flat.gaps[lo:hi], flat.debut[lo:hi]
        f = bag.team_scores(h_slots, o_states, ctx_n, gaps=gaps, debut=debut)  # [K,G,2]
        hist = _hist_snapped_bag(bag, flat, h_slots, lo=lo, hi=hi)
        logit = bag.pair_logit(f, hist)  # [K,G]
        logits_parts.append(logit)
        if with_margin:
            m_parts.append(
                bag.expected_margin(f)
                + bag.margin_head(h_slots, o_states, ctx_n, gaps=gaps, debut=debut)
            )

        p_a = torch.sigmoid(logit)
        p_own = torch.stack([p_a, 1.0 - p_a], dim=-1)  # [K,G,2]
        surprise = flat.outcome[lo:hi] - p_own
        f_opp = f.flip(-1)

        x_team = torch.cat(
            [bag._expand_k(flat.x_team_static[lo:hi]), surprise.unsqueeze(-1), f_opp.unsqueeze(-1)],
            dim=-1,
        )
        x_player = torch.cat(
            [
                bag._expand_k(flat.x_player_static[lo:hi]),
                to_slots_k(surprise.unsqueeze(-1)),
                to_slots_k(f_opp.unsqueeze(-1)),
            ],
            dim=-1,
        )
        kfac = bag.update_step_boost(flat.roster_cont[lo:hi])
        known_f = (~unk).to(dtype)
        deltas = bag.player_deltas(h_slots, x_player, kfac) * known_f
        team_perf = bag._expand_k(flat.team_perf[lo:hi])
        org_deltas = bag.org_deltas(o_states, x_team, team_perf)

        W_new = W_new.index_put(
            (p_krow, p_ecol),
            deltas.reshape(K, -1, d)[:, known_pos].reshape(-1, d),
            accumulate=True,
        )
        U_new = U_new.index_put((o_krow, o_ecol), org_deltas.reshape(-1, d_org), accumulate=True)

        if with_diag:
            assert dd_p is not None
            assert dd_o is not None
            hh = h_slots.detach()
            xp = x_player.detach()
            kf = kfac.detach()

            def fp(t, xp=xp, kf=kf, known_f=known_f):
                return bag.player_deltas(t, xp, kf) * known_f

            for c in range(d):
                tangent = torch.zeros_like(hh)
                tangent[..., c] = 1.0
                jv = torch.func.jvp(fp, (hh,), (tangent,))[1]
                col = jv[..., c].reshape(K, -1)[:, known_pos].reshape(-1)
                dd_p[:, :, c] = dd_p[:, :, c].index_put((p_krow, p_ecol), col, accumulate=True)

            oo = o_states.detach()
            xt = x_team.detach()
            tp = team_perf.detach()

            def fo(t, xt=xt, tp=tp):
                return bag.org_deltas(t, xt, tp)

            for c in range(d_org):
                tangent = torch.zeros_like(oo)
                tangent[..., c] = 1.0
                jv = torch.func.jvp(fo, (oo,), (tangent,))[1]
                dd_o[:, :, c] = dd_o[:, :, c].index_put(
                    (o_krow, o_ecol), jv[..., c].reshape(-1), accumulate=True
                )

    logits = torch.cat(logits_parts, dim=1)  # [K,N]
    m_pred = torch.cat(m_parts, dim=1) if with_margin else None
    return W_new, U_new, logits, m_pred, dd_p, dd_o


# ── chain scans (device-agnostic; CPU-tested) ──────────────────────────────── #


def _chain_scan(D, R, *, ent, rank, n_ent, max_rank):
    """Solve ``delta[m] = D[m] * delta[prev(m)] + R[m]`` along every entity chain at once.

    Returns:
        The per-appearance solution: chains padded to ``[n_ent, max_rank, C]``, Hillis-Steele
            inclusive scan over affine maps (A,B) with combine (A_j*A_i, A_j*B_i + B_j),
            gathered back — log2(max_rank) tensor steps. Debut rows carry D=0 (the caller
            zeroes them), so a constant row contributes exactly its own R.
    """
    A = torch.ones(n_ent, max_rank, D.shape[-1], dtype=D.dtype, device=D.device)
    B = torch.zeros_like(A)
    A[ent, rank] = D
    B[ent, rank] = R
    s = 1
    while s < max_rank:
        A_prev = torch.ones_like(A)
        B_prev = torch.zeros_like(B)
        A_prev[:, s:] = A[:, :-s]
        B_prev[:, s:] = B[:, :-s]
        B = B + A * B_prev
        A = A * A_prev
        s *= 2
    return B[ent, rank]


def _chain_scan_rev(D, R, *, ent, rank, n_ent, max_rank):
    """Solve the TRANSPOSED chain system ``y[m] = D[child(m)] * y[child(m)] + R[m]``.

    Returns:
        The adjoint's chain part: the same associative scan run right-to-left on the padded
            layout (child = the same entity's next appearance; tails carry nothing).
    """
    Dp = torch.zeros(n_ent, max_rank, D.shape[-1], dtype=D.dtype, device=D.device)
    B = torch.zeros_like(Dp)
    Dp[ent, rank] = D
    B[ent, rank] = R
    # A[e, r] = D of the appearance at rank r+1 (0 past the chain tail)
    A = torch.zeros_like(Dp)
    A[:, :-1] = Dp[:, 1:]
    s = 1
    while s < max_rank:
        A_nxt = torch.ones_like(A)
        B_nxt = torch.zeros_like(B)
        A_nxt[:, : max_rank - s] = A[:, s:]
        B_nxt[:, : max_rank - s] = B[:, s:]
        B = A * B_nxt + B
        A = A * A_nxt
        s *= 2
    return B[ent, rank]


# ── quasi-Newton pieces (K-batched) ────────────────────────────────────────── #


def _decay_bag(gap, log_tau):
    """[M] gaps x [K,C] log-tau -> [K,M,C] evolve carry; exactly 1 at gap<=0, as evolve does."""
    tau = torch.exp(log_tau).unsqueeze(1)  # [K,1,C]
    dec = torch.exp(-gap.view(1, -1, 1) / tau)
    return torch.where(gap.view(1, -1, 1) > 0, dec, torch.ones_like(dec))


def _scan_k(scan_fn, D, R, *, ent, rank, n_ent: int, max_rank: int):
    """Run a chain scan with the K axis folded into the channel: [K,M,C] -> [M,K*C] -> scan."""
    K, M, C = D.shape
    D2 = D.permute(1, 0, 2).reshape(M, K * C)
    R2 = R.permute(1, 0, 2).reshape(M, K * C)
    out = scan_fn(D2, R2, ent=ent, rank=rank, n_ent=n_ent, max_rank=max_rank)
    return out.reshape(M, K, C).permute(1, 0, 2)


def _chain_D(bag: BatchedSkillFilter, flat: FlatReplay, dd_p, dd_o):  # noqa: N802 - D_p/D_o
    """Chain Jacobian diagonals D = decay * (1 + own-delta diag), zeroed at debuts (no prev)."""
    dec_p = _decay_bag(flat.app_gap, bag.params["log_tau"])
    dec_o = _decay_bag(flat.oapp_gap, bag.params["log_tau_org"])
    D_p = dec_p * (1.0 + dd_p)
    D_o = dec_o * (1.0 + dd_o)
    zero = torch.zeros(1, dtype=D_p.dtype, device=D_p.device)
    D_p = torch.where((flat.app_prev >= 0).view(1, -1, 1), D_p, zero)
    D_o = torch.where((flat.oapp_prev >= 0).view(1, -1, 1), D_o, zero)
    return D_p, D_o


class _AndersonMixer:
    """Anderson(m) acceleration over preconditioned iterates, difference form.

    Attributes:
        m: mixing depth. ``x+ = x_k + g_k - sum_j gamma_j (dx_j + dg_j)``, gamma from the small
            LS ``min ||g_k - dG^T gamma||``. Exactness is unaffected (the caller residual-tests)
            — mixing only changes the RATE. Vector arithmetic stays in the iterate dtype; only
            the tiny m x m solve runs in f64. Wild extrapolations (|gamma| >= 50) fall back to
            the plain step.
    """

    def __init__(self, m: int) -> None:
        self.m = m
        self._xs: list[torch.Tensor] = []
        self._gs: list[torch.Tensor] = []

    def restart(self) -> None:
        """Drop the memory (the forward solve restarts on two consecutive residual rises)."""
        self._xs.clear()
        self._gs.clear()

    def mix(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """One mixed update for the fixed-point step ``x <- x + g`` (flat vectors)."""
        self._xs.append(x)
        self._gs.append(g)
        if len(self._xs) > self.m + 1:
            self._xs.pop(0)
            self._gs.pop(0)
        if len(self._xs) < 2:  # noqa: PLR2004 - a difference needs two iterates
            return x + g
        Gm = torch.stack(self._gs)
        Xm = torch.stack(self._xs)
        dG = Gm[1:] - Gm[:-1]
        A = (dG @ dG.T).double() + 1e-10 * torch.eye(
            dG.shape[0], device=x.device, dtype=torch.float64
        )
        gamma = torch.linalg.solve(A, (dG @ Gm[-1]).double()).to(x.dtype)
        if float(gamma.abs().max()) >= _MAX_MIXING_GAMMA:
            return x + g
        dX = Xm[1:] - Xm[:-1]
        return (Xm[-1] + Gm[-1]) - gamma @ (dX + dG)


# ── forward solve + implicit backward ──────────────────────────────────────── #


@dataclass
class BagSolve:
    """One converged trajectory + the diagnostics the trainer and backward reuse."""

    W: torch.Tensor
    U: torch.Tensor
    logits: torch.Tensor
    m_pred: torch.Tensor | None
    dd_p: torch.Tensor  # own-delta diagonals at the converged trajectory
    dd_o: torch.Tensor


def solve_forward_bag(
    bag: BatchedSkillFilter,
    flat: FlatReplay,
    plans: BagPlans,
    *,
    with_margin: bool,
    W0: torch.Tensor | None = None,
    U0: torch.Tensor | None = None,
    tol: float | None = None,
    max_iters: int = 200,
    jvp_every: int | None = _JVP_EVERY,
    anderson_m: int = _ANDERSON_M,
) -> BagSolve:
    """K-batched quasi-DEER forward solve of the replay fixed point; builds no autograd graph.

    Args:
        W0: warm-starts the trajectory with ``U0`` (an adjacent epoch's converged solve);
            ``None`` cold-starts at the evolve-only trajectory (every appearance at its prior).
        jvp_every: refresh the own-delta diagonal via jvp every n-th iteration; ``None`` never
            (decay-only carry — any diagonal is a valid splitting, only the RATE changes).
        anderson_m: mixing depth, 0 = plain preconditioned iteration.

    Raises:
        RuntimeError: the residual budget ``max_iters`` is exhausted.
    """
    cfg = bag.config
    dtype = bag.global_prior.dtype
    if tol is None:
        tol = 1e-10 if dtype == torch.float64 else 1e-5

    with torch.no_grad():
        prior = bag.prior_for(flat.debut_league)  # [K,n_players,d]
        W = W0.clone() if W0 is not None else prior[:, flat.app_ent].clone()
        U = (
            U0.clone()
            if U0 is not None
            else bag.org_prior[:, None, :].expand(bag.K, flat.n_oapp, cfg.d_org).contiguous()
        )

        dd_p = torch.zeros(bag.K, flat.n_app, cfg.d, dtype=dtype, device=W.device)
        dd_o = torch.zeros(bag.K, flat.n_oapp, cfg.d_org, dtype=dtype, device=W.device)
        mixer = _AndersonMixer(anderson_m) if anderson_m > 0 else None

        err_history: list[float] = []
        for it in range(max_iters):
            refresh = jvp_every is not None and it % jvp_every == 0
            W_new, U_new, logits, m_pred, dd_p_new, dd_o_new = evaluate_F_bag(
                bag, flat, W, U, prior, plans, with_margin=with_margin, with_diag=refresh
            )
            if refresh:
                assert dd_p_new is not None  # with_diag was set
                assert dd_o_new is not None  # with_diag was set
                dd_p, dd_o = dd_p_new, dd_o_new

            Rp = W_new - W
            Ro = U_new - U
            err = max(float(Rp.abs().max()), float(Ro.abs().max()) if Ro.numel() else 0.0)
            err_history.append(err)
            if err <= tol:
                return BagSolve(W, U, logits, m_pred, dd_p, dd_o)
            rising = len(err_history) >= 3 and err > err_history[-2] > err_history[-3]  # noqa: PLR2004 - [-3] is read
            if mixer is not None and rising:
                mixer.restart()  # two consecutive rises: the memory is misdirecting

            D_p, D_o = _chain_D(bag, flat, dd_p, dd_o)
            dW = _scan_k(
                _chain_scan,
                D_p,
                Rp,
                ent=flat.app_ent,
                rank=flat.app_rank,
                n_ent=flat.n_players,
                max_rank=flat.max_rank,
            )
            dU = _scan_k(
                _chain_scan,
                D_o,
                Ro,
                ent=flat.oapp_ent,
                rank=flat.oapp_rank,
                n_ent=flat.n_orgs,
                max_rank=flat.max_orank,
            )
            dW = torch.nan_to_num(dW, nan=0.0).clamp(-1e8, 1e8)
            dU = torch.nan_to_num(dU, nan=0.0).clamp(-1e8, 1e8)

            if mixer:
                x_new = mixer.mix(
                    torch.cat([W.reshape(-1), U.reshape(-1)]),
                    torch.cat([dW.reshape(-1), dU.reshape(-1)]),
                )
                nW = W.numel()
                W = x_new[:nW].reshape(W.shape)
                U = x_new[nW:].reshape(U.shape)
            else:
                W = W + dW
                U = U + dU

        residual = err_history[-1] if err_history else float("inf")
        raise RuntimeError(
            f"forward DEER solve did not converge after {max_iters} iterations "
            f"(residual={residual:.6g}, tolerance={tol:.6g})"
        )


def _adjoint_seed(lam0, gW, gU):
    """The adjoint's starting pair: the warm start when one is given, else zeros."""
    if lam0 is None:
        return torch.zeros_like(gW), torch.zeros_like(gU)
    return lam0[0].clone(), lam0[1].clone()


def backward_implicit_bag(
    bag: BatchedSkillFilter,
    flat: FlatReplay,
    sol: BagSolve,
    plans: BagPlans,
    loss_fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor],
    param_leaves: Mapping,
    *,
    with_margin: bool,
    tol: float | None = None,
    max_iters: int = 100,
    lam0: tuple[torch.Tensor, torch.Tensor] | None = None,
    anderson_m: int = _ANDERSON_M,
) -> tuple[dict, tuple[torch.Tensor, torch.Tensor]]:
    """EXACT gradients at the converged trajectory via implicit differentiation.

    Args:
        loss_fn: ``(logits [K,N], m_pred) -> scalar``, the sum of member losses — so each
            member's slice of the returned grads is exactly its own gradient.
        lam0: warm-starts the adjoint (the previous epoch's pair).

    Returns:
        ``(gradients over param_leaves, (lamW, lamU))``.
            With ``s* = F(s*, theta)``, ``dL/dtheta = dL/dtheta|direct + lambda^T dF/dtheta``;
            the adjoint solves ``(I - J^T) lambda = g`` by a chain-preconditioned iteration
            whose correction uses the TRUE Jacobian via one vjp through the single F graph, so
            the result is exact at convergence however rough the diagonal is (it sets the rate).

    Raises:
        RuntimeError: the adjoint does not converge within ``max_iters``.
    """
    dtype = bag.global_prior.dtype
    if tol is None:
        tol = 1e-11 if dtype == torch.float64 else 1e-6
    names, params = zip(*[(n, p) for n, p in param_leaves.items() if p.requires_grad], strict=True)

    W = sol.W.detach().requires_grad_(True)  # noqa: FBT003 - torch requires_grad_ is positional
    U = sol.U.detach().requires_grad_(True)  # noqa: FBT003 - torch requires_grad_ is positional
    prior = bag.prior_for(flat.debut_league)  # in-graph: prior params get their vjp share
    W2, U2, logits, m_pred, _, _ = evaluate_F_bag(
        bag, flat, W, U, prior, plans, with_margin=with_margin
    )
    loss = loss_fn(logits, m_pred)

    gW, gU = torch.autograd.grad(loss, (W, U), retain_graph=True)

    with torch.no_grad():
        D_p, D_o = _chain_D(bag, flat, sol.dd_p, sol.dd_o)

    d, d_org = bag.config.d, bag.config.d_org
    lamW, lamU = _adjoint_seed(lam0, gW, gU)
    err_history: list[float] = []
    mixer = _AndersonMixer(anderson_m) if anderson_m > 0 else None
    it = 0
    converged = False
    for it in range(max_iters):  # noqa: B007 - the final it is the iteration count returned below
        JtW, JtU = torch.autograd.grad(
            (W2, U2), (W, U), grad_outputs=(lamW, lamU), retain_graph=True
        )
        with torch.no_grad():
            # chain^T lambda: (Jc^T lam)[prev(m)] = D[m] * lam[m]
            DlamW = (D_p * lamW)[:, plans.hp_pos].reshape(-1, d)
            chW = torch.zeros_like(lamW).index_put(
                (plans.hp_krow, plans.hp_prev), DlamW, accumulate=True
            )
            DlamU = (D_o * lamU)[:, plans.ho_pos].reshape(-1, d_org)
            chU = torch.zeros_like(lamU).index_put(
                (plans.ho_krow, plans.ho_prev), DlamU, accumulate=True
            )
            rhsW = gW + (JtW - chW)
            rhsU = gU + (JtU - chU)
            lamW_new = _scan_k(
                _chain_scan_rev,
                D_p,
                rhsW,
                ent=flat.app_ent,
                rank=flat.app_rank,
                n_ent=flat.n_players,
                max_rank=flat.max_rank,
            )
            lamU_new = _scan_k(
                _chain_scan_rev,
                D_o,
                rhsU,
                ent=flat.oapp_ent,
                rank=flat.oapp_rank,
                n_ent=flat.n_orgs,
                max_rank=flat.max_orank,
            )
            err = max(float((lamW_new - lamW).abs().max()), float((lamU_new - lamU).abs().max()))
            err_history.append(err)
            scale = max(1.0, float(lamW_new.abs().max()))
            if mixer:
                x_new = mixer.mix(
                    torch.cat([lamW.reshape(-1), lamU.reshape(-1)]),
                    torch.cat([(lamW_new - lamW).reshape(-1), (lamU_new - lamU).reshape(-1)]),
                )
                nW = lamW.numel()
                lamW = x_new[:nW].reshape(lamW.shape)
                lamU = x_new[nW:].reshape(lamU.shape)
            else:
                lamW, lamU = lamW_new, lamU_new
        if err <= tol * scale:  # relative stop: lambda magnitudes scale with the loss sum
            converged = True
            break

    if not converged:
        residual = err_history[-1] if err_history else float("inf")
        raise RuntimeError(
            f"adjoint DEER solve did not converge after {max_iters} iterations "
            f"(residual={residual:.6g}, tolerance={tol:.6g})"
        )

    # direct + implicit param grads in one graph traversal: d(loss)/dtheta holding s fixed
    # rides the same backward as lambda^T dF/dtheta via the multi-output grad call.
    combined = torch.autograd.grad(
        (loss, W2, U2),
        params,
        grad_outputs=(torch.ones_like(loss), lamW, lamU),
        allow_unused=True,
    )
    grads = {
        n: (g if g is not None else torch.zeros_like(p))
        for n, p, g in zip(names, params, combined, strict=True)
    }
    return grads, (lamW.detach(), lamU.detach())
