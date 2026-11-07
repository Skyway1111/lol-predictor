"""Independent differentiable sequential oracle for DEER parity and leakage tests."""

# Per day (start-of-day batching): gather + evolve participants over their full gap (decay
# composes exponentially, so evolve-at-read ≡ continuous decay) → predict → gated update.

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from league_predictor.predraft.ratings.skill_filter.replay_inputs import (
    _day_statics,
    _ReplayDayStatics,
    _to_slots,
)

if TYPE_CHECKING:
    import torch

    from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus, NormStats
    from league_predictor.predraft.ratings.skill_filter.model import SkillFilter


@dataclass
class ReplayResult:
    """Per-game pre-game predictions in corpus (chronological) order.

    Attributes:
        logits/labels: ``[N]`` in chronological game order. Deliberately NO ``dates`` field —
            a caller that needs per-game dates rebuilds them from the corpus.
        margin_pred: ``[N]`` predicted side-0 dominance (``expected_margin`` + residual head).
        margin_target: ``[N]`` observed masked dominance composite.
        margin_mask: ``[N]`` 1 where any dominance channel was present. The three ``margin_*``
            are the train-only auxiliary signal, populated ONLY under ``with_margin``.
    """

    logits: torch.Tensor
    labels: torch.Tensor
    margin_pred: torch.Tensor | None = None
    margin_target: torch.Tensor | None = None
    margin_mask: torch.Tensor | None = None


def replay_for_training(  # noqa: PLR0915 - independent chronological tensor oracle
    corpus: Corpus,
    model: SkillFilter,
    *,
    norm: NormStats | None = None,
    with_margin: bool = False,
    statics: list[_ReplayDayStatics] | None = None,
) -> ReplayResult:
    """Full-unroll differentiable replay; returns per-game logits/labels.

    Args:
        with_margin: also emit the per-game dominance prediction + masked target (the
            trainer's auxiliary loss).
        statics: the per-day epoch-invariant inputs from ``precompute_replay_statics``;
            ``None`` builds them on the fly per day, byte-identically via the same builder.
    """
    import torch

    norm = norm if norm is not None else corpus.norm
    cfg = model.config
    d, d_org = cfg.d, cfg.d_org
    n_orgs = max(1, corpus.vocab.n_orgs)
    # Every internal tensor lives wherever the model's params live, so a
    # GPU-resident model + GPU corpus replay entirely on-device.
    device = model.global_prior.device

    # Empty vocab (no eligible games) degenerates to one sentinel prior row.
    raw_leagues = corpus.vocab.debut_league or [0]
    debut_league = torch.tensor(raw_leagues, dtype=torch.int64, device=device)
    prior = model.prior_for(debut_league)  # [n_players, d]
    H = prior
    H_org = model.org_prior.unsqueeze(0).expand(n_orgs, d_org).contiguous()

    grad_on = torch.is_grad_enabled()

    all_logits, all_labels = [], []
    all_m_pred, all_m_target, all_m_mask = [], [], []
    for day_i, db in enumerate(corpus.days):
        # Epoch-invariant inputs: precomputed once per fit (``statics``) or built on the fly.
        st = statics[day_i] if statics is not None else _day_statics(torch, db, norm)
        n_games = db.a_won.shape[0]
        unknown = st.unknown  # [G,2,5,1]
        idx = st.idx  # [G,2,5] clamped player rows
        h_raw = H[idx.reshape(-1)].reshape(n_games, 2, 5, d)
        prior_slots = prior[idx.reshape(-1)].reshape(n_games, 2, 5, d)
        glob = model.global_prior.expand_as(h_raw)
        h_raw = torch.where(unknown, glob, h_raw)
        prior_slots = torch.where(unknown, glob, prior_slots)
        h_slots = model.evolve(h_raw, prior_slots, db.gaps)

        orgs_flat = st.orgs_flat
        o_raw = H_org[orgs_flat].reshape(n_games, 2, d_org)
        o_states = model.evolve_org(o_raw, model.org_prior.expand_as(o_raw), db.team_gaps)
        ctx_n = st.ctx_n
        f = model.team_scores(h_slots, o_states, ctx_n, gaps=db.gaps, debut=db.debut)
        hist = model.matchup_hist(db.roster_cont, db.debut, h_slots)
        logit = model.pair_logit(f, hist)
        all_logits.append(logit)
        all_labels.append(db.a_won)

        if with_margin:
            # Predicted side-0 dominance: gap-explained part + latent residual, both read from
            # PRE-game state, so the prediction is leak-safe.
            m_pred = model.expected_margin(f) + model.margin_head(
                h_slots, o_states, ctx_n, gaps=db.gaps, debut=db.debut
            )
            all_m_pred.append(m_pred)
            all_m_target.append(st.margin_target)
            all_m_mask.append(st.margin_mask)

        p_a = torch.sigmoid(logit)
        p_own = torch.stack([p_a, 1.0 - p_a], dim=1)
        surprise = st.outcome - p_own
        f_opp = f.flip(1)
        x_team = torch.cat([st.x_team_static, surprise.unsqueeze(-1), f_opp.unsqueeze(-1)], dim=-1)
        x_player = torch.cat(
            [
                st.x_player_static,
                _to_slots(surprise.unsqueeze(-1)),
                _to_slots(f_opp.unsqueeze(-1)),
            ],
            dim=-1,
        )
        deltas = model.player_deltas(h_slots, x_player, model.update_step_boost(db.roster_cont))
        org_deltas = model.org_deltas(o_states, x_team, st.team_perf)

        deltas = deltas * (~unknown).float()
        flat_known = st.flat_known
        known_idx = st.known_idx
        # Evolved-state writes are DEDUPED per entity-day on the grad-on path (see
        # ``_write_evolved``); delta writes keep all occurrences (accumulate=True sums).
        H = _write_evolved(torch, H, known_idx, h_slots.reshape(-1, d)[flat_known], grad_on=grad_on)
        H = H.index_put((known_idx,), deltas.reshape(-1, d)[flat_known], accumulate=True)
        H_org = _write_evolved(
            torch, H_org, orgs_flat, o_states.reshape(-1, d_org), grad_on=grad_on
        )
        H_org = H_org.index_put((orgs_flat,), org_deltas.reshape(-1, d_org), accumulate=True)

    return ReplayResult(
        logits=torch.cat(all_logits) if all_logits else torch.zeros(0, device=device),
        labels=torch.cat(all_labels) if all_labels else torch.zeros(0, device=device),
        margin_pred=torch.cat(all_m_pred) if all_m_pred else None,
        margin_target=torch.cat(all_m_target) if all_m_target else None,
        margin_mask=torch.cat(all_m_mask) if all_m_mask else None,
    )


def _write_evolved(torch, state, idx, values, *, grad_on):
    """Out-of-place write of the evolved start-of-day states into the big state tensor."""
    # ``grad_on`` dedups to one write per entity-day: ``index_put``'s backward would otherwise
    # hand the full gradient to every duplicate, compounding x maps-per-day down a timeline.
    if grad_on:
        uniq, ev = _dedup_first(torch, idx, values)
        return state.index_put((uniq,), ev)
    # The skip is sound ONLY when there is no backward to double-count; tie it to that fact.
    assert not torch.is_grad_enabled()
    return state.index_put((idx,), values)


def _dedup_first(torch, idx, values):
    """(indices, values) → one (index, value) per unique index — the FIRST occurrence."""
    uniq, inverse = torch.unique(idx, return_inverse=True)
    first = torch.full((uniq.numel(),), idx.numel(), dtype=torch.long, device=idx.device)
    first.scatter_reduce_(0, inverse, torch.arange(idx.numel(), device=idx.device), reduce="amin")
    return uniq, values[first]
