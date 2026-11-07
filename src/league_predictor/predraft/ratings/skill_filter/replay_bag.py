"""The K-batched serving replay: one day loop for a whole seed bag."""

from __future__ import annotations

from typing import TYPE_CHECKING

from league_predictor.predraft.ratings.skill_filter.model import BatchedSkillFilter, SkillFilter
from league_predictor.predraft.ratings.skill_filter.replay_inputs import (
    _day_statics,
    _ReplayDayStatics,
    to_slots_k,
)
from league_predictor.predraft.ratings.skill_filter.replay_timelines import (
    BagServingTimelines,
    _snapshot_day_bag,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus, NormStats


# ── seed-batched replay day used by production serving ────────────────────── #
# A bag is K members sharing one corpus; batching them through one day loop pays Python once.
def _replay_bag_day_and_write(
    bag: BatchedSkillFilter,
    H,
    H_org,
    prior,
    *,
    st: _ReplayDayStatics,
    db,
    plan,
    with_margin: bool,
):
    """One day's batched computation plus K-aware state writes."""

    d, d_org = bag.config.d, bag.config.d_org
    scored, h_slots, deltas, o_states, org_deltas, m_pred = _replay_bag_day(
        bag, H, H_org, prior, st=st, db=db, with_margin=with_margin
    )

    kpos = plan["known_pos"]
    H = _bag_scatter(
        H, h_slots.reshape(bag.K, -1, d)[:, kpos], plan["known_dedup"], accumulate=False
    )
    H = _bag_scatter(H, deltas.reshape(bag.K, -1, d)[:, kpos], plan["known_full"], accumulate=True)
    if d_org > 0:
        H_org = _bag_scatter(
            H_org, o_states.reshape(bag.K, -1, d_org), plan["org_dedup"], accumulate=False
        )
        H_org = _bag_scatter(
            H_org, org_deltas.reshape(bag.K, -1, d_org), plan["org_full"], accumulate=True
        )
    return H, H_org, scored, m_pred


def _replay_bag_day(
    bag: BatchedSkillFilter, H, H_org, prior, *, st: _ReplayDayStatics, db, with_margin: bool
):
    """one day's pure (write-free) training computation for ALL K members at once.

    Returns:
        The evolved states + deltas the caller writes back K-aware, plus the scored logit and
            the margin prediction (``None`` when off). Mirrors ``_replay``'s non-elo day body
            with a leading K axis; ``st``/``db`` are the SHARED day tensors, broadcast
            un-expanded wherever the op allows.
    """
    import torch

    cfg = bag.config
    K, d, d_org = bag.K, cfg.d, cfg.d_org
    n_games = db.a_won.shape[0]
    unknown = st.unknown  # [G,2,5,1] shared
    idx = st.idx  # [G,2,5] shared
    h_raw = H[:, idx.reshape(-1)].reshape(K, n_games, 2, 5, d)
    prior_slots = prior[:, idx.reshape(-1)].reshape(K, n_games, 2, 5, d)
    glob = bag.global_prior.view(K, 1, 1, 1, d)
    h_raw = torch.where(unknown, glob, h_raw)
    prior_slots = torch.where(unknown, glob, prior_slots)
    h_slots = bag.evolve(h_raw, prior_slots, db.gaps)

    orgs_flat = st.orgs_flat
    o_raw = H_org[:, orgs_flat].reshape(K, n_games, 2, d_org)
    o_states = bag.evolve_org(o_raw, bag.org_prior.view(K, 1, 1, d_org), db.team_gaps)
    ctx_n = st.ctx_n
    f = bag.team_scores(h_slots, o_states, ctx_n, gaps=db.gaps, debut=db.debut)  # [K,G,2]
    hist = bag.matchup_hist(db.roster_cont, db.debut, h_slots)
    scored = logit = bag.pair_logit(f, hist)

    if with_margin:
        m_pred = bag.expected_margin(f) + bag.margin_head(
            h_slots, o_states, ctx_n, gaps=db.gaps, debut=db.debut
        )
    else:
        m_pred = None

    p_a = torch.sigmoid(logit)  # [K,G]
    p_own = torch.stack([p_a, 1.0 - p_a], dim=-1)  # [K,G,2]
    surprise = st.outcome - p_own  # shared [G,2] broadcast up
    f_opp = f.flip(-1)

    team_perf = bag._expand_k(st.team_perf)
    x_team = torch.cat(
        [bag._expand_k(st.x_team_static), surprise.unsqueeze(-1), f_opp.unsqueeze(-1)], dim=-1
    )
    x_player = torch.cat(
        [
            bag._expand_k(st.x_player_static),
            to_slots_k(surprise.unsqueeze(-1)),
            to_slots_k(f_opp.unsqueeze(-1)),
        ],
        dim=-1,
    )
    kfac = bag.update_step_boost(db.roster_cont)
    deltas = bag.player_deltas(h_slots, x_player, kfac)
    org_deltas = bag.org_deltas(o_states, x_team, team_perf)
    deltas = deltas * (~unknown).float()
    return scored, h_slots, deltas, o_states, org_deltas, m_pred


def _scatter_plan(torch, ent_idx, K: int, *, dedup: bool):
    """One day's precomputed index tuple for ``_bag_scatter``: ``(src, krow, ecol)``.

    Args:
        dedup: gather each unique entity's FIRST occurrence into ``src`` (the evolved write's
            backward-double-count guard); off keeps every occurrence for the accumulate write.

    Returns:
        ``(src, krow, ecol)`` — ``krow``/``ecol`` are the flattened (member, entity)
            ``index_put`` coordinates. Depends only on the day's entity list and K.
    """
    if dedup:
        uniq, inverse = torch.unique(ent_idx, return_inverse=True)
        first = torch.full(
            (uniq.numel(),), ent_idx.numel(), dtype=torch.long, device=ent_idx.device
        )
        first.scatter_reduce_(
            0, inverse, torch.arange(ent_idx.numel(), device=ent_idx.device), reduce="amin"
        )
        ent, src = uniq, first
    else:
        ent, src = ent_idx, None
    m = ent.numel()
    krow = torch.arange(K, device=ent_idx.device).view(K, 1).expand(K, m).reshape(-1)
    ecol = ent.view(1, m).expand(K, m).reshape(-1)
    return (src, krow, ecol)


def precompute_bag_scatter_plans(statics: Iterable[_ReplayDayStatics], K: int) -> list[dict]:
    """Every day's four scatter plans, built ONCE per fit (epoch-invariant).

    Returns:
        Per day: the (dedup, accumulate) pairs for the player (``known``) and org state
            writes, plus ``known_pos`` — the INTEGER positions of the known slots. The
            compiled day graph selects known rows by integer gather because a boolean mask's
            data-dependent output size is an unbacked symbol inductor cannot lower (measured:
            ExpandView broadcast failure).
    """
    import torch

    return [
        {
            "known_dedup": _scatter_plan(torch, st.known_idx, K, dedup=True),
            "known_full": _scatter_plan(torch, st.known_idx, K, dedup=False),
            "org_dedup": _scatter_plan(torch, st.orgs_flat, K, dedup=True),
            "org_full": _scatter_plan(torch, st.orgs_flat, K, dedup=False),
            "known_pos": torch.nonzero(st.flat_known, as_tuple=True)[0],
        }
        for st in statics
    ]


def _bag_scatter(state, values, plan, *, accumulate: bool):
    """K-aware out-of-place state write; ``accumulate`` sums duplicate coordinates (deltas)."""
    src, krow, ecol = plan
    vals = values[:, src] if src is not None else values
    return state.index_put((krow, ecol), vals.reshape(-1, state.shape[-1]), accumulate=accumulate)


def replay_bag_for_serving(
    corpus: Corpus,
    models: list[SkillFilter],
    *,
    norm: NormStats,
    league_remap: dict[int, int] | None = None,
) -> BagServingTimelines:
    """K members' serving timelines from one K-batched no-grad replay.

    Args:
        models: share the one deployed architecture, differing only by fitted weights.

    Returns:
        one K-wide ``BagServingTimelines`` in member order (``.member(k)`` is member k's
            solo-shaped view). Drift pin
            ``tests/predraft/ratings/test_skill_filter_replay.py::TestBagServingReplay``:
            ``.member(k)`` equals ``replay_for_serving(corpus, models[k],...)`` — exact on
            structure, within 1e-6 on states.
    """
    import torch
    from torch.func import stack_module_state

    cfg = models[0].config
    K = len(models)
    d_org = cfg.d_org
    n_orgs = max(1, corpus.vocab.n_orgs)

    with torch.no_grad():
        params, _buffers = stack_module_state(models)
        bag = BatchedSkillFilter({name: p.detach() for name, p in params.items()}, cfg)

        raw_leagues = corpus.vocab.debut_league or [0]
        if league_remap is not None:
            raw_leagues = [league_remap.get(li, -1) for li in raw_leagues]
        debut_league = torch.tensor(raw_leagues, dtype=torch.int64)
        prior = bag.prior_for(debut_league)  # [K, n_players, d]
        H = prior
        H_org = bag.org_prior[:, None, :].expand(K, n_orgs, d_org).contiguous()
        timelines = BagServingTimelines(K=K, prior=prior.detach().clone())
        n_org_games = [0] * n_orgs
        for db in corpus.days:
            st = _day_statics(torch, db, norm)
            plan = precompute_bag_scatter_plans([st], K)[0]
            H, H_org, _scored, _m = _replay_bag_day_and_write(
                bag, H, H_org, prior, st=st, db=db, plan=plan, with_margin=False
            )
            _snapshot_day_bag(timelines, db, H, H_org, n_org_games=n_org_games)
    return timelines
