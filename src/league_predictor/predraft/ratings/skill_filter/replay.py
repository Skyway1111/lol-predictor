"""Chronological serving replay from corpus and frozen weights to query timelines."""

# Per day (start-of-day batching): gather + evolve participants over their full gap (decay
# composes exponentially, so evolve-at-read ≡ continuous decay) → predict → gated update.

from __future__ import annotations

from typing import TYPE_CHECKING

from league_predictor.predraft.ratings.skill_filter.replay_inputs import (
    _day_statics,
    _to_slots,
)
from league_predictor.predraft.ratings.skill_filter.replay_timelines import (
    ServingTimelines,
    _snapshot_day,
)

if TYPE_CHECKING:
    from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus, NormStats
    from league_predictor.predraft.ratings.skill_filter.model import SkillFilter


# sightline-ok: 11 - the test-owned differentiable replay is an independent DEER parity oracle
def replay_for_serving(
    corpus: Corpus,
    model: SkillFilter,
    *,
    norm: NormStats,
    league_remap: dict[int, int] | None = None,
) -> ServingTimelines:
    """Replay frozen weights without gradients into the query-time timelines."""
    import torch

    with torch.no_grad():
        cfg = model.config
        d, d_org = cfg.d, cfg.d_org
        n_orgs = max(1, corpus.vocab.n_orgs)
        # Every internal tensor lives wherever the model's params live, so a
        # GPU-resident model + GPU corpus replay entirely on-device.
        device = model.global_prior.device

        # Empty vocab (no eligible games) degenerates to one sentinel prior row.
        raw_leagues = corpus.vocab.debut_league or [0]
        if league_remap is not None:
            raw_leagues = [league_remap.get(li, -1) for li in raw_leagues]
        debut_league = torch.tensor(raw_leagues, dtype=torch.int64, device=device)
        prior = model.prior_for(debut_league)  # [n_players, d]
        H = prior
        H_org = model.org_prior.unsqueeze(0).expand(n_orgs, d_org).contiguous()

        timelines = ServingTimelines(prior=prior.detach().clone())
        n_org_games = [0] * n_orgs

        for db in corpus.days:
            st = _day_statics(torch, db, norm)
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
            p_a = torch.sigmoid(logit)
            p_own = torch.stack([p_a, 1.0 - p_a], dim=1)
            surprise = st.outcome - p_own
            f_opp = f.flip(1)
            x_team = torch.cat(
                [st.x_team_static, surprise.unsqueeze(-1), f_opp.unsqueeze(-1)], dim=-1
            )
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
            H = H.index_put((known_idx,), h_slots.reshape(-1, d)[flat_known])
            H = H.index_put((known_idx,), deltas.reshape(-1, d)[flat_known], accumulate=True)
            H_org = H_org.index_put((orgs_flat,), o_states.reshape(-1, d_org))
            H_org = H_org.index_put((orgs_flat,), org_deltas.reshape(-1, d_org), accumulate=True)

            _snapshot_day(timelines, db, H, H_org, n_org_games=n_org_games)

        return timelines
