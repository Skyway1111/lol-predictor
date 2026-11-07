"""Structural invariants of the deployed latent-skill formula."""

from __future__ import annotations

import pytest
import torch

from league_predictor.predraft.ratings.skill_filter.corpus_schema import CTX_CHANNELS
from league_predictor.predraft.ratings.skill_filter.model import FilterConfig, SkillFilter


def _inputs(d=4, d_org=4):
    generator = torch.Generator().manual_seed(7)
    slots = torch.randn(3, 2, 5, d, generator=generator)
    org = torch.randn(3, 2, d_org, generator=generator)
    ctx = torch.randn(3, 2, len(CTX_CHANNELS), generator=generator)
    gaps = torch.rand(3, 2, 5, generator=generator) * 30
    debut = (torch.rand(3, 2, 5, generator=generator) > 0.8).float()
    continuity = torch.randint(0, 40, (3, 2), generator=generator).float()
    return slots, org, ctx, gaps, debut, continuity


def test_decay_is_identity_at_zero_gap_and_moves_toward_prior():
    model = SkillFilter(FilterConfig(d=4), 2)
    state = torch.randn(3, 4)
    prior = torch.zeros_like(state)
    zero = model.evolve(state, prior, torch.zeros(3))
    later = model.evolve(state, prior, torch.full((3,), 100.0))
    assert torch.equal(zero, state)
    assert torch.all(later.abs() < state.abs())


def test_pair_logit_is_antisymmetric_with_deployed_dispersion():
    model = SkillFilter(FilterConfig(d=4), 3)
    slots, org, ctx, gaps, debut, continuity = _inputs()

    def score(s, o, c, gp, db, cont):
        strengths = model.team_scores(s, o, c, gaps=gp, debut=db)
        return model.pair_logit(strengths, model.matchup_hist(cont, db, s))

    forward = score(slots, org, ctx, gaps, debut, continuity)
    swapped = score(
        slots.flip(1), org.flip(1), ctx.flip(1), gaps.flip(1), debut.flip(1), continuity.flip(1)
    )
    assert torch.allclose(forward, -swapped, atol=1e-6)


def test_matchup_history_contains_continuity_debut_and_dispersion():
    model = SkillFilter(FilterConfig(d=4), 1)
    slots, _org, _ctx, _gaps, debut, continuity = _inputs()
    history = model.matchup_hist(continuity, debut, slots)
    assert history.shape == (3, 2, 3)
    assert torch.allclose(history[..., 2], slots.std(dim=-2).mean(dim=-1))


def test_update_boost_is_larger_for_fresh_lineups():
    model = SkillFilter(FilterConfig(), 1)
    boost = model.update_step_boost(torch.tensor([[0.0, 1000.0]]))
    assert boost[0, 0] > boost[0, 1]
    assert boost[0, 1].item() == pytest.approx(1.0, abs=1e-6)


def test_player_and_org_updates_have_expected_shapes_and_gradients():
    model = SkillFilter(FilterConfig(d=4), 1)
    slots, org, _ctx, _gaps, _debut, continuity = _inputs()
    x_player = torch.randn(3, 2, 5, model.gate_mlp[0].in_features - 4)
    x_org = torch.randn(3, 2, model.org_gate_mlp[0].in_features - 4)
    team_perf = torch.randn(3, 2, model.org_credit_mlp[0].in_features)
    player = model.player_deltas(slots, x_player, model.update_step_boost(continuity))
    organization = model.org_deltas(org, x_org, team_perf)
    (player.sum() + organization.sum()).backward()
    assert player.shape == slots.shape
    assert organization.shape == org.shape
    assert model.gate_mlp[-1].weight.grad is not None
    assert model.cand_mlp[-1].weight.grad is not None
    assert model.org_gate_mlp[-1].weight.grad is not None
    assert model.org_credit_mlp[-1].weight.grad is not None
