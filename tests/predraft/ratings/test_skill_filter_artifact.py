"""Persistence and frozen-artifact compatibility for the Rating artifact."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from league_predictor.predraft.ratings.skill_filter.artifact import (
    _deployed_config,
    _deployed_norm,
    load_bag_artifact,
    save_bag_artifact,
)
from league_predictor.predraft.ratings.skill_filter.corpus_schema import (
    CTX_CHANNELS,
    PLAYER_CHANNELS,
    TEAM_CHANNELS,
    NormStats,
)
from league_predictor.predraft.ratings.skill_filter.model import FilterConfig, SkillFilter


def _norm() -> NormStats:
    return NormStats(
        player_mean=torch.randn(len(PLAYER_CHANNELS)),
        player_std=torch.rand(len(PLAYER_CHANNELS)) + 0.5,
        team_mean=torch.randn(len(TEAM_CHANNELS)),
        team_std=torch.rand(len(TEAM_CHANNELS)) + 0.5,
        ctx_mean=torch.randn(len(CTX_CHANNELS)),
        ctx_std=torch.rand(len(CTX_CHANNELS)) + 0.5,
    )


def _verbose_prd_config() -> dict:
    return {
        "d": 2,
        "d_org": 4,
        "hidden": 32,
        "seed": 0,
        "elo_mode": False,
        "elo_k": 0.1,
        "use_margins": True,
        "perf_encoder": False,
        "variance_readout": "off",
        "org_factor": "off",
        "dispersion_readout": "raw",
        "familiarity_readout": "off",
        "familiarity_pool": "mean",
        "staleness_readout": "off",
        "drop_channels": (),
        "international_head": False,
        "update_kfactor": "settled",
        "credit_split": True,
        "box_pace": False,
        "box_role_relative": False,
        "query_ctx": False,
    }


def test_current_verbose_config_translates_to_fixed_formula():
    assert _deployed_config(_verbose_prd_config()) == FilterConfig()


def test_deployed_norm_discards_retired_experiment_variants():
    raw = dataclasses.asdict(_norm())
    raw["ctx_mean_q"] = torch.full((len(CTX_CHANNELS),), 99.0)
    raw["player_mean_pace"] = torch.full((len(PLAYER_CHANNELS),), 99.0)

    norm = _deployed_norm(raw)

    assert set(norm.__dict__) == {
        "player_mean",
        "player_std",
        "team_mean",
        "team_std",
        "ctx_mean",
        "ctx_std",
    }


def test_verbose_config_rejects_a_non_prd_formula():
    raw = _verbose_prd_config()
    raw["variance_readout"] = "dilute"
    with pytest.raises(ValueError, match="unsupported Rating artifact formula"):
        _deployed_config(raw)


def test_single_artifact_round_trip(tmp_path):
    cfg = FilterConfig(d=4, seed=11)
    model = SkillFilter(cfg, n_leagues=2)
    with torch.no_grad():
        model.global_prior += 1.5
    path = tmp_path / "rating.pt"
    sha = save_bag_artifact(
        path,
        [model],
        cfg,
        _norm(),
        {"LCK": 0, "LPL": 1},
        signature=(5, "2026-01-20", "m5"),
        notes={"cv": "test"},
    )

    loaded, meta = load_bag_artifact(path, expected_sha=sha)
    loaded = loaded[0]

    assert loaded.config == cfg
    assert meta["signature"] == (5, "2026-01-20", "m5")
    assert torch.equal(loaded.global_prior, model.global_prior)


def test_bag_artifact_round_trips_member_weights(tmp_path):
    cfg = FilterConfig(d=4, hidden=8)
    models = [SkillFilter(FilterConfig(d=4, hidden=8, seed=seed), 2) for seed in range(3)]
    path = tmp_path / "rating_bag.pt"
    sha = save_bag_artifact(
        path,
        models,
        cfg,
        _norm(),
        {"LCK": 0},
        signature=(3, "2026-01-20", "bag"),
        notes={"k": 3},
    )

    loaded, meta = load_bag_artifact(path, expected_sha=sha)

    assert len(loaded) == 3
    assert meta["notes"] == {"k": 3}
    for expected, actual in zip(models, loaded, strict=True):
        assert all(
            torch.equal(expected.state_dict()[key], actual.state_dict()[key])
            for key in expected.state_dict()
        )


def test_sha_mismatch_refuses_load(tmp_path):
    cfg = FilterConfig()
    path = tmp_path / "rating.pt"
    save_bag_artifact(
        path,
        [SkillFilter(cfg, 1)],
        cfg,
        _norm(),
        {"LCK": 0},
        signature=(1,),
        notes={},
    )
    with pytest.raises(ValueError, match="SHA"):
        load_bag_artifact(path, expected_sha="deadbeef" * 8)
