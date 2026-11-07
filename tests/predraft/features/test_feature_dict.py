"""Tests for the composed pre-draft feature dict: its keys and its locked parity snapshot."""

import sqlite3
from datetime import date

import pytest

from league_predictor.db.schema import apply_current_schema as create_schema
from league_predictor.predraft.features.snapshots import (
    build_pre_draft_features,
)
from tests.predraft.features._feature_helpers import (
    _setup_team,
)

# --- build_pre_draft_features expanded feature set ---


# --- Feature-dict parity (pins the pruned feature dict byte-for-byte) ---

# Captured from build_pre_draft_features on the _setup_team(teamA)/_setup_team(teamB)
# fixture at as_of_date=2025-03-15; a deliberate change means re-capturing this dict.
EXPECTED_PARITY_FEATURES = {
    "diff_avg_game_duration": 0.0,
    "diff_controlwardsbought": 0.0,
    "diff_csdiffat15": 0.0,
    "diff_cspm": 0.0,
    "diff_damageshare": 0.0,
    "diff_dpm": 0.0,
    "diff_earnedgoldshare": 0.0,
    "diff_first_blood_rate": 0.0,
    "diff_first_dragon_rate": 0.0,
    "diff_first_herald_rate": 0.0,
    "diff_gamedur_variance": 0.0,
    "diff_games_together": 0,
    "diff_golddiffat15": 0.0,
    "diff_kda": 0.0,
    "diff_killparticipation": 0.0,
    "diff_kills_per_min": 0.0,
    "diff_kpm_variance": 0.0,
    "diff_lineup_win_rate": 0.0,
    "diff_tower_rate": 0.0,
    "diff_vspm": 0.0,
    "diff_weighted_win_rate": 0.0,
    "diff_xpdiffat15": 0.0,
    # Extended features (symmetric fixture -> diffs are 0)
    "diff_games_7d": 0,
    "diff_roster_days_stable": 0,
    "diff_roster_total_games": 0,
    "diff_blue_wr": 0.0,
    "diff_red_wr": 0.0,
    "diff_avg_gold_diff_end": 0.0,
    "diff_close_game_rate": 0.0,
    "diff_unique_champs": 0,
    "diff_avg_champ_diversity": 0.0,
    "diff_early_game_index": 0.0,
    "diff_late_game_index": 0.0,
    # EMA + Bayesian + draft
    "diff_ema_wr_slow": 0.0,
    "diff_bayesian_wr": 0.0,
    "diff_draft_score_avg": 0.0,
    "diff_upset_rate": 0.0,
    # Match-level contextual features (is_playoffs is pruned dead weight)
    "format_bo": 3.0,
    "league_tier_avg": 3.0,
    "is_cross_tier": 0.0,
    # Series state: map one of a Bo3
    "t1_series_wins": 0.0,
    "t2_series_wins": 0.0,
    "diff_series_wins": 0.0,
    # H2H features (fixture has 10 games: 5 teamA wins + 5 teamB wins)
    "h2h_games": 10,
    "h2h_win_rate": 0.5,
    "h2h_recency_wr": 0.5,
    "h2h_days_since": 10,
    "is_t1_blue_side": 1.0,
    "t1_days_since_last": 10.0,
    "t1_performance_slope": 0.0,
    "t1_win_rate_20": 0.5,
    "t2_days_since_last": 10.0,
    "t2_performance_slope": 0.0,
    "t2_win_rate_20": 0.5,
}


def test_feature_dict_parity(tmp_db):
    """Lock the full pre-draft feature dict to the recorded symmetric-fixture snapshot."""
    tmp_db.row_factory = sqlite3.Row
    create_schema(tmp_db)
    _setup_team(tmp_db, "teamA")
    _setup_team(tmp_db, "teamB")
    tmp_db.commit()

    features = build_pre_draft_features(
        tmp_db,
        team1_id="teamA",
        team2_id="teamB",
        match_format="Bo3",
        stage="playoff",
        as_of=date(2025, 3, 15),
    )

    assert set(features) == set(EXPECTED_PARITY_FEATURES), (
        f"Key set drift. Extra: {set(features) - set(EXPECTED_PARITY_FEATURES)}, "
        f"missing: {set(EXPECTED_PARITY_FEATURES) - set(features)}"
    )
    for key, expected in EXPECTED_PARITY_FEATURES.items():
        assert features[key] == pytest.approx(expected, abs=1e-9), (
            f"Mismatch on {key!r}: got {features[key]!r}, expected {expected!r}"
        )


def test_feature_dict_parity_asymmetric(tmp_db):
    """Exercise the pruning and diffing arithmetic on asymmetric inputs."""
    tmp_db.row_factory = sqlite3.Row
    create_schema(tmp_db)
    _setup_team(tmp_db, "teamA")  # teamA: 5 wins recorded against teamB
    tmp_db.commit()

    features = build_pre_draft_features(
        tmp_db,
        team1_id="teamA",
        team2_id="teamB",
        match_format="Bo3",
        stage="playoff",
        as_of=date(2025, 3, 15),
    )

    # Form: teamA 5-0, teamB 0-5. diff_win_rate_5 is pruned; diff_weighted_win_rate
    # carries the same signal.
    assert "diff_win_rate_5" not in features
    assert features["diff_weighted_win_rate"] == pytest.approx(1.0)

    # Pruning still in effect: individuals for diff-only bases are gone
    assert "t1_kda" not in features
    assert "t2_dragon_rate" not in features
    # Dead-weight set is honored.
    assert "diff_win_rate_20" not in features
