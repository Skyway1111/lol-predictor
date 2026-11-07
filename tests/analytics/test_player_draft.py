"""Per-axis aggregator tests for analytics.player_draft — one pinned property each."""

from __future__ import annotations

import pytest

from league_predictor.analytics.player_draft import (
    AXIS_WEIGHTS,
    MIN_GAMES,
    blend_overall_quality,
    compute_consistency_raw,
    compute_counter_rate_raw,
    compute_pool_depth_raw,
    mean_or_none,
    percentile_rank_within_role,
)


def test_pool_depth_caps_per_champion_at_one() -> None:
    """A champ with 100 games contributes 1.0, not 33."""
    assert compute_pool_depth_raw([("Aatrox", 100)]) == 1.0


def test_pool_depth_sums_partial_contributions() -> None:
    """Two champs at 2 games each contribute 2/3 each → total 4/3."""
    out = compute_pool_depth_raw([("Aatrox", 2), ("Camille", 2)])
    assert out == pytest.approx(4 / 3)


def test_pool_depth_empty_returns_zero() -> None:
    assert compute_pool_depth_raw([]) == 0.0


def test_consistency_shrinks_with_low_sample() -> None:
    """A 10-game player's shrunk stddev sits between raw and the role prior."""
    n = 10
    pillars = {"lane": [0.5] * n, "combat": [0.5] * n, "vision": [0.5] * n}
    # All-identical → raw stddev = 0, shrunk toward role_mean_stddev = 1.0 with
    # shrink_weight = n/(n+SHRINK_K) = 10/30 = 1/3, so out = 1/3*0 + 2/3*1 = 2/3.
    out = compute_consistency_raw(pillars, role_mean_stddev=1.0)
    assert out == pytest.approx(2 / 3)


def test_consistency_null_below_min_games() -> None:
    """With (MIN_GAMES - 1) games, return None even with valid pillar shape."""
    n = MIN_GAMES - 1
    pillars = {"lane": [0.5] * n, "combat": [0.5] * n, "vision": [0.5] * n}
    assert compute_consistency_raw(pillars, role_mean_stddev=1.0) is None


def test_consistency_uses_input_length_for_n_games() -> None:
    """The function infers n_games from the pillar lists; 8 games (< MIN_GAMES) returns None."""
    n = 8
    pillars = {"lane": [0.5] * n, "combat": [0.5] * n}
    assert compute_consistency_raw(pillars, role_mean_stddev=1.0) is None


def test_consistency_rejects_unequal_pillar_lengths() -> None:
    """Pillars must have the same length — different lengths is a contract bug."""
    pillars = {"lane": [0.5] * 10, "combat": [0.5] * 9}
    with pytest.raises(ValueError, match="equal length"):
        compute_consistency_raw(pillars, role_mean_stddev=1.0)


def test_consistency_at_high_n_approaches_raw() -> None:
    """At n >> SHRINK_K, output is much closer to raw than to the role prior."""
    n = 1000
    # Alternating pairs → pstdev = 1.0 for both pillars → raw mean stddev = 1.0.
    pillars = {"a": [0.0, 2.0] * (n // 2), "b": [1.0, 3.0] * (n // 2)}
    raw, prior = 1.0, 10.0
    out = compute_consistency_raw(pillars, role_mean_stddev=prior)
    assert out is not None
    # At n=1000, shrink_weight = 1000/1020 ≈ 0.98 → out lies near raw.
    assert abs(out - raw) < abs(out - prior)


def test_mean_null_below_min_games() -> None:
    assert mean_or_none([0.5] * (MIN_GAMES - 1)) is None


def test_mean_at_min_games() -> None:
    assert mean_or_none([0.5] * MIN_GAMES) == pytest.approx(0.5)


def test_counter_rate_null_below_min_games() -> None:
    assert compute_counter_rate_raw(n_counter=5, n_total=MIN_GAMES - 1) is None


def test_counter_rate_basic() -> None:
    assert compute_counter_rate_raw(n_counter=4, n_total=10) == 0.4


def test_blend_overall_quality_renormalizes_when_blind_missing() -> None:
    """blind_quality=None: the remaining weights re-sum to 0.70 and the blend renormalizes."""
    axes = {
        "blind_quality": None,
        "counter_quality": 70.0,
        "pool_depth": 60.0,
        "consistency": 50.0,
        "counter_rate": 30.0,  # weight 0, shouldn't move the headline
    }
    out = blend_overall_quality(axes)
    expected = (0.30 * 70 + 0.20 * 60 + 0.20 * 50) / 0.70
    assert out == pytest.approx(expected)


def test_blend_overall_quality_null_when_every_weighted_axis_is_null() -> None:
    """Counter Rate carries weight 0, so it alone cannot produce a headline."""
    axes = {
        "blind_quality": None,
        "counter_quality": None,
        "pool_depth": None,
        "consistency": None,
        "counter_rate": 30.0,
    }
    assert blend_overall_quality(axes) is None


def test_counter_rate_weight_is_zero_in_overall_quality() -> None:
    """counter_rate never moves overall_quality — Counter Rate is display-only."""
    base = {
        "blind_quality": 70.0,
        "counter_quality": 60.0,
        "pool_depth": 50.0,
        "consistency": 40.0,
    }
    out_zero = blend_overall_quality({**base, "counter_rate": 0.0})
    out_high = blend_overall_quality({**base, "counter_rate": 100.0})
    assert out_zero == out_high


def test_axis_weights_sum_to_one_minus_counter_rate() -> None:
    """Sanity check: 0.30 + 0.30 + 0.20 + 0.20 = 1.0; counter_rate 0.0."""
    assert sum(w for k, w in AXIS_WEIGHTS.items() if k != "counter_rate") == pytest.approx(1.0)
    assert AXIS_WEIGHTS["counter_rate"] == 0.0


def test_percentile_rank_within_role_handles_null_inputs() -> None:
    """NULL inputs pass through as NULL; the percentile covers non-NULL same-role peers."""
    raws = {"p1": 10.0, "p2": 20.0, "p3": None, "p4": 30.0}
    roles = {"p1": "top", "p2": "top", "p3": "top", "p4": "top"}
    out = percentile_rank_within_role(raws, roles)
    assert out["p3"] is None
    # p1, p2, p4 spread evenly: ranks 1, 2, 3 over 3 → 0, 50, 100.
    assert out["p1"] == pytest.approx(0.0)
    assert out["p2"] == pytest.approx(50.0)
    assert out["p4"] == pytest.approx(100.0)


def test_percentile_rank_inverts_for_consistency() -> None:
    """invert=True: lower raw value gets higher percentile (lower stddev = more consistent)."""
    raws = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
    roles = {"p1": "top", "p2": "top", "p3": "top"}
    out = percentile_rank_within_role(raws, roles, invert=True)
    # p1 has lowest raw → highest percentile (100); p3 has highest raw → lowest (0).
    assert out["p1"] == pytest.approx(100.0)
    assert out["p2"] == pytest.approx(50.0)
    assert out["p3"] == pytest.approx(0.0)


def test_percentile_rank_within_role_separates_roles() -> None:
    """Players are ranked against same-role peers, not the global pool."""
    raws = {"top1": 100.0, "top2": 200.0, "mid1": 50.0, "mid2": 51.0}
    roles = {"top1": "top", "top2": "top", "mid1": "mid", "mid2": "mid"}
    out = percentile_rank_within_role(raws, roles)
    # Within each role: lower → 0, higher → 100. mid1 (50) < mid2 (51) → 0 / 100.
    assert out["top1"] == pytest.approx(0.0)
    assert out["top2"] == pytest.approx(100.0)
    assert out["mid1"] == pytest.approx(0.0)
    assert out["mid2"] == pytest.approx(100.0)


def test_percentile_rank_role_with_one_player_returns_50() -> None:
    """Edge case: a lone player in their role gets percentile 50 — no peers, no signal."""
    raws = {"solo": 42.0}
    roles = {"solo": "top"}
    out = percentile_rank_within_role(raws, roles)
    assert out["solo"] == pytest.approx(50.0)


def test_percentile_rank_handles_ties_with_average_rank() -> None:
    """Tied raw values get the average of their ranks (standard convention)."""
    raws = {"p1": 1.0, "p2": 2.0, "p3": 2.0, "p4": 3.0}
    roles = dict.fromkeys(raws, "top")
    out = percentile_rank_within_role(raws, roles)
    # Ranks: p1=1, p2=2.5, p3=2.5, p4=4 over n=4. Percentile = (r-1)/(n-1)*100.
    assert out["p1"] == pytest.approx(0.0)
    assert out["p2"] == pytest.approx((2.5 - 1) / 3 * 100)
    assert out["p3"] == pytest.approx((2.5 - 1) / 3 * 100)
    assert out["p4"] == pytest.approx(100.0)
