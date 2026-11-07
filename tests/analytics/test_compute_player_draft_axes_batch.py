"""Integration tests for compute_player_draft_axes_batch — the seam wiring the aggregators."""

from __future__ import annotations

import pytest

from league_predictor.analytics.player_draft import (
    compute_player_draft_axes_batch,
)


def _make_pick_rows(
    n_blind: int, n_counter: int, blind_pct: float, counter_pct: float
) -> list[dict]:
    """Build pick_score row dicts with the requested counts and percentile means."""
    rows = [{"is_blind": True, "percentile": blind_pct} for _ in range(n_blind)]
    rows += [{"is_blind": False, "percentile": counter_pct} for _ in range(n_counter)]
    return rows


def test_end_to_end_on_fixture_dicts() -> None:
    """Three players across two roles: top x 2 with full data, mid x 1 alone in role."""
    roles = {"p_top_a": "top", "p_top_b": "top", "p_mid": "mid"}
    season_champions = {
        "p_top_a": [("Aatrox", 5), ("Camille", 3), ("Renekton", 1)],  # depth 1+1+1/3 = 7/3
        "p_top_b": [("Sett", 10)],  # depth 1
        "p_mid": [("Ahri", 4), ("Syndra", 4), ("Azir", 4)],  # depth 3
    }
    per_game_pillars = {
        "p_top_a": {
            "lane": [1.0, 2.0] * 8,
            "combat": [1.0, 2.0] * 8,
            "vision": [1.0, 2.0] * 8,
        },  # 16 games, raw stddev 0.5
        "p_top_b": {
            "lane": [1.5] * 16,
            "combat": [1.5] * 16,
            "vision": [1.5] * 16,
        },  # 16 games, raw stddev 0
        "p_mid": {"lane": [1.0, 3.0] * 8, "combat": [1.0, 3.0] * 8},  # 16 games, raw stddev 1.0
    }
    pick_rows = {
        "p_top_a": _make_pick_rows(n_blind=12, n_counter=5, blind_pct=0.7, counter_pct=0.6),
        "p_top_b": _make_pick_rows(n_blind=10, n_counter=10, blind_pct=0.5, counter_pct=0.5),
        "p_mid": _make_pick_rows(n_blind=15, n_counter=15, blind_pct=0.8, counter_pct=0.9),
    }
    role_stddev = {"top": 0.5, "mid": 1.0}

    out = compute_player_draft_axes_batch(
        roles_by_player=roles,
        season_champions_by_player=season_champions,
        per_game_pillars_by_player=per_game_pillars,
        pick_rows_by_player=pick_rows,
        role_mean_stddev_by_role=role_stddev,
    )

    # All three players present.
    assert set(out) == {"p_top_a", "p_top_b", "p_mid"}

    # Gameplay passes through directly.

    # Within-role percentile: in 'top', p_top_a has deeper pool than p_top_b.
    assert out["p_top_a"].pool_depth == 100.0
    assert out["p_top_b"].pool_depth == 0.0
    # 'mid' has one player → 50.
    assert out["p_mid"].pool_depth == 50.0

    # Blind quality: p_top_a has n_blind=12 and p_top_b n_blind=10, both ≥ MIN_GAMES
    # → both defined.
    assert out["p_top_a"].blind_quality is not None
    assert out["p_top_b"].blind_quality is not None

    # Counter quality: p_top_a counter n=5 < MIN_GAMES → NULL.
    assert out["p_top_a"].counter_quality is None
    # p_top_b counter n=10 ≥ MIN_GAMES → defined. With only one defined counter_quality
    # in role 'top', the lone-player convention returns 50.
    assert out["p_top_b"].counter_quality == 50.0

    # Counter rate: p_top_a = 5 / 17, p_top_b = 10 / 20 = 0.5. Both ≥ MIN_GAMES totals.
    # Within role 'top': p_top_b higher → 100; p_top_a lower → 0.
    assert out["p_top_b"].counter_rate == 100.0
    assert out["p_top_a"].counter_rate == 0.0

    # Sample counts surface honestly.
    assert out["p_top_a"].samples["blind_quality"] == 12
    assert out["p_top_a"].samples["counter_quality"] == 5
    assert out["p_top_a"].samples["counter_rate"] == 17


def test_player_without_scored_picks_is_dropped() -> None:
    """The radar reads drafted behavior: no pick-scores axis, no row at all."""
    out = compute_player_draft_axes_batch(
        roles_by_player={"p1": "top", "p2": "top"},
        season_champions_by_player={"p1": [("X", 10)], "p2": [("Y", 10)]},
        per_game_pillars_by_player={},
        pick_rows_by_player={"p1": _make_pick_rows(15, 15, 0.5, 0.5)},
        role_mean_stddev_by_role={},
    )
    assert "p1" in out
    assert "p2" not in out


def test_pool_and_consistency_alone_do_not_produce_a_row() -> None:
    """Roster shape without any scored pick is not a draft profile."""
    out = compute_player_draft_axes_batch(
        roles_by_player={"p": "top"},
        season_champions_by_player={"p": [("X", 10)] * 5},
        per_game_pillars_by_player={"p": {"a": [1.0] * 15}},
        pick_rows_by_player={},
        role_mean_stddev_by_role={"top": 1.0},
    )
    assert out == {}


def test_role_with_one_player_returns_50_percentile() -> None:
    """Edge case: a lone player in their role gets percentile rank 50, by convention."""
    roles = {"solo": "top"}
    season_champions = {"solo": [("X", 10), ("Y", 10), ("Z", 10)]}
    pillars = {"solo": {"a": [1.0] * 15}}
    pick_rows = {"solo": _make_pick_rows(15, 15, 0.5, 0.5)}
    out = compute_player_draft_axes_batch(
        roles_by_player=roles,
        season_champions_by_player=season_champions,
        per_game_pillars_by_player=pillars,
        pick_rows_by_player=pick_rows,
        role_mean_stddev_by_role={"top": 1.0},
    )
    # The five "soft" axes all return 50.
    a = out["solo"]
    assert a.pool_depth == 50.0
    assert a.blind_quality == 50.0
    assert a.counter_quality == 50.0
    assert a.counter_rate == 50.0
    assert a.consistency == 50.0


def test_overall_quality_matches_blend_formula() -> None:
    """Overall = 0.30*blind + 0.30*counter + 0.20*pool + 0.20*consistency."""
    roles = {"a": "top", "b": "top"}  # need 2 so percentile-rank doesn't degenerate to 50
    season_champions = {"a": [("X", 10)] * 10, "b": [("Y", 1)]}
    pillars = {
        "a": {"p1": [1.0, 2.0] * 8, "p2": [1.0, 2.0] * 8},
        "b": {"p1": [1.0] * 16, "p2": [1.0] * 16},
    }
    pick_rows = {
        "a": _make_pick_rows(15, 15, 1.0, 1.0),
        "b": _make_pick_rows(15, 15, 0.0, 0.0),
    }
    out = compute_player_draft_axes_batch(
        roles_by_player=roles,
        season_champions_by_player=season_champions,
        per_game_pillars_by_player=pillars,
        pick_rows_by_player=pick_rows,
        role_mean_stddev_by_role={"top": 0.5},
    )
    # Player "a" should max out the draft-derived axes.
    a = out["a"]
    assert a.pool_depth == 100.0
    assert a.blind_quality == 100.0
    assert a.counter_quality == 100.0
    expected = (
        0.30 * a.blind_quality
        + 0.30 * a.counter_quality
        + 0.20 * a.pool_depth
        + 0.20 * a.consistency
    )
    assert a.overall_quality == pytest.approx(expected)
