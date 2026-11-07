"""Tests for the feature families registry + the build_features assembler."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

import pytest

from league_predictor.asof import AsOf
from league_predictor.predraft.features import FEATURE_FAMILIES, build_features
from league_predictor.predraft.features.families.recency import RecencyFamily
from league_predictor.predraft.ratings.skill_filter import derived_feature_names

# Four teams with FIXED rosters (stable lineups); AAA strongest.
_ROSTERS = {
    t: {role: f"{t}_{role}" for role in ("top", "jng", "mid", "bot", "sup")}
    for t in ("AAA", "BBB", "CCC", "DDD")
}
_SKILL = {"AAA": 3.0, "BBB": 1.0, "CCC": -1.0, "DDD": -3.0}


def _form_db(
    *, through: date = date(2026, 4, 15), days: int = 90, seed: int = 0
) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from league_predictor.db.schema import apply_current_schema

    apply_current_schema(conn)
    rng = random.Random(seed)
    teams = list(_ROSTERS)
    d, n = through - timedelta(days=days), 0
    while d < through:
        for _ in range(2):
            a, b = rng.sample(teams, 2)
            p_a = 1.0 / (1.0 + 10 ** (-(_SKILL[a] - _SKILL[b]) / 4.0))
            winner = a if rng.random() < p_a else b
            mid = f"m{n}"
            ta, tb = sorted((a, b))
            conn.execute(
                "INSERT INTO series (series_id, league_id, team_a, team_b, "
                "scheduled_ts, status) VALUES (?, 'LCK', ?, ?, ?, 'completed')",
                (f"ser_{mid}", ta, tb, f"{d.isoformat()}T10:00:00+00:00"),
            )
            conn.execute(
                "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
                "VALUES (?, ?, 1, ?, 1800)",
                (mid, f"ser_{mid}", winner),
            )
            for tid in (a, b):
                for role, pid in _ROSTERS[tid].items():
                    conn.execute(
                        "INSERT INTO game_players (game_id, team_id, player_id, "
                        "role, kills, deaths, assists) VALUES (?,?,?,?,?,?,?)",
                        (mid, tid, pid, role, 5, 5, 10),
                    )
            n += 1
        d += timedelta(days=1)
    conn.commit()
    return conn


_AS_OF = date(2026, 4, 14)
_Q = AsOf.of(_AS_OF)


def test_registry_is_exactly_the_served_four_families():
    """The registry equals serving reality: the four served families and no other rating family."""
    assert set(FEATURE_FAMILIES) == {"learned_rating", "original", "recency", "lineup"}
    assert "rating" not in FEATURE_FAMILIES
    assert "online_elo_prod" not in FEATURE_FAMILIES


def test_latent_strength_factory_is_torchfree_and_emits_the_20_derived_columns():
    """Registry construction stays torch-free and yields the 20 ``derived_feature_names(2, 4)``."""
    fam = FEATURE_FAMILIES["learned_rating"]()
    assert fam.name == "learned_rating"
    assert fam.feature_names == derived_feature_names(2, 4)
    assert len(fam.feature_names) == 20
    assert fam.categorical_names == []


def test_recency_family_emits_antisymmetric_signed_diffs():
    fam = RecencyFamily()
    assert fam.name == "recency"
    index = fam.build(_form_db(), through=date(2026, 4, 15), query_dates=[_AS_OF])
    strong_vs_weak = index.features("AAA", "DDD", _Q)
    weak_vs_strong = index.features("DDD", "AAA", _Q)
    assert set(strong_vs_weak) == set(fam.feature_names)
    assert strong_vs_weak["recency_winrate20_diff"] > 0  # the stronger team is favoured
    assert strong_vs_weak["recency_winrate20_diff"] == pytest.approx(
        -weak_vs_strong["recency_winrate20_diff"]
    )


def test_build_features_unions_enabled_families():
    index = RecencyFamily().build(_form_db(), through=date(2026, 4, 15), query_dates=[_AS_OF])
    out = build_features("AAA", "DDD", _AS_OF, indexes={"recency": index}, families=["recency"])
    assert set(out) == {"recency_winrate20_diff", "recency_ema_diff", "recency_momentum_diff"}
