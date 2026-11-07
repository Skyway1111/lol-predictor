"""Tests for the canonical cleaning stage: series identity + eligibility predicates."""

from __future__ import annotations

import sqlite3

from league_predictor.predraft.cleaning import (
    competitive_map_eligible,
    enriched_matches,
    remake_reason,
)
from tests.predraft.conftest import facts_seed


def _enrich_db() -> sqlite3.Connection:
    """In-memory facts DB. Match B's team Z has no player rows."""
    conn = sqlite3.connect(":memory:")
    facts_seed(
        conn,
        [
            ("A", "2026-05-01", "X", "Y", "X", "LCK", 1800),
            ("B", "2026-05-02", "X", "Z", "X", "LCK", 1700),
        ],
        [
            ("A", "X", 6, 2, 10),
            ("A", "X", 4, 3, 10),  # team X totals: 10/5/20
            ("A", "Y", 3, 12, 8),  # team Y totals: 3/12/8
            ("B", "X", 5, 5, 5),
        ],  # match B: team Z absent
    )
    return conn


def test_enriched_matches_sums_team_kda_from_player_games():
    rows = {r["match_id"]: r for r in enriched_matches(_enrich_db())}
    a = rows["A"]
    assert (a["team1_kills"], a["team1_deaths"], a["team1_assists"]) == (10, 5, 20)
    assert (a["team2_kills"], a["team2_deaths"], a["team2_assists"]) == (3, 12, 8)


def test_enriched_matches_leaves_missing_team_stats_null():
    """A team with no player_games rows gets NULL stats -- a gap, not a zero."""
    rows = {r["match_id"]: r for r in enriched_matches(_enrich_db())}
    b = rows["B"]
    assert b["team2_kills"] is None
    assert b["team2_deaths"] is None


def _match(**over):
    """A clean, real, full-K/D/A competitive match; override fields per test."""
    m = {
        "winner": "TES",
        "tournament": "LCK",
        "gamelength": 1800,
        "team1_kills": 12,
        "team1_deaths": 8,
        "team1_assists": 25,
        "team2_kills": 8,
        "team2_deaths": 12,
        "team2_assists": 16,
    }
    m.update(over)
    return m


def _row(match_id, d, t1, t2, winner, gn, *, fmt="Bo1", stage="regular", tournament="LCK"):
    return {
        "match_id": match_id,
        "date": d,
        "team1_id": t1,
        "team2_id": t2,
        "winner": winner,
        "game_number": gn,
        "format": fmt,
        "stage": stage,
        "tournament": tournament,
    }


def test_remake_reason_flags_short_game():
    """A present gamelength under 900s is a remake artifact (instant remake/forfeit)."""
    assert remake_reason(_match(gamelength=120)) == "short_game"


def test_remake_reason_none_for_a_normal_game():
    """A full-length game with real stats is not an artifact."""
    assert remake_reason(_match()) is None


def test_remake_reason_flags_team_with_entire_kda_line_zero():
    """An all-zero present K/D/A line for one team is a remake artifact."""
    m = _match(team2_kills=0, team2_deaths=0, team2_assists=0)
    assert remake_reason(m) == "zero_team_stat"


def test_remake_reason_does_not_flag_a_lone_zero_stat():
    """A single zero stat is a real result; only the whole K/D/A line zeroed is an artifact."""
    assert remake_reason(_match(team1_deaths=0)) is None
    assert remake_reason(_match(team2_kills=0)) is None


def test_remake_reason_treats_null_stats_as_a_data_gap_not_a_remake():
    """Missing K/D/A is a data gap, not a remake: the row stays rating-eligible."""
    m = _match(team1_kills=None, team1_deaths=None, team1_assists=None)
    assert remake_reason(m) is None


def test_competitive_map_eligible_true_for_a_clean_competitive_result():
    assert competitive_map_eligible(_match()) is True


def test_competitive_map_eligible_false_for_a_remake():
    assert competitive_map_eligible(_match(gamelength=120)) is False


def test_competitive_map_eligible_false_without_a_winner():
    """An undecided/voided match must not move ratings."""
    assert competitive_map_eligible(_match(winner=None)) is False
