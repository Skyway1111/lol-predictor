"""Tests ``_LineupIndex.days_since_last_game``: raw days, ``inf`` when there is no prior game."""

from __future__ import annotations

import math
import sqlite3
from datetime import date

from league_predictor.predraft.features.lineup import LineupFamily

_ROLES = ("top", "jng", "mid", "bot", "sup")


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from league_predictor.db.schema import apply_current_schema

    apply_current_schema(conn)
    return conn


def _facts_game(conn, mid, d, a, b, winner, tournament="LCK", gamelength=1800):
    ta, tb = sorted((a, b))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, status) "
        "VALUES (?, ?, ?, ?, ?, 'completed')",
        (f"ser_{mid}", tournament, ta, tb, f"{d}T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
        "VALUES (?, ?, 1, ?, ?)",
        (mid, f"ser_{mid}", winner, gamelength),
    )


def _game(conn, n: int, d: date, a: str, b: str) -> None:
    _facts_game(conn, f"m{n}", d.isoformat(), a, b, a)
    for tid in (a, b):
        for role in _ROLES:
            conn.execute(
                "INSERT INTO game_players (game_id, team_id, player_id, role, "
                "kills, deaths, assists) VALUES (?,?,?,?,?,?,?)",
                (f"m{n}", tid, f"{tid}_{role}", role, 5, 5, 10),
            )


def _index():
    conn = _db()
    _game(conn, 1, date(2026, 2, 1), "AAA", "BBB")
    _game(conn, 2, date(2026, 2, 11), "AAA", "BBB")  # AAA/BBB last play 2026-02-11
    return LineupFamily().build(conn, through=date(2026, 5, 1), query_dates=[])


def test_days_since_counts_to_the_last_prior_game() -> None:
    idx = _index()
    assert idx.days_since_last_game("AAA", date(2026, 3, 28)) == 45  # 2026-02-11 -> 03-28


def test_a_game_on_the_as_of_day_does_not_count() -> None:
    """Pre-draft predicts before the match: a game dated on ``as_of`` is excluded from the gap."""
    idx = _index()
    assert idx.days_since_last_game("AAA", date(2026, 2, 11)) == 10  # back to 2026-02-01


def test_no_prior_game_is_infinite() -> None:
    idx = _index()
    assert idx.days_since_last_game("AAA", date(2026, 2, 1)) == math.inf  # first game ever
    assert idx.days_since_last_game("NOBODY", date(2026, 3, 1)) == math.inf
