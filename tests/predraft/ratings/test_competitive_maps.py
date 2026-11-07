"""Tests for load_competitive_maps on a tiny in-memory DB under the real schema."""

from __future__ import annotations

import sqlite3
from datetime import date

from league_predictor.predraft.ratings.competitive_maps import load_competitive_maps


def _db() -> sqlite3.Connection:
    """In-memory facts DB: one clean game, one remake (short)."""
    from league_predictor.db.schema import apply_current_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    for sid, ts, t1, t2, w, dur in (
        ("s1", "2026-05-01T10:00:00+00:00", "TES", "BLG", "TES", 1800),
        ("s2", "2026-05-02T10:00:00+00:00", "T1", "GEN", "GEN", 120),
    ):  # remake
        a, b = sorted((t1, t2))
        conn.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
            "status) VALUES (?, 'LCK', ?, ?, ?, 'completed')",
            (sid, a, b, ts),
        )
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
            "VALUES (?, ?, 1, ?, ?)",
            (f"m{sid[1]}", sid, w, dur),
        )
    conn.executemany(
        "INSERT INTO game_players (game_id, team_id, player_id, kills, deaths, assists) "
        "VALUES (?,?,?,?,?,?)",
        [
            ("m1", "TES", "p1", 5, 2, 8),
            ("m1", "BLG", "p2", 2, 5, 4),
            ("m2", "T1", "p3", 1, 1, 1),
            ("m2", "GEN", "p4", 1, 1, 1),
        ],
    )
    conn.commit()
    return conn


def test_load_competitive_maps_returns_clean_maps_in_reference_orientation():
    maps = load_competitive_maps(_db(), through=date(2026, 6, 1))
    assert len(maps) == 1  # the remake (m2) is dropped
    d, team_a, team_b, won_a = maps[0]
    assert (team_a, team_b) == ("BLG", "TES")  # sorted orientation
    assert won_a == 0  # TES (team_b) won, so team_a lost
    assert d.day == date(2026, 5, 1)


def test_load_competitive_maps_through_is_exclusive():
    maps = load_competitive_maps(_db(), through=date(2026, 5, 1))  # excludes 2026-05-01
    assert maps == []
