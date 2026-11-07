"""Player Impact schema constraints."""

from __future__ import annotations

import sqlite3

import pytest

from league_predictor.db.schema import apply_current_schema

# --- (a) fresh schema ------------------------------------------------------


def test_player_impact_scores_constraints_bite():
    """player_impact_scores enforces the asof GLOB, the model FK, and the natural PK."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    apply_current_schema(conn)
    conn.execute("INSERT INTO players (player_id) VALUES ('p1')")
    conn.execute("INSERT INTO models (model_id, kind) VALUES ('impact:abc', 'impact')")

    ins = (
        "INSERT INTO player_impact_scores (player_id, model_id, asof_date, role, impact) "
        "VALUES (?, ?, ?, ?, ?)"
    )
    # bad asof format rejected
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(ins, ("p1", "impact:abc", "20260101", "mid", 1.2))
    # unknown model_id FK rejected
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(ins, ("p1", "impact:zzz", "2026-01-01", "mid", 1.2))
    # valid row accepted
    conn.execute(ins, ("p1", "impact:abc", "2026-01-01", "mid", 1.2))
    # duplicate natural key (player_id, model_id, asof_date) rejected
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(ins, ("p1", "impact:abc", "2026-01-01", "top", 3.4))
    conn.close()
