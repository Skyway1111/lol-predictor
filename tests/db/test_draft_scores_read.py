"""The draft_scores read rule: one row per (game, model), the oos walk winning the backfill."""

from __future__ import annotations

import sqlite3

from league_predictor.db.draft_scores_read import preferred_context_predicate
from league_predictor.db.schema import apply_current_schema


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    conn.executemany(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, context) VALUES (?, ?, ?, ?)",
        [
            ("g1", "draft:a", 0.40, "backfill"),
            ("g1", "draft:a", 0.70, "oos"),
            ("g2", "draft:a", 0.55, "backfill"),
            ("g1", "draft:b", 0.10, "backfill"),
        ],
    )
    return conn


def _preferred(conn: sqlite3.Connection) -> dict[tuple[str, str], float]:
    rows = conn.execute(
        "SELECT ds.game_id, ds.model_id, ds.p_blue FROM draft_scores ds "
        f"WHERE {preferred_context_predicate()}"
    ).fetchall()
    return {(r["game_id"], r["model_id"]): r["p_blue"] for r in rows}


def test_the_oos_row_wins_and_the_backfill_only_game_survives() -> None:
    """Exactly one row per (game, model): oos where the walk covers it, backfill elsewhere."""
    assert _preferred(_db()) == {
        ("g1", "draft:a"): 0.70,
        ("g2", "draft:a"): 0.55,
        ("g1", "draft:b"): 0.10,
    }


def test_the_preference_is_per_model_not_per_game() -> None:
    """One model's oos coverage never displaces another model's backfill row for that game."""
    conn = _db()
    conn.execute("DELETE FROM draft_scores WHERE model_id = 'draft:a'")
    conn.execute(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, context) "
        "VALUES ('g1', 'draft:a', 0.9, 'oos')"
    )
    assert _preferred(conn) == {("g1", "draft:a"): 0.9, ("g1", "draft:b"): 0.10}
