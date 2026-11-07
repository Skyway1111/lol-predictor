"""Meta-frequency prior over opposite-side same-role responses, for scoring blind picks."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from league_predictor.draft_grading.grader.corpus_memo import memo


def enemy_response_distribution(
    db: sqlite3.Connection,
    role: str,
    as_of: datetime,
    *,
    window_days: int = 30,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Top-K most-picked champions at ``role`` in ``[as_of - window_days, as_of)``.

    Returns:
        (champion, weight) pairs whose weights sum to 1; empty when the window is empty.
    """
    start = as_of - timedelta(days=window_days)
    lo = start.strftime("%Y-%m-%dT%H:%M:%S")
    hi = as_of.strftime("%Y-%m-%dT%H:%M:%S")
    # Raw string range on s.scheduled_ts keeps idx_series_scheduled usable; wrapping the
    # column in datetime() walks every entry. Lower bound inclusive, upper exclusive.
    rows = db.execute(
        """
        SELECT gp.champion_id, COUNT(*) AS n
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE gp.role = ?
          AND s.scheduled_ts >= ?
          AND s.scheduled_ts <  ?
        GROUP BY gp.champion_id
        ORDER BY n DESC, gp.champion_id
        LIMIT ?
        """,
        (role, lo, hi, top_k),
    ).fetchall()
    if not rows:
        return []
    total = sum(n for _, n in rows)
    return [(champ, n / total) for champ, n in rows]


def cached_enemy_response_distribution(
    db: sqlite3.Connection,
    role: str,
    as_of: datetime,
    *,
    window_days: int = 30,
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """Day-keyed memoization of ``enemy_response_distribution``; keyed without the connection."""
    return memo(
        db,
        ("enemy_response", role, as_of, window_days, top_k),
        lambda: enemy_response_distribution(db, role, as_of, window_days=window_days, top_k=top_k),
    )
