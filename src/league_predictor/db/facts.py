"""Cheap facts invalidation and the shared played-date bound."""

from __future__ import annotations

import sqlite3
from datetime import date


def facts_generation(conn: sqlite3.Connection) -> int:
    """Generation visible to this connection, advanced transactionally by facts writes."""
    return conn.execute("SELECT generation FROM facts_state WHERE id = 1").fetchone()[0]


def cache_revision(conn: sqlite3.Connection) -> tuple[int, int, int | None] | None:
    """Visible facts revision; legacy transactions without generation triggers cannot be cached."""
    try:
        generation = facts_generation(conn)
    except sqlite3.OperationalError as exc:
        if str(exc) != "no such table: facts_state":
            raise
        if conn.in_transaction:
            return None
        generation = None
    return conn.total_changes, conn.execute("PRAGMA data_version").fetchone()[0], generation


def latest_played_date(conn: sqlite3.Connection, *, through: date | None = None) -> date | None:
    """Latest series day containing a map with a recorded winner, or None."""
    value = conn.execute(
        "SELECT MAX(date(s.scheduled_ts)) FROM series s "
        "JOIN games g ON g.series_id = s.series_id WHERE g.winner IS NOT NULL "
        "AND (? IS NULL OR date(s.scheduled_ts) <= ?)",
        (through.isoformat() if through else None,) * 2,
    ).fetchone()[0]
    return date.fromisoformat(value) if value else None
