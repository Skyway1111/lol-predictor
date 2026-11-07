"""Database connections, schema bootstrap, and shared facts reads."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from league_predictor.db.facts import facts_generation, latest_played_date
from league_predictor.db.schema import apply_current_schema as create_schema


def get_connection(db_path: Path | str, *, check_same_thread: bool = True) -> sqlite3.Connection:
    """Get a SQLite connection, creating the DB file if needed."""
    # check_same_thread=False lets another thread use it; the caller then serializes access.
    conn = sqlite3.connect(str(db_path), check_same_thread=check_same_thread)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Pinned explicitly (the DB-API default is also 5s) so a write-write conflict between
    # overlapping writers retries instead of raising SQLITE_BUSY.
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


ensure_database_ready = create_schema


__all__ = [
    "create_schema",
    "ensure_database_ready",
    "facts_generation",
    "get_connection",
    "latest_played_date",
]
