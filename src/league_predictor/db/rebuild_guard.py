"""Shared guard: derived-table writers refuse a database missing the current schema."""

from __future__ import annotations

import sqlite3


def require_rebuild_schema(conn: sqlite3.Connection, *tables: str, purpose: str) -> None:
    """Raise ``RuntimeError`` unless every named table exists; ``purpose`` names the caller."""
    have = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    missing = [t for t in tables if t not in have]
    if missing:
        raise RuntimeError(
            f"{purpose} requires the current DB schema; missing table(s): "
            f"{', '.join(missing)}. This is not a current-schema database — "
            f"point at the production DB or run `lol-predict refresh` to build one."
        )
