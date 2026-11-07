"""Fixtures for ingestion-domain tests."""

from __future__ import annotations

import sqlite3

import pytest

from league_predictor.db.schema import apply_current_schema


@pytest.fixture
def schema_db() -> sqlite3.Connection:
    """In-memory DB carrying the current schema, with no foreign-key enforcement."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    yield conn
    conn.close()
