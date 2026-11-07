"""Tests for the rebuild-schema guard: a database missing the current schema is refused."""

from __future__ import annotations

import sqlite3

import pytest

from league_predictor.db.rebuild_guard import require_rebuild_schema


def test_passes_on_the_rebuild_schema(rebuild_db):
    require_rebuild_schema(
        rebuild_db, "models", "predictions", "games", purpose="test run"
    )  # no raise


def test_refuses_a_pre_rebuild_db_naming_purpose_and_remedy():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE matches (match_id TEXT)")  # old-schema shape

    with pytest.raises(RuntimeError) as exc:
        require_rebuild_schema(conn, "models", "predictions", purpose="eval persistence")

    msg = str(exc.value)
    assert "eval persistence" in msg  # names the refused caller
    assert "models, predictions" in msg  # names what is missing
    assert "current-schema database" in msg  # names the remedy
    conn.close()
