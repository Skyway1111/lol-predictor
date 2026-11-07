"""Native persisted-corpus facts digest."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from league_predictor.db import facts_prefix
from league_predictor.predraft.pipeline.fit_corpus_store import (
    _facts_prefix_digests,
)


def _facts_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE series (series_id TEXT PRIMARY KEY, league_id TEXT, team_a TEXT, "
        "team_b TEXT, scheduled_ts TEXT, best_of INTEGER)"
    )
    conn.execute(
        "CREATE TABLE games (game_id TEXT PRIMARY KEY, series_id TEXT, game_no INTEGER, "
        "winner TEXT)"
    )
    conn.execute("INSERT INTO series VALUES ('s1','LCK','A','B','2026-01-01',3)")
    conn.execute("INSERT INTO games VALUES ('g1','s1',1,'A')")
    conn.commit()
    return conn


def test_native_digest_detects_content_and_schema_changes(tmp_path):
    conn = _facts_db(tmp_path / "facts.db")
    cutoff = date(2026, 2, 1)
    try:
        before = _facts_prefix_digests(conn, [cutoff])[cutoff]
        conn.execute("UPDATE games SET winner='B' WHERE game_id='g1'")
        conn.commit()
        corrected = _facts_prefix_digests(conn, [cutoff])[cutoff]
        assert corrected != before

        conn.execute("ALTER TABLE games ADD COLUMN unexpected TEXT")
        conn.commit()
        changed_schema = _facts_prefix_digests(conn, [cutoff])[cutoff]
        assert changed_schema != corrected
    finally:
        conn.close()


def test_native_digest_is_prefix_stable_for_future_appends(tmp_path):
    conn = _facts_db(tmp_path / "facts.db")
    cutoff = date(2026, 2, 1)
    try:
        before = _facts_prefix_digests(conn, [cutoff])[cutoff]
        conn.execute("INSERT INTO series VALUES ('s2','LCK','A','B','2027-01-01',3)")
        conn.execute("INSERT INTO games VALUES ('g2','s2',1,'A')")
        conn.commit()
        assert _facts_prefix_digests(conn, [cutoff])[cutoff] == before
    finally:
        conn.close()


def test_native_digest_failure_refuses_the_slow_file_backed_path(tmp_path, monkeypatch):
    conn = _facts_db(tmp_path / "facts.db")
    cutoff = date(2026, 2, 1)
    monkeypatch.setattr(
        facts_prefix,
        "native_prefix_digests",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("extension unavailable")),
    )
    try:
        with pytest.raises(RuntimeError, match=r"DuckDB.*extension unavailable"):
            _facts_prefix_digests(conn, [cutoff])
    finally:
        conn.close()
