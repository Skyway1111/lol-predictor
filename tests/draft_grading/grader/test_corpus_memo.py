"""One corpus build per process and database state; a write invalidates it."""

from __future__ import annotations

import sqlite3

import pytest

from league_predictor.db.schema import apply_current_schema
from league_predictor.draft_grading.grader.corpus_memo import memo
from league_predictor.draft_grading.grader.pro_excess import _team_form
from tests.draft_grading.grader._facts_helpers import BLUE_COMP, RED_COMP, _game
from tests.draft_grading.grader.conftest import make_grader_test_db

BLUE_WON = True


def test_committed_updates_on_another_connection_invalidate_the_memo(tmp_path):
    path = tmp_path / "facts.db"
    with sqlite3.connect(path) as writer, sqlite3.connect(path) as reader:
        writer.executescript(
            "PRAGMA journal_mode=WAL; CREATE TABLE facts (value); INSERT INTO facts VALUES (1)"
        )

        def read():
            return reader.execute("SELECT value FROM facts").fetchone()[0]

        assert memo(reader, "value", read) == 1
        writer.execute("UPDATE facts SET value = 2")
        writer.commit()
        assert memo(reader, "value", read) == 2


def _seeded(*game_ids: str):
    db = make_grader_test_db()
    for i, gid in enumerate(game_ids):
        _game(db, gid, f"2025-06-{1 + i:02d}", "LCK", BLUE_COMP, RED_COMP, BLUE_WON)
    db.commit()
    return db


def test_the_same_corpus_state_reuses_one_build_and_a_write_replaces_it():
    db = _seeded("g1")
    builds: list[object] = []

    def build() -> object:
        builds.append(object())
        return builds[-1]

    first = memo(db, "probe", build)
    assert memo(db, "probe", build) is first
    assert len(builds) == 1
    other = memo(db, "other", build)
    assert memo(db, "probe", build) is first  # a second key never evicts the first
    assert memo(db, "other", build) is other
    assert len(builds) == 2
    _game(db, "g2", "2025-06-02", "LCK", BLUE_COMP, RED_COMP, BLUE_WON)
    db.commit()
    assert memo(db, "probe", build) is not first
    assert len(builds) == 3


@pytest.mark.parametrize("current_schema", [False, True])
def test_pending_fact_memo_does_not_survive_rollback(current_schema):
    db = sqlite3.connect(":memory:") if current_schema else make_grader_test_db()
    if current_schema:
        apply_current_schema(db)
    for i, gid in enumerate(("g1", "g2")):
        _game(db, gid, f"2025-06-{i + 1:02d}", "LCK", BLUE_COMP, RED_COMP, BLUE_WON)
    db.commit()
    before = _team_form(db)
    db.execute("UPDATE games SET winner = 'red' WHERE game_id = 'g1'")
    assert _team_form(db)[("blue", "g2")] < before[("blue", "g2")]
    db.rollback()
    assert _team_form(db) == before
    db.close()
