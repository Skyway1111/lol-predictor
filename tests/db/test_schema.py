"""Tests for database bootstrap and the canonical current schema."""

from league_predictor.db import (
    create_schema,
    ensure_database_ready,
    get_connection,
)
from tests.db._schema_helpers import CANONICAL_TABLES, _column_names, _schema_objects, _table_names


def test_create_schema_creates_canonical_tables(tmp_db):
    create_schema(tmp_db)
    assert _table_names(tmp_db) == sorted(CANONICAL_TABLES)


def test_create_schema_is_idempotent(tmp_db):
    create_schema(tmp_db)
    create_schema(tmp_db)
    assert _table_names(tmp_db) == sorted(CANONICAL_TABLES)


def test_get_connection_creates_db(tmp_path):
    db_path = tmp_path / "new.db"
    conn = get_connection(db_path)
    assert db_path.exists()
    conn.close()


def test_get_connection_sets_busy_timeout(tmp_path):
    """A 5s busy timeout lets overlapping writers retry instead of failing SQLITE_BUSY."""
    conn = get_connection(tmp_path / "new.db")
    busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    conn.close()
    assert busy_timeout == 5000


def test_ensure_database_ready_bootstraps_schema_and_migration_log(tmp_db):
    ensure_database_ready(tmp_db)
    assert _table_names(tmp_db) == sorted(CANONICAL_TABLES)


def test_ensure_database_ready_is_no_op_on_a_current_db(tmp_db):
    """Re-running bootstrap on an already-current DB changes nothing."""
    ensure_database_ready(tmp_db)
    before = {(t, n): s for t, n, s in _schema_objects(tmp_db)}
    ensure_database_ready(tmp_db)
    after = {(t, n): s for t, n, s in _schema_objects(tmp_db)}
    assert before == after


def test_canonical_schema_contains_expected_facts_columns(tmp_db):
    create_schema(tmp_db)
    assert {"series_id", "team_a", "team_b", "winner", "status", "best_of"} <= _column_names(
        tmp_db, "series"
    )
    assert {
        "game_id",
        "series_id",
        "game_no",
        "blue_team",
        "red_team",
        "winner",
        "event_type",
    } <= _column_names(tmp_db, "games")
    assert {
        "game_id",
        "player_id",
        "team_id",
        "role",
        "champion_id",
        "golddiffat15",
    } <= _column_names(tmp_db, "game_players")
    assert {"game_id", "team_id", "dragons", "barons", "towers", "void_grubs"} <= _column_names(
        tmp_db, "game_team_stats"
    )


def test_canonical_schema_contains_expected_derived_columns(tmp_db):
    create_schema(tmp_db)
    assert {"model_id", "kind", "artifact_sha"} == _column_names(tmp_db, "models")
    assert {"team_id", "model_id", "asof_date", "strength"} == _column_names(tmp_db, "team_state")
    assert {
        "subject_kind",
        "subject_id",
        "model_id",
        "p_a",
        "asof_date",
        "context",
    } <= _column_names(tmp_db, "predictions")
    assert {"game_id", "model_id", "p_blue", "grade_blue", "grade_red"} <= _column_names(
        tmp_db, "draft_scores"
    )
