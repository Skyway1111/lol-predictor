"""Current bootstrap refuses obsolete databases before touching their facts."""

import pytest

from league_predictor.db import create_schema, ensure_database_ready


def test_unversioned_database_is_rejected_before_any_ddl(tmp_db):
    tmp_db.execute("CREATE TABLE legacy_fact (value TEXT)")
    before = tmp_db.execute("SELECT name FROM sqlite_master").fetchall()
    with pytest.raises(RuntimeError, match="obsolete"):
        ensure_database_ready(tmp_db)
    assert tmp_db.execute("SELECT name FROM sqlite_master").fetchall() == before


def test_supported_baseline_preserves_facts_and_rebuilds_current_schema(tmp_db):
    create_schema(tmp_db)
    tmp_db.execute("PRAGMA user_version = 0")
    tmp_db.execute("CREATE TABLE schema_migrations (migration_id TEXT PRIMARY KEY)")
    tmp_db.executemany(
        "INSERT INTO schema_migrations VALUES (?)", [(f"{n:04}_baseline",) for n in range(1, 21)]
    )
    tmp_db.execute("ALTER TABLE player_daily_snapshot ADD COLUMN laning_rating REAL")
    tmp_db.execute("INSERT INTO teams (team_id, name) VALUES ('A', 'Alpha')")
    tmp_db.commit()
    ensure_database_ready(tmp_db)
    assert tuple(tmp_db.execute("SELECT team_id, name FROM teams").fetchone()) == ("A", "Alpha")
    assert "laning_rating" not in {
        row[1] for row in tmp_db.execute("PRAGMA table_info(player_daily_snapshot)")
    }
    ensure_database_ready(tmp_db)


def test_baseline_ledger_cannot_hide_an_obsolete_table_shape(tmp_db):
    create_schema(tmp_db)
    tmp_db.execute("PRAGMA user_version = 0")
    tmp_db.execute("CREATE TABLE schema_migrations (migration_id TEXT PRIMARY KEY)")
    tmp_db.executemany(
        "INSERT INTO schema_migrations VALUES (?)", [(f"{n:04}_baseline",) for n in range(1, 21)]
    )
    tmp_db.execute("ALTER TABLE games DROP COLUMN event_type")
    with pytest.raises(RuntimeError, match="Incompatible table: games"):
        ensure_database_ready(tmp_db)
