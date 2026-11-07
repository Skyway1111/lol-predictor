"""Tests for analytics/row_types.py — SQL generation and schema parity."""

import dataclasses
import shutil
import sqlite3

import pytest

from league_predictor.analytics.row_types import (
    PlayerDailySnapshotRow,
    TeamDailySnapshotRow,
    TeamPowerDisplayRow,
)

ALL_ROW_CLASSES = [
    TeamDailySnapshotRow,
    PlayerDailySnapshotRow,
    TeamPowerDisplayRow,
]


def _get_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    """Get column names in declaration order from the actual DB schema."""
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row[1]) for row in rows]


@pytest.fixture
def schema_conn(tmp_path, db_template_path):
    db_path = tmp_path / "schema_test.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


class TestSchemaParity:
    """Dataclass fields are an ordered subsequence of the DB table columns."""

    # ``insert_sql`` names every column, so an interspersed DB column the row type
    # omits simply takes its schema DEFAULT.

    @pytest.mark.parametrize("row_cls", ALL_ROW_CLASSES, ids=lambda c: c.__name__)
    def test_fields_match_columns(self, row_cls, schema_conn):
        expected = _get_table_columns(schema_conn, row_cls.__table__)
        actual = [f.name for f in dataclasses.fields(row_cls)]
        # Every field must exist in the table, in declaration order (ordered subsequence).
        it = iter(expected)
        ok = all(field in it for field in actual)
        assert ok, (
            f"{row_cls.__name__} fields are not an ordered subsequence of "
            f"{row_cls.__table__} columns.\n"
            f"  Dataclass: {actual}\n"
            f"  DB table:  {expected}"
        )


class TestRoundTrip:
    """Insert via row types and read back from DB."""

    def test_power_rating_round_trip(self, schema_conn):
        row = TeamPowerDisplayRow(
            date="2026-03-20",
            team_id="t1",
            display_name="T1",
            region="LCK",
            rating=1800.5,
            global_rank=1,
        )
        schema_conn.execute(TeamPowerDisplayRow.insert_sql(), dataclasses.astuple(row))
        schema_conn.commit()
        result = schema_conn.execute(
            "SELECT * FROM power_rating_snapshots WHERE team_id = 't1'"
        ).fetchone()
        assert result["rating"] == 1800.5
        assert result["global_rank"] == 1
        assert result["display_name"] == "T1"
