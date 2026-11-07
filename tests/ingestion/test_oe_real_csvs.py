"""The real Oracle's Elixir CSVs in ``config.RAW_DIR`` ingest cleanly into a fresh DB.

The newest year runs in the fast lane, about 30 s; every older year runs in the slow lane.
"""

from __future__ import annotations

import sqlite3

import pytest

from league_predictor import config
from league_predictor.db.schema import apply_current_schema
from league_predictor.ingestion.oracles_elixir import ingest_oe_results, load_csv

_CSVS = sorted(config.RAW_DIR.glob("oe_*.csv"))
_ID_COLUMNS = {
    "series": ("team_a", "team_b", "winner"),
    "games": ("blue_team", "red_team", "winner"),
    "game_players": ("player_id", "team_id"),
    "game_team_stats": ("team_id",),
    "teams": ("team_id",),
    "players": ("player_id",),
}


@pytest.mark.parametrize(
    "csv",
    [
        pytest.param(path, id=path.stem, marks=() if path == _CSVS[-1] else pytest.mark.slow)
        for path in _CSVS
    ]
    or [pytest.param(None, marks=pytest.mark.skip(reason=f"no OE CSVs in {config.RAW_DIR}"))],
)
def test_real_csv_ingests_with_ids_intact(csv):
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys=ON")
    apply_current_schema(conn)

    assert ingest_oe_results(conn, load_csv(csv)) > 0

    # A blank OE id must be skipped, never stored as the text 'nan' and pooled.
    pooled = {
        f"{table}.{col}": n
        for table, cols in _ID_COLUMNS.items()
        for col in cols
        if (n := conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {col} = 'nan'").fetchone()[0])  # noqa: S608 - fixed identifiers
    }
    assert pooled == {}
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
    conn.close()
