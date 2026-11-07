"""Canonical display-name reads for persisted team and player identities."""

import sqlite3

from league_predictor.db.names import (
    canonical_player_name_sql,
    canonical_team_name_sql,
    player_name_lookup,
)


def test_player_name_reads_choose_the_newest_alias() -> None:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE player_aliases (player_id TEXT, source TEXT, source_name TEXT)")
    conn.executemany(
        "INSERT INTO player_aliases (player_id, source, source_name) VALUES (?, ?, ?)",
        [
            ("p1", "oe", "Old Handle"),
            ("p2", "oe", "Only Handle"),
            ("p1", "manual", "Current Handle"),
        ],
    )

    name_sql = canonical_player_name_sql("q.player_id")
    rows = conn.execute(
        f"SELECT DISTINCT q.player_id, {name_sql} AS display_name "
        "FROM player_aliases q ORDER BY q.player_id"
    ).fetchall()
    scalar_names = {row["player_id"]: row["display_name"] for row in rows}

    assert scalar_names == {"p1": "Current Handle", "p2": "Only Handle"}
    assert player_name_lookup(conn) == scalar_names
    conn.close()


def test_canonical_team_name_sql_resolves_newest_name():
    """The scalar subquery returns a renamed team's newest name (MAX rowid), deterministically."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE oe_team_names (team_id TEXT, team_name TEXT)")
    conn.executemany(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES (?, ?)",
        [("t1", "Old Name"), ("t1", "New Name"), ("t2", "Solo")],
    )
    name_sql = canonical_team_name_sql("q.team_id")
    rows = conn.execute(
        f"SELECT DISTINCT q.team_id AS team_id, {name_sql} AS name "
        "FROM oe_team_names q ORDER BY team_id"
    ).fetchall()
    conn.close()
    resolved = {r["team_id"]: r["name"] for r in rows}
    assert resolved == {"t1": "New Name", "t2": "Solo"}
    assert len(rows) == 2  # one row per team_id despite the rename
