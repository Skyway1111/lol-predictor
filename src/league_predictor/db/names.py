"""Canonical display-name reads for database identities."""

# Name tables are append-only, so the greatest rowid for an id is its current
# name; that ordering has one home here (scalar-subquery and bulk forms).

from __future__ import annotations

import sqlite3


def _newest_name_sql(table: str, *, name_col: str, id_col: str, id_expr: str, alias: str) -> str:
    return (
        f"(SELECT {alias}.{name_col} FROM {table} {alias} "
        f"WHERE {alias}.{id_col} = {id_expr} "
        f"ORDER BY {alias}.rowid DESC LIMIT 1)"
    )


def _newest_name_lookup(
    conn: sqlite3.Connection, table: str, *, name_col: str, id_col: str
) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for row in conn.execute(f"SELECT {id_col}, {name_col} FROM {table} ORDER BY rowid DESC"):
        lookup.setdefault(str(row[0]), str(row[1]))
    return lookup


def canonical_team_name_sql(team_id_expr: str) -> str:
    """Return a newest-name-wins scalar subquery for a team-id expression."""
    return _newest_name_sql(
        "oe_team_names", name_col="team_name", id_col="team_id", id_expr=team_id_expr, alias="otn"
    )


# sightline-ok: 37 - kept parametric to mirror canonical_team_name_sql, whose expr does vary
def canonical_player_name_sql(player_id_expr: str) -> str:
    """Return a newest-alias-wins scalar subquery for a player-id expression."""
    return _newest_name_sql(
        "player_aliases",
        name_col="source_name",
        id_col="player_id",
        id_expr=player_id_expr,
        alias="pa",
    )


def team_name_lookup(conn: sqlite3.Connection) -> dict[str, str]:
    """Return newest canonical team names keyed by team id."""
    return _newest_name_lookup(conn, "oe_team_names", name_col="team_name", id_col="team_id")


def player_name_lookup(conn: sqlite3.Connection) -> dict[str, str]:
    """Return newest canonical player aliases keyed by player id."""
    return _newest_name_lookup(conn, "player_aliases", name_col="source_name", id_col="player_id")
