"""sqlite_master readers shared by the schema and migration test modules."""

import sqlite3


def _table_names(conn: sqlite3.Connection) -> list[str]:
    cursor = conn.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    return [row[0] for row in cursor.fetchall()]


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _schema_objects(conn: sqlite3.Connection) -> list[tuple]:
    """(type, name, sql) for every user object, excluding the migration ledger."""
    return [
        tuple(row)
        for row in conn.execute(
            """
            SELECT type, name, sql FROM sqlite_master
            WHERE name NOT LIKE 'sqlite_%' AND name != 'schema_migrations'
            ORDER BY type, name
            """
        ).fetchall()
    ]


# The canonical table set the current schema creates. A hardcoded list so an
# accidental table add/drop trips this guard and forces an intentional update.
CANONICAL_TABLES = [
    "analytics_runs",
    "draft_actions",
    "draft_scores",
    "facts_state",
    "game_players",
    "game_team_stats",
    "games",
    "impact_history",
    "leagues",
    "models",
    "oe_team_names",
    "pandascore_team_map",
    "pick_scores",
    "player_aliases",
    "player_daily_snapshot",
    "player_id_map",
    "player_impact_scores",
    "players",
    "postdraft_scores",
    "postdraft_stack",
    "power_rating_snapshots",
    "predictions",
    "rosters",
    "series",
    "team_daily_snapshot",
    "team_state",
    "teams",
    "upcoming_matches",
]
