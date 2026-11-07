"""Recognize the supported schema baseline before any bootstrap DDL."""

from __future__ import annotations

import sqlite3

SCHEMA_VERSION = 21

# Frozen upgrade inputs for the supported 0020 schema, not current model features.
_RETIRED_COLUMNS = {
    "player_daily_snapshot": (
        "laning_json",
        "combat_json",
        "vision_json",
        "laning_rating",
        "combat_rating",
        "vision_rating",
        "confidence",
        "champion_diversity",
    ),
    "team_daily_snapshot": (
        "win_rate_20",
        "performance_slope",
        "days_since_last",
        "avg_game_duration",
        "kills_per_min",
        "first_blood_rate",
        "avg_dpm",
        "avg_kda",
        "avg_vision_score",
        "lineup_games_together",
        "lineup_win_rate",
        "weighted_win_rate",
    ),
    "power_rating_snapshots": ("regional_rank", "rating_delta"),
}


def schema_upgrade_sql(conn: sqlite3.Connection, current_schema_sql: str) -> str | None:
    """DDL for a fresh DB or full 0020 baseline, None for current; reject anything older."""
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == SCHEMA_VERSION:
        return None
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if not tables and version == 0:
        return ""
    message = (
        "Database schema is obsolete or unsupported. Upgrade a copy through migration 0020 "
        "with the prior release before opening it here; no data has been changed."
    )
    if version != 0 or "schema_migrations" not in tables:
        raise RuntimeError(message)
    applied = {row[0][:4] for row in conn.execute("SELECT migration_id FROM schema_migrations")}
    if applied != {f"{n:04}" for n in range(1, 21)}:
        raise RuntimeError(message)
    # The ledger alone cannot certify table shapes. Compare every retained column first.
    canonical = sqlite3.connect(":memory:")
    try:
        canonical.executescript(current_schema_sql)
        expected_tables = [
            row[0]
            for row in canonical.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name != 'facts_state'"
            )
        ]
        statements = []
        for table in expected_tables:
            actual = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            expected = {row[1] for row in canonical.execute(f"PRAGMA table_info({table})")}
            retired = set(_RETIRED_COLUMNS.get(table, ()))
            if actual - retired != expected:
                raise RuntimeError(f"{message} Incompatible table: {table}")
            statements.extend(
                f"ALTER TABLE {table} DROP COLUMN {column};" for column in sorted(actual & retired)
            )
    finally:
        canonical.close()
    statements.extend(
        f"DROP TABLE IF EXISTS {table};"
        for table in ("pipeline_health", "golgg_player_laning", "schema_migrations")
    )
    return "\n".join(statements) + "\n"
