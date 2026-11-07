"""Analytics mart refresh — the one generation transaction over the three mart builders."""

from __future__ import annotations

import sqlite3
from datetime import date

from league_predictor.analytics._common import _as_of_iso, _write_run_log
from league_predictor.analytics.player_snapshot import refresh_player_daily_snapshot
from league_predictor.analytics.team_power_display import (
    refresh_team_power_display,
)
from league_predictor.analytics.team_snapshot import refresh_team_daily_snapshot
from league_predictor.db.rebuild_guard import require_rebuild_schema
from league_predictor.predraft.ratings.skill_filter.state_export import (
    ensure_team_state_current,
)


def refresh_analytics_marts(
    conn: sqlite3.Connection,
    *,
    as_of_date: date | None = None,
) -> dict[str, int | str]:
    """Rebuild all three marts for ``as_of_date`` in one transaction; per-mart row counts."""

    require_rebuild_schema(
        conn,
        "series",
        "games",
        "game_players",
        "models",
        "team_state",
        "pick_scores",
        purpose="dashboard analytics mart refresh",
    )
    resolved = date.fromisoformat(_as_of_iso(as_of_date))
    # Serving replay lands the served rating's team_state ONCE at refresh entry (when
    # stale); the per-team power-rating readout downstream is then a team_state lookup.
    ensure_team_state_current(conn)
    conn.commit()  # close the reconcile's txn so the generation begins from a clean slate

    run_log: list[dict] = []
    counts: dict[str, int] = {}
    prev_isolation = conn.isolation_level
    conn.isolation_level = (
        None  # explicit control: our BEGIN owns the generation, not implicit DML begins
    )
    try:
        conn.execute("BEGIN")
        try:
            counts["team_daily_snapshot"] = refresh_team_daily_snapshot(
                conn, as_of_date=resolved, run_log=run_log
            )
            counts["player_daily_snapshot"] = refresh_player_daily_snapshot(
                conn, as_of_date=resolved, run_log=run_log
            )
            counts["power_rating_snapshots"] = refresh_team_power_display(
                conn, as_of_date=resolved, run_log=run_log
            )
        except Exception:
            conn.execute("ROLLBACK")  # withhold the whole generation; nothing half-published
            raise
        else:
            conn.execute("COMMIT")  # publish all three marts atomically
    finally:
        conn.isolation_level = prev_isolation
        _write_run_log(conn, run_log)  # provenance in its own txn, survives a rollback

    return {**counts, "as_of_date": resolved.isoformat()}
