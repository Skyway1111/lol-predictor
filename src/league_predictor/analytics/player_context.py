"""Bulk-queried lookup tables (``SnapshotContext``) for the player-snapshot passes."""

from __future__ import annotations

import dataclasses
import sqlite3
from collections import defaultdict
from datetime import date, timedelta

from league_predictor.predraft.ratings.player_scoring import (
    compute_position_baselines,
    resolve_tournament_tier,
)


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotContext:
    """Bulk-queried lookup tables for the draft-axes pass and row assembly."""

    tier_baselines: dict[int, dict]
    player_tiers: dict[str, int]
    season_champions: dict[str, list[tuple[str, int]]]


def _load_snapshot_context(
    conn: sqlite3.Connection,
    as_of: date,
) -> SnapshotContext:
    """Load all bulk-queried lookup tables needed by the snapshot passes."""
    tier_baselines: dict[int, dict] = {}
    for t in (1, 2, 3):
        tier_baselines[t] = compute_position_baselines(conn, as_of, tier=t)
    return SnapshotContext(
        tier_baselines=tier_baselines,
        player_tiers=_bulk_player_tiers(conn, as_of.isoformat()),
        season_champions=_bulk_current_season_champions(conn, as_of),
    )


def _bulk_player_tiers(conn: sqlite3.Connection, as_of: str) -> dict[str, int]:
    """player_id -> tier of their most recent tournament before ``as_of``, in one query."""
    rows = conn.execute(
        """
        SELECT player_id, tournament FROM (
            SELECT gp.player_id, s.league_id AS tournament,
                   ROW_NUMBER() OVER (
                       PARTITION BY gp.player_id ORDER BY s.scheduled_ts DESC
                   ) AS rn
            FROM game_players gp
            JOIN games g ON g.game_id = gp.game_id
            JOIN series s ON s.series_id = g.series_id
            WHERE date(s.scheduled_ts) < ? AND gp.player_id IS NOT NULL
        ) WHERE rn = 1
        """,
        (as_of,),
    ).fetchall()
    return {row["player_id"]: resolve_tournament_tier(row["tournament"]) for row in rows}


def _bulk_current_season_champions(
    conn: sqlite3.Connection,
    as_of_date: date,
) -> dict[str, list[tuple[str, int]]]:
    """Champion game counts since Jan 1 of the as-of year, per player, most-played first."""
    season_start = f"{as_of_date.year}-01-01"
    next_day = (as_of_date + timedelta(days=1)).isoformat()
    rows = conn.execute(
        """
        SELECT gp.player_id, gp.champion_id, COUNT(*) AS games
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) < ?
          AND gp.champion_id IS NOT NULL
        GROUP BY gp.player_id, gp.champion_id
        """,
        (season_start, next_day),
    ).fetchall()
    by_player: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for row in rows:
        by_player[row["player_id"]].append((row["champion_id"], int(row["games"])))
    for champs in by_player.values():
        champs.sort(key=lambda c: c[1], reverse=True)
    return dict(by_player)
