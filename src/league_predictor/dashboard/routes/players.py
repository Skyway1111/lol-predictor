"""Player terminal endpoints backed by analytics marts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query

from league_predictor.analytics.player_draft import AXIS_WEIGHTS, RADAR_AXES
from league_predictor.dashboard.common import (
    active_player_ids_sql,
    default_date_range,
    latest_snapshot_date,
    major_region_clauses,
    parse_league_csv,
)
from league_predictor.db import get_connection
from league_predictor.db.names import canonical_team_name_sql
from league_predictor.predraft.ratings.impact_artifact import (
    IMPACT_PAYLOAD_COLUMNS,
    impact_asof_payload_params,
    impact_asof_payload_select,
    production_recipe,
)

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path


def _impact_threshold() -> dict:
    """The live artifact's bar, per request: a retrain relabels the page like the live scores."""
    recipe = production_recipe()
    return {
        "min_games": recipe.min_games,
        "window_months": recipe.qualification_window_months,
    }


def _players_query(
    snapshot_date: str,
    team_snapshot_date: str | None,
    *,
    start_date: str,
    end_date: str,
    league_ids: list[str],
    team_id: str | None,
    role: str | None,
    major_only: bool,
    min_games: int,
) -> tuple[str, list[object]]:
    """The players-list SELECT and its bound parameters, filters applied in bind order."""
    # Team name comes from the newest-name-wins scalar, never a join: oe_team_names
    # has one row per rename, which would duplicate the player.
    query = f"""
        SELECT p.player_id, p.display_name, p.team_id, p.role,
               p.games_played,
               {impact_asof_payload_select("p.player_id")},
               {canonical_team_name_sql("p.team_id")} AS team_display_name,
               ts.region AS region
        FROM player_daily_snapshot p
        LEFT JOIN team_daily_snapshot ts
          ON ts.team_id = p.team_id AND ts.date = ?
        WHERE p.date = ?
          AND p.display_name NOT LIKE 'oe:player:%'
    """
    # The Impact payload binds first, then the two snapshot dates.
    params: list[object] = [
        *impact_asof_payload_params(end_date),
        team_snapshot_date or snapshot_date,
        snapshot_date,
    ]
    # Only players who actually played in the window, scoped to the selected leagues:
    # drops benched/retired players lingering on an active team's roster.
    activity_sql, activity_params = active_player_ids_sql(start_date, end_date, league_ids)
    query += f" AND p.player_id IN {activity_sql}"
    params.extend(activity_params)
    if major_only and not league_ids:
        clauses, major_params = major_region_clauses("ts.region", "ts.display_name")
        query += " AND " + " AND ".join(clauses)
        params.extend(major_params)
    if team_id:
        query += " AND p.team_id = ?"
        params.append(team_id)
    if role:
        query += " AND p.role = ?"
        params.append(role)
    if min_games > 0:
        query += " AND p.games_played >= ?"
        params.append(min_games)
    # Rank by Impact, the one player number; sub-threshold players last.
    query += " ORDER BY impact IS NULL, impact DESC, p.display_name"
    return query, params


def _players_list(
    conn: sqlite3.Connection,
    *,
    start_date: str,
    end_date: str,
    leagues: str | None,
    team_id: str | None,
    role: str | None,
    major_only: bool,
    min_games: int,
) -> dict:
    """Impact-ranked players active in the window, as of the newest player snapshot."""
    # Shared contract: an explicit league selection wins and silences the
    # major-only lens (else a minor-league pick would empty the list).
    league_ids = parse_league_csv(leagues)
    snapshot_date = latest_snapshot_date(conn, "player_daily_snapshot", end_date=end_date)
    if snapshot_date is None:
        return {"as_of_date": None, "players": []}
    query, params = _players_query(
        snapshot_date,
        latest_snapshot_date(conn, "team_daily_snapshot", end_date=end_date),
        start_date=start_date,
        end_date=end_date,
        league_ids=league_ids,
        team_id=team_id,
        role=role,
        major_only=major_only,
        min_games=min_games,
    )
    rows = conn.execute(query, params).fetchall()
    return {
        "as_of_date": snapshot_date,
        "players": [
            {
                "player_id": row["player_id"],
                "display_name": row["display_name"],
                "team_id": row["team_id"],
                "role": row["role"],
                "games_played": row["games_played"],
                **{column: row[column] for column in IMPACT_PAYLOAD_COLUMNS},
                "impact_asof": row["impact_asof"],
                "team_display_name": row["team_display_name"] or row["team_id"],
                "region": row["region"],
            }
            for row in rows
        ],
    }


def _player_overview(conn: sqlite3.Connection, player_id: str, end_date: str) -> dict:
    """One player's newest snapshot: live Impact, pillars, details and the draft radar."""
    snapshot_date = latest_snapshot_date(conn, "player_daily_snapshot", end_date=end_date)
    if snapshot_date is None:
        raise HTTPException(status_code=404, detail="No player analytics available")
    row = conn.execute(
        "SELECT * FROM player_daily_snapshot WHERE player_id = ? AND date = ?",
        (player_id, snapshot_date),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Player not found")
    # Impact is a live as-of read, never copied into the mart (analytics-architecture).
    impact_row = conn.execute(
        f"SELECT {impact_asof_payload_select('?')}",
        impact_asof_payload_params(end_date, player_id=player_id),
    ).fetchone()
    team_name_row = conn.execute(
        f"SELECT {canonical_team_name_sql('?')} AS team_name", (row["team_id"],)
    ).fetchone()
    team_display_name = team_name_row["team_name"] or row["team_id"]
    pillars_array = json.loads(row["pillars_json"]) if row["pillars_json"] else []
    return {
        "as_of_date": snapshot_date,
        "player_id": player_id,
        "display_name": row["display_name"],
        "team_id": row["team_id"],
        "team_display_name": team_display_name,
        "role": row["role"],
        "games_played": row["games_played"],
        **{column: impact_row[column] for column in IMPACT_PAYLOAD_COLUMNS},
        "impact_asof": impact_row["impact_asof"],
        # Served so the UI states the bar without a second copy.
        "impact_threshold": _impact_threshold(),
        "pillars": pillars_array,
        "details": {
            "champion_pool": json.loads(row["champion_pool_json"]),
        },
        "draft_profile": {
            "overall_quality": row["overall_quality"],
            # Spoke order, labels and weights all come from analytics' RADAR_AXES,
            # so the page draws the radar without its own axis list.
            "axes": {
                axis.key: {
                    "label": axis.label,
                    "value": row[axis.key],
                    "sample": row[f"{axis.key}_n"] or 0,
                }
                for axis in RADAR_AXES
            },
            "weights": AXIS_WEIGHTS,
            "year": int(snapshot_date[:4]),
        },
    }


def build_players_router(db_path: Path) -> APIRouter:
    router = APIRouter(prefix="/api/players", tags=["players"])

    @router.get("")
    def list_players(
        *,
        team_id: Annotated[str | None, Query()] = None,
        role: Annotated[str | None, Query()] = None,
        major_only: Annotated[bool, Query()] = True,
        min_games: Annotated[int, Query(ge=0)] = 0,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
        leagues: Annotated[str | None, Query()] = None,
    ):
        default_sd, default_ed = default_date_range()
        conn = get_connection(db_path)
        try:
            return _players_list(
                conn,
                start_date=start_date or default_sd,
                end_date=end_date or default_ed,
                leagues=leagues,
                team_id=team_id,
                role=role,
                major_only=major_only,
                min_games=min_games,
            )
        finally:
            conn.close()

    @router.get("/{player_id}/overview")
    # sightline-ok: 11 - FastAPI's per-request connection lifecycle; the routes share no fact
    def player_overview(
        player_id: str,
        end_date: Annotated[str | None, Query()] = None,
    ):
        conn = get_connection(db_path)
        try:
            return _player_overview(conn, player_id, end_date or default_date_range()[1])
        finally:
            conn.close()

    return router
