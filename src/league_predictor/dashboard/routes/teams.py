"""Team terminal endpoints backed by analytics marts."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, HTTPException, Query

from league_predictor.dashboard.common import (
    TEAM_ACTIVITY_WINDOW_DAYS,
    active_team_ids_sql,
    default_date_range,
    latest_snapshot_date,
    major_region_clauses,
    parse_filters,
    parse_league_csv,
)
from league_predictor.dashboard.game_rows import GamePage, fetch_games_page
from league_predictor.db import get_connection
from league_predictor.predraft.ratings.impact_artifact import (
    impact_asof_payload_params,
    impact_asof_payload_select,
)
from league_predictor.vocab.roles import ROLE_RANK

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path


def _teams_list(
    conn: sqlite3.Connection, *, end_date: str, leagues: str | None, major_only: bool
) -> dict:
    """Activity-gated team rankings as of the newest snapshot on or before ``end_date``."""
    # Shared contract: an explicit league selection wins and silences the
    # major-only lens (else a minor-league pick would empty the list).
    league_ids = parse_league_csv(leagues)
    snapshot_date = latest_snapshot_date(conn, "team_daily_snapshot", end_date=end_date)
    if snapshot_date is None:
        return {"as_of_date": None, "teams": []}
    rating_date = latest_snapshot_date(conn, "power_rating_snapshots", end_date=end_date)

    # Anchored to the rating date shown, not the browsed range: dead team ids stay
    # in the mart forever, so a range-relative gate would rank frozen ratings.
    window_start = (
        date.fromisoformat(snapshot_date) - timedelta(days=TEAM_ACTIVITY_WINDOW_DAYS)
    ).isoformat()
    activity_sql, activity_params = active_team_ids_sql(window_start, snapshot_date, league_ids)

    join_params: list[object] = [rating_date or snapshot_date]
    where_clauses = ["t.date = ?"]
    where_params: list[object] = [snapshot_date]
    where_clauses.append(f"t.team_id IN {activity_sql}")
    where_params.extend(activity_params)
    if major_only and not league_ids:
        clauses, major_params = major_region_clauses("t.region", "t.display_name")
        where_clauses.extend(clauses)
        where_params.extend(major_params)
    where_sql = " AND ".join(where_clauses)
    params = join_params + where_params

    rows = conn.execute(
        f"""
        SELECT t.team_id, t.display_name, t.region, t.power_rating,
               p.global_rank
        FROM team_daily_snapshot t
        LEFT JOIN power_rating_snapshots p
          ON p.team_id = t.team_id AND p.date = ?
        WHERE {where_sql}
        ORDER BY COALESCE(p.global_rank, 999999), t.display_name
        """,
        params,
    ).fetchall()
    teams = [
        {
            "team_id": row["team_id"],
            "display_name": row["display_name"],
            "region": row["region"],
            "power_rating": row["power_rating"],
            "global_rank": row["global_rank"],
        }
        for row in rows
    ]
    return {"as_of_date": snapshot_date, "teams": teams}


def _draft_utilization(conn: sqlite3.Connection, team_id: str) -> dict | None:
    """The team's fitted draft coefficient ``beta + u`` and the oos games behind it, or None."""
    row = conn.execute(
        "SELECT (SELECT beta FROM postdraft_stack WHERE team_id = '*') + t.u AS value, t.n_oos "
        "FROM postdraft_stack t WHERE t.team_id = ?",
        (team_id,),
    ).fetchone()
    if row is None or row["value"] is None:
        return None
    return {"value": row["value"], "n_oos": row["n_oos"]}


def _team_overview(conn: sqlite3.Connection, team_id: str) -> dict:
    """One team's newest snapshot: power, form, macro and composite blocks."""
    max_dates = conn.execute(
        """
        SELECT
            (SELECT MAX(date) FROM team_daily_snapshot) AS team_date,
            (SELECT MAX(date) FROM power_rating_snapshots) AS power_date
        """
    ).fetchone()
    snapshot_date = str(max_dates["team_date"]) if max_dates and max_dates["team_date"] else None
    if snapshot_date is None:
        raise HTTPException(status_code=404, detail="No team analytics available")
    power_date = str(max_dates["power_date"]) if max_dates["power_date"] else snapshot_date
    snapshot = conn.execute(
        "SELECT * FROM team_daily_snapshot WHERE team_id = ? AND date = ?",
        (team_id, snapshot_date),
    ).fetchone()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Team not found")
    power = conn.execute(
        "SELECT * FROM power_rating_snapshots WHERE team_id = ? AND date = ?",
        (team_id, power_date),
    ).fetchone()
    return {
        "as_of_date": snapshot_date,
        "team_id": team_id,
        "display_name": snapshot["display_name"],
        "region": snapshot["region"],
        "draft_utilization": _draft_utilization(conn, team_id),
        "power": {
            # rating is the skill-filter-derived team power.
            "rating": snapshot["power_rating"],
            "global_rank": power["global_rank"] if power else None,
        },
        "form": {
            "win_rate_5": snapshot["win_rate_5"],
            "win_rate_10": snapshot["win_rate_10"],
            "win_streak": snapshot["win_streak"],
            "loss_streak": snapshot["loss_streak"],
        },
        "macro": {
            "dragon_rate": snapshot["dragon_rate"],
            "baron_rate": snapshot["baron_rate"],
            "tower_rate": snapshot["tower_rate"],
            "first_dragon_rate": snapshot["first_dragon_rate"],
            "first_baron_rate": snapshot["first_baron_rate"],
            "first_tower_rate": snapshot["first_tower_rate"],
        },
        "composite": {
            "avg_gold_diff_15": snapshot["avg_gold_diff_15"],
            "avg_cs_diff_15": snapshot["avg_cs_diff_15"],
            "avg_xp_diff_15": snapshot["avg_xp_diff_15"],
        },
    }


def _team_rating_history(conn: sqlite3.Connection, team_id: str, days: int) -> dict:
    """Power-rating trace over the last ``days`` days, oldest first."""
    rows = conn.execute(
        """
        SELECT date, rating, global_rank
        FROM power_rating_snapshots
        WHERE team_id = ? AND date >= date('now', ?)
        ORDER BY date ASC
        """,
        (team_id, f"-{days} day"),
    ).fetchall()
    history = [
        {"date": row["date"], "rating": row["rating"], "global_rank": row["global_rank"]}
        for row in rows
    ]
    return {"team_id": team_id, "history": history}


def _roster_player_rows(
    conn: sqlite3.Connection, player_ids: list[str], *, player_date: str, end_date: str
) -> dict[str, dict]:
    """One batched fetch of the roster's player snapshot rows, keyed by player id."""
    if not player_ids:
        return {}
    placeholders = ",".join("?" for _ in player_ids)
    # Live Impact per player, bound to the SAME end date /api/players binds:
    # one Impact per (player, date), whichever page shows it.
    rows = conn.execute(
        f"""
        SELECT p.player_id, p.display_name,
               p.games_played,
               {impact_asof_payload_select("p.player_id")}
        FROM player_daily_snapshot p
        WHERE p.player_id IN ({placeholders}) AND p.date = ?
        """,
        (*impact_asof_payload_params(end_date), *player_ids, player_date),
    ).fetchall()
    return {row["player_id"]: dict(row) for row in rows}


def _team_roster(conn: sqlite3.Connection, team_id: str, end_date: str) -> dict:
    """The team's lane-ordered roster with each player's live Impact payload."""
    snapshot_date = latest_snapshot_date(conn, "team_daily_snapshot", end_date=end_date)
    if snapshot_date is None:
        raise HTTPException(status_code=404, detail="No team analytics available")
    snapshot = conn.execute(
        "SELECT roster_json FROM team_daily_snapshot WHERE team_id = ? AND date = ?",
        (team_id, snapshot_date),
    ).fetchone()
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Team not found")
    roster = json.loads(snapshot["roster_json"])
    player_date = (
        latest_snapshot_date(conn, "player_daily_snapshot", end_date=end_date) or snapshot_date
    )
    player_map = _roster_player_rows(
        conn,
        [pid for pid in roster.values() if pid],
        player_date=player_date,
        end_date=end_date,
    )
    players = []
    for role, player_id in sorted(roster.items(), key=lambda kv: ROLE_RANK.get(kv[0], 99)):
        p_row = player_map.get(player_id)
        players.append(
            {
                "role": role,
                "player_id": player_id,
                "display_name": p_row["display_name"] if p_row else player_id,
                "impact": p_row["impact"] if p_row else None,
                "grade": p_row["grade"] if p_row else None,
                "games_played": p_row["games_played"] if p_row else None,
            }
        )
    return {"as_of_date": snapshot_date, "team_id": team_id, "players": players}


def build_teams_router(db_path: Path) -> APIRouter:
    router = APIRouter(prefix="/api/teams", tags=["teams"])

    @router.get("")
    def list_teams(
        *,
        major_only: Annotated[bool, Query()] = True,
        # Accepted for filter-bar compatibility; the rankings population is
        # activity-gated off the snapshot date, so the range's start is unused.
        start_date: Annotated[str | None, Query()] = None,  # noqa: ARG001 - filter-bar parity; rankings are activity-gated
        end_date: Annotated[str | None, Query()] = None,
        leagues: Annotated[str | None, Query()] = None,
    ):
        conn = get_connection(db_path)
        try:
            return _teams_list(
                conn,
                end_date=end_date or default_date_range()[1],
                leagues=leagues,
                major_only=major_only,
            )
        finally:
            conn.close()

    @router.get("/{team_id}/overview")
    # sightline-ok: 11 - FastAPI's per-request connection lifecycle; the routes share no fact
    def team_overview(team_id: str):
        conn = get_connection(db_path)
        try:
            return _team_overview(conn, team_id)
        finally:
            conn.close()

    @router.get("/{team_id}/rating-history")
    def team_rating_history(team_id: str, days: Annotated[int, Query(ge=7, le=365)] = 120):
        """Power-rating trace over the last `days` days, oldest first."""
        conn = get_connection(db_path)
        try:
            return _team_rating_history(conn, team_id, days)
        finally:
            conn.close()

    @router.get("/{team_id}/roster")
    # sightline-ok: 11 - FastAPI's per-request connection lifecycle; the routes share no fact
    def team_roster(team_id: str, end_date: Annotated[str | None, Query()] = None):
        conn = get_connection(db_path)
        try:
            return _team_roster(conn, team_id, end_date or default_date_range()[1])
        finally:
            conn.close()

    @router.get("/{team_id}/matches")
    def team_matches(
        team_id: str,
        *,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
        leagues: Annotated[str | None, Query()] = None,
        major_only: Annotated[bool, Query()] = True,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        """The team's games in scope, same row shape as `/api/games`."""
        scope = parse_filters(start_date, end_date, leagues, major_only=major_only)
        conn = get_connection(db_path)
        try:
            return fetch_games_page(
                conn, scope, GamePage(team_id=team_id, limit=limit, offset=offset)
            )
        finally:
            conn.close()

    return router
