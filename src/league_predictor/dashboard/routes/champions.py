"""Champion browser and champion-detail endpoints over pick lines and bans in scope."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, TypedDict

from fastapi import APIRouter, HTTPException, Query

from league_predictor.dashboard.common import (
    FilterScope,
    parse_filters,
)
from league_predictor.db import get_connection
from league_predictor.db.draft_scores_read import preferred_context_predicate
from league_predictor.db.names import canonical_player_name_sql

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

_SIDE_TEAM_COL = {"blue": "g.blue_team", "red": "g.red_team"}


def kda(kills: float | None, deaths: float | None, assists: float | None) -> float | None:
    """Aggregate KDA = (kills + assists) / max(deaths, 1). Null when no lines."""
    if kills is None or assists is None:
        return None
    return (kills + assists) / max(deaths or 0, 1)


def _pick_scope(scope: FilterScope, role: str | None, side: str | None) -> tuple[str, list]:
    """WHERE and params for `game_players ⋈ games ⋈ series`, optionally by role and side."""
    extra = ["gp.champion_id IS NOT NULL"]
    params = list(scope.params)
    if role:
        extra.append("gp.role = ?")
        params.append(role)
    if side in _SIDE_TEAM_COL:
        extra.append(f"gp.team_id = {_SIDE_TEAM_COL[side]}")
    return scope.where(*extra), params


def _n_games(conn: sqlite3.Connection, scope: FilterScope) -> int:
    return conn.execute(
        f"SELECT COUNT(*) FROM games g JOIN series s ON s.series_id = g.series_id "
        f"WHERE {scope.where()}",
        scope.params,
    ).fetchone()[0]


class ChampionTable(TypedDict):
    n_games: int
    champions: list[dict]


def champion_table(
    conn: sqlite3.Connection, scope: FilterScope, role: str | None, side: str | None
) -> ChampionTable:
    """The full champion table for `GET /api/champions`."""
    n_games = _n_games(conn, scope)
    pick_where, pick_params = _pick_scope(scope, role, side)

    pick_rows = conn.execute(
        f"""
        SELECT gp.champion_id AS champion,
               COUNT(*) AS picks,
               SUM(gp.result) AS wins,
               AVG(gp.kills) AS avg_kills, AVG(gp.deaths) AS avg_deaths,
               AVG(gp.assists) AS avg_assists,
               SUM(gp.kills) AS sum_kills, SUM(gp.deaths) AS sum_deaths,
               SUM(gp.assists) AS sum_assists,
               AVG(gp.dpm) AS dpm, AVG(gp.cspm) AS cspm,
               AVG(gp.golddiffat15) AS gd15, AVG(gp.csdiffat15) AS csd15,
               AVG(gp.xpdiffat15) AS xpd15
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {pick_where}
        GROUP BY gp.champion_id
        """,
        pick_params,
    ).fetchall()

    # Main role = the champion's most-picked lane in scope.
    role_where, role_params = _pick_scope(scope, None, side)
    main_role: dict[str, str] = {}
    best_count: dict[str, int] = {}
    for row in conn.execute(
        f"""
        SELECT gp.champion_id AS champion, gp.role AS role, COUNT(*) AS n
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {role_where}
        GROUP BY gp.champion_id, gp.role
        """,
        role_params,
    ):
        champ, n = row["champion"], row["n"]
        if n > best_count.get(champ, -1):
            best_count[champ] = n
            main_role[champ] = row["role"]

    bans = _ban_counts(conn, scope, side)

    champions: dict[str, dict] = {}
    for row in pick_rows:
        champ = row["champion"]
        picks = row["picks"] or 0
        wins = row["wins"] or 0
        champions[champ] = {
            "champion": champ,
            "picks": picks,
            "bans": bans.get(champ, 0),
            "presence": (picks + bans.get(champ, 0)) / n_games if n_games else None,
            "wins": wins,
            "losses": picks - wins,
            "winrate": wins / picks if picks else None,
            "kda": kda(row["sum_kills"], row["sum_deaths"], row["sum_assists"]),
            "avg_kills": row["avg_kills"],
            "avg_deaths": row["avg_deaths"],
            "avg_assists": row["avg_assists"],
            "dpm": row["dpm"],
            "cspm": row["cspm"],
            "gd15": row["gd15"],
            "csd15": row["csd15"],
            "xpd15": row["xpd15"],
            "main_role": main_role.get(champ),
        }
    # Ban-only champions (bans but no picks in scope) still list.
    for champ, n_ban in bans.items():
        if champ not in champions:
            champions[champ] = {
                "champion": champ,
                "picks": 0,
                "bans": n_ban,
                "presence": n_ban / n_games if n_games else None,
                "wins": 0,
                "losses": 0,
                "winrate": None,
                "kda": None,
                "avg_kills": None,
                "avg_deaths": None,
                "avg_assists": None,
                "dpm": None,
                "cspm": None,
                "gd15": None,
                "csd15": None,
                "xpd15": None,
                "main_role": None,
            }
    return {"n_games": n_games, "champions": list(champions.values())}


def _ban_counts(conn: sqlite3.Connection, scope: FilterScope, side: str | None) -> dict[str, int]:
    """Ban count per champion in scope. `side` restricts to that side's bans."""
    extra = ["da.kind = 'ban'", "da.champion_id IS NOT NULL"]
    params = list(scope.params)
    if side in _SIDE_TEAM_COL:
        extra.append("da.side = ?")
        params.append(side)
    rows = conn.execute(
        f"""
        SELECT da.champion_id AS champion, COUNT(*) AS bans
        FROM draft_actions da
        JOIN games g ON g.game_id = da.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {scope.where(*extra)}
        GROUP BY da.champion_id
        """,
        params,
    ).fetchall()
    return {row["champion"]: row["bans"] for row in rows}


def _champion_detail(
    conn: sqlite3.Connection, champion: str, scope: FilterScope, role: str | None
) -> dict:
    n_games = _n_games(conn, scope)
    pick_where, pick_params = _pick_scope(scope, role, None)
    pick_where += " AND gp.champion_id = ?"
    pick_params = [*pick_params, champion]

    agg = conn.execute(
        f"""
        SELECT COUNT(*) AS picks, SUM(gp.result) AS wins,
               SUM(CASE WHEN gp.team_id = g.blue_team THEN 1 ELSE 0 END) AS picks_blue,
               SUM(CASE WHEN gp.team_id = g.red_team  THEN 1 ELSE 0 END) AS picks_red,
               AVG(gp.kills) AS avg_kills, AVG(gp.deaths) AS avg_deaths,
               AVG(gp.assists) AS avg_assists,
               SUM(gp.kills) AS sum_kills, SUM(gp.deaths) AS sum_deaths,
               SUM(gp.assists) AS sum_assists,
               AVG(gp.dpm) AS dpm, AVG(gp.cspm) AS cspm,
               AVG(gp.golddiffat15) AS gd15, AVG(gp.csdiffat15) AS csd15,
               AVG(gp.xpdiffat15) AS xpd15
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {pick_where}
        """,
        pick_params,
    ).fetchone()

    ban_where = scope.where("da.kind = 'ban'", "da.champion_id = ?")
    ban_row = conn.execute(
        f"""
        SELECT COUNT(*) AS bans,
               SUM(CASE WHEN da.side = 'blue' THEN 1 ELSE 0 END) AS bans_blue,
               SUM(CASE WHEN da.side = 'red'  THEN 1 ELSE 0 END) AS bans_red
        FROM draft_actions da
        JOIN games g ON g.game_id = da.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {ban_where}
        """,
        [*scope.params, champion],
    ).fetchone()

    picks = agg["picks"] or 0
    wins = agg["wins"] or 0
    bans = ban_row["bans"] or 0
    stats = {
        "champion": champion,
        "picks": picks,
        "picks_blue": agg["picks_blue"] or 0,
        "picks_red": agg["picks_red"] or 0,
        "bans": bans,
        "bans_blue": ban_row["bans_blue"] or 0,
        "bans_red": ban_row["bans_red"] or 0,
        "presence": (picks + bans) / n_games if n_games else None,
        "wins": wins,
        "losses": picks - wins,
        "winrate": wins / picks if picks else None,
        "kda": kda(agg["sum_kills"], agg["sum_deaths"], agg["sum_assists"]),
        "avg_kills": agg["avg_kills"],
        "avg_deaths": agg["avg_deaths"],
        "avg_assists": agg["avg_assists"],
        "dpm": agg["dpm"],
        "cspm": agg["cspm"],
        "gd15": agg["gd15"],
        "csd15": agg["csd15"],
        "xpd15": agg["xpd15"],
    }
    return {
        "stats": stats,
        "grade_presence": _grade_presence(conn, champion, scope, role),
        "best_players": _best_players(conn, champion, scope, role),
        "recent_games": _recent_games(conn, champion, scope, role),
    }


def _grade_presence(
    conn: sqlite3.Connection, champion: str, scope: FilterScope, role: str | None
) -> dict[str, int]:
    """Games per draft grade (S-D) of the picking side; `ungraded` means unscored games."""
    where, params = _pick_scope(scope, role, None)
    rows = conn.execute(
        f"""
        SELECT CASE WHEN gp.team_id = g.blue_team THEN ds.grade_blue
                    ELSE ds.grade_red END AS grade,
               COUNT(*) AS n
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        LEFT JOIN draft_scores ds
          ON ds.game_id = gp.game_id AND {preferred_context_predicate()}
        WHERE {where} AND gp.champion_id = ?
        GROUP BY grade
        """,
        [*params, champion],
    ).fetchall()
    out = {"S": 0, "A": 0, "B": 0, "C": 0, "D": 0, "ungraded": 0}
    for r in rows:
        out[r["grade"] if r["grade"] in out else "ungraded"] += r["n"]
    return out


def _best_players(
    conn: sqlite3.Connection, champion: str, scope: FilterScope, role: str | None
) -> list[dict]:
    """Top players on the champion (≥2 games), by win rate then games; top 10."""
    where, params = _pick_scope(scope, role, None)
    rows = conn.execute(
        f"""
        SELECT gp.player_id AS player_id,
               {canonical_player_name_sql("gp.player_id")} AS name,
               COUNT(*) AS games, SUM(gp.result) AS wins,
               SUM(gp.kills) AS sk, SUM(gp.deaths) AS sd, SUM(gp.assists) AS sa
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {where} AND gp.champion_id = ?
        GROUP BY gp.player_id
        HAVING COUNT(*) >= 2
        """,
        [*params, champion],
    ).fetchall()
    players = [
        {
            "player_id": r["player_id"],
            "name": r["name"] or r["player_id"],
            "games": r["games"],
            "wins": r["wins"] or 0,
            "winrate": (r["wins"] or 0) / r["games"] if r["games"] else None,
            "kda": kda(r["sk"], r["sd"], r["sa"]),
        }
        for r in rows
    ]
    players.sort(key=lambda p: (p["winrate"] or 0, p["games"]), reverse=True)
    return players[:10]


def _recent_games(
    conn: sqlite3.Connection, champion: str, scope: FilterScope, role: str | None
) -> list[dict]:
    """Latest 50 appearances of the champion in scope, with opponent + matchup."""
    where, params = _pick_scope(scope, role, None)
    rows = conn.execute(
        f"""
        SELECT gp.game_id, date(s.scheduled_ts) AS date, s.league_id,
               gp.player_id,
               {canonical_player_name_sql("gp.player_id")} AS player_name,
               gp.role, gp.result, gp.kills, gp.deaths, gp.assists,
               gp.csdiffat15 AS csd15, gp.dpm, gp.killparticipation AS kp,
               (SELECT o.champion_id FROM game_players o
                  WHERE o.game_id = gp.game_id AND o.role = gp.role
                    AND o.team_id <> gp.team_id LIMIT 1) AS opponent_champion
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {where} AND gp.champion_id = ?
        ORDER BY s.scheduled_ts DESC
        LIMIT 50
        """,
        [*params, champion],
    ).fetchall()
    return [
        {
            "game_id": r["game_id"],
            "date": r["date"],
            "league_id": r["league_id"],
            "player_id": r["player_id"],
            "player_name": r["player_name"] or r["player_id"],
            "role": r["role"],
            "result": "win" if r["result"] else "loss",
            "kills": r["kills"],
            "deaths": r["deaths"],
            "assists": r["assists"],
            "csd15": r["csd15"],
            "dpm": r["dpm"],
            "kp": r["kp"],
            "opponent_champion": r["opponent_champion"],
        }
        for r in rows
    ]


def build_champions_router(db_path: Path) -> APIRouter:
    router = APIRouter(prefix="/api/champions", tags=["champions"])

    @router.get("")
    def list_champions(  # sightline-ok: 14 - FilterScope owns the fields; FastAPI binds by name
        *,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
        leagues: Annotated[str | None, Query()] = None,
        major_only: Annotated[bool, Query()] = True,
        role: Annotated[str | None, Query()] = None,
        side: Annotated[str | None, Query()] = None,
    ):
        scope = parse_filters(start_date, end_date, leagues, major_only=major_only)
        conn = get_connection(db_path)
        try:
            return champion_table(conn, scope, role, side)
        finally:
            conn.close()

    @router.get("/{champion}")
    def champion_detail(  # sightline-ok: 14 - FilterScope owns the fields; FastAPI binds by name
        champion: str,
        *,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
        leagues: Annotated[str | None, Query()] = None,
        major_only: Annotated[bool, Query()] = True,
        role: Annotated[str | None, Query()] = None,
    ):
        scope = parse_filters(start_date, end_date, leagues, major_only=major_only)
        conn = get_connection(db_path)
        try:
            detail = _champion_detail(conn, champion, scope, role)
            if detail["stats"]["picks"] == 0 and detail["stats"]["bans"] == 0:
                raise HTTPException(status_code=404, detail="Champion not found in range")
            return detail
        finally:
            conn.close()

    return router
