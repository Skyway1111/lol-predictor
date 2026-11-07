"""League list (`/api/leagues`) and per-league tournament detail aggregates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Query

from league_predictor.dashboard.common import (
    DASHBOARD_MAJOR_LEAGUES,
    default_date_range,
    parse_filters,
)
from league_predictor.dashboard.routes.champions import champion_table, kda
from league_predictor.db import get_connection
from league_predictor.db.names import canonical_player_name_sql, canonical_team_name_sql

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping
    from pathlib import Path

_MAJOR = set(DASHBOARD_MAJOR_LEAGUES)


def _team_names(conn: sqlite3.Connection, team_ids: set[str]) -> dict[str, str]:
    """Canonical (newest-name-wins) display name per team_id, one row each.

    Returns:
        One name per id; ``oe_team_names`` keeps a row per rename, so the DISTINCT plus
        canonical scalar is what collapses a team to its newest name.
    """
    if not team_ids:
        return {}
    placeholders = ",".join("?" * len(team_ids))
    name_sql = canonical_team_name_sql("o.team_id")
    return {
        row["team_id"]: row["team_name"]
        for row in conn.execute(
            f"SELECT DISTINCT o.team_id AS team_id, {name_sql} AS team_name "
            f"FROM oe_team_names o WHERE o.team_id IN ({placeholders})",
            list(team_ids),
        )
    }


def _leagues_in_range(
    conn: sqlite3.Connection,
    sd: str,
    ed: str,
    *,
    major_only: bool = False,
) -> list[dict]:
    """Leagues with at least one game in [sd, ed], n_games desc; `major_only` filters."""
    rows = conn.execute(
        """
        SELECT s.league_id AS league_id,
               COUNT(DISTINCT g.game_id) AS n_games,
               COUNT(DISTINCT s.series_id) AS n_series
        FROM series s
        JOIN games g ON g.series_id = s.series_id
        WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) <= ?
        GROUP BY s.league_id
        HAVING n_games >= 1
        ORDER BY n_games DESC
        """,
        (sd, ed),
    ).fetchall()
    names = {
        row["league_id"]: row["name"] for row in conn.execute("SELECT league_id, name FROM leagues")
    }
    leagues = [
        {
            "league_id": r["league_id"],
            "name": names.get(r["league_id"]) or r["league_id"],
            "n_games": r["n_games"],
            "n_series": r["n_series"],
            "is_major": r["league_id"] in _MAJOR,
        }
        for r in rows
    ]
    if major_only:
        leagues = [lg for lg in leagues if lg["is_major"]]
    return leagues


def _league_label(conn: sqlite3.Connection, league_id: str) -> dict:
    row = conn.execute(
        "SELECT name FROM leagues WHERE league_id = ? LIMIT 1", (league_id,)
    ).fetchone()
    return {
        "league_id": league_id,
        "name": (row["name"] if row and row["name"] else league_id),
        "is_major": league_id in _MAJOR,
    }


def _share(blue: int, denom: int) -> float | None:
    """Blue-side share of `denom` events, or null when nothing happened."""
    return blue / denom if denom else None


def _tournament_detail(conn: sqlite3.Connection, league_id: str, sd: str, ed: str) -> dict:
    """Aggregate one league's games in [sd, ed] into the tournament sheet."""
    scope = parse_filters(sd, ed, league_id, major_only=False)
    where = scope.where()
    params = scope.params

    games = conn.execute(
        f"""
        SELECT g.game_id, g.series_id, g.game_no, g.blue_team, g.red_team, g.winner,
               g.duration_s, g.patch, s.best_of, date(s.scheduled_ts) AS date
        FROM games g JOIN series s ON s.series_id = g.series_id
        WHERE {where}
        ORDER BY s.scheduled_ts DESC
        """,
        params,
    ).fetchall()

    if not games:
        return {
            "league": _league_label(conn, league_id),
            "overview": {
                "n_games": 0,
                "n_series": 0,
                "avg_duration_s": None,
                "avg_kills": None,
                "shortest_game": None,
                "longest_game": None,
                "most_kills_game": None,
                "blue_red": {
                    "win_rate": None,
                    "first_blood": None,
                    "first_tower": None,
                    "first_dragon": None,
                    "first_baron": None,
                },
                "top_kda": [],
            },
            "ranking": [],
            "picks_bans": [],
            "matches": [],
        }

    game_ids = [g["game_id"] for g in games]
    ph = ",".join("?" * len(game_ids))
    gts = {}
    for row in conn.execute(f"SELECT * FROM game_team_stats WHERE game_id IN ({ph})", game_ids):
        gts.setdefault(row["game_id"], {})[row["team_id"]] = row
    fb_team = {
        row["game_id"]: row["team_id"]
        for row in conn.execute(
            f"SELECT game_id, team_id FROM game_players "
            f"WHERE game_id IN ({ph}) AND firstblood = 1 GROUP BY game_id",
            game_ids,
        )
    }

    team_ids: set[str] = set()
    for g in games:
        team_ids.update((g["blue_team"], g["red_team"]))
    names = _team_names(conn, team_ids)

    overview = _overview(
        games, gts=gts, fb_team=fb_team, names=names, conn=conn, where=where, params=params
    )
    ranking = _ranking(games, gts, names)
    picks_bans = sorted(
        champion_table(conn, scope, None, None)["champions"],
        key=lambda c: c["presence"] or 0,
        reverse=True,
    )
    matches = _matches(games, names)
    return {
        "league": _league_label(conn, league_id),
        "overview": overview,
        "ranking": ranking,
        "picks_bans": picks_bans,
        "matches": matches,
    }


def _overview(
    games: list[sqlite3.Row],
    *,
    gts: dict[str, dict[str, sqlite3.Row]],
    fb_team: Mapping[str, str],
    names: Mapping[str, str],
    conn: sqlite3.Connection,
    where: str,
    params: list,
) -> dict:
    n_games = len(games)
    n_series = len({g["series_id"] for g in games})
    durations = [g["duration_s"] for g in games if g["duration_s"] is not None]
    avg_duration = sum(durations) / len(durations) if durations else None

    def game_kills(gid: str) -> int | None:
        rows = gts.get(gid)
        if not rows:
            return None
        vals = [r["teamkills"] for r in rows.values() if r["teamkills"] is not None]
        return sum(vals) if vals else None

    kills_by_game = {g["game_id"]: game_kills(g["game_id"]) for g in games}
    kills_vals = [v for v in kills_by_game.values() if v is not None]
    avg_kills = sum(kills_vals) / len(kills_vals) if kills_vals else None

    def glabel(g) -> str:
        blue, red = g["blue_team"], g["red_team"]
        return f"{names.get(blue, blue)} vs {names.get(red, red)}"

    with_dur = [g for g in games if g["duration_s"] is not None]
    shortest = min(with_dur, key=lambda g: g["duration_s"], default=None)
    longest = max(with_dur, key=lambda g: g["duration_s"], default=None)
    with_kills = [g for g in games if kills_by_game[g["game_id"]] is not None]
    most_kills = max(with_kills, key=lambda g: kills_by_game[g["game_id"]], default=None)  # pyright: ignore[reportCallIssue, reportArgumentType]  # with_kills filters the key to non-None

    # Blue-side shares.
    decided = [g for g in games if g["winner"] in (g["blue_team"], g["red_team"])]
    blue_wins = sum(1 for g in decided if g["winner"] == g["blue_team"])
    fb_games = [g for g in games if g["game_id"] in fb_team]
    fb_blue = sum(1 for g in fb_games if fb_team[g["game_id"]] == g["blue_team"])

    def obj_share(flag: str):
        blue = denom = 0
        for g in games:
            rows = gts.get(g["game_id"])
            if not rows:
                continue
            b = rows.get(g["blue_team"])
            r = rows.get(g["red_team"])
            b_has = bool(b[flag]) if b and b[flag] is not None else False
            r_has = bool(r[flag]) if r and r[flag] is not None else False
            if b_has or r_has:
                denom += 1
                if b_has:
                    blue += 1
        return _share(blue, denom)

    return {
        "n_games": n_games,
        "n_series": n_series,
        "avg_duration_s": avg_duration,
        "avg_kills": avg_kills,
        "shortest_game": None
        if shortest is None
        else {
            "game_id": shortest["game_id"],
            "label": glabel(shortest),
            "duration_s": shortest["duration_s"],
        },
        "longest_game": None
        if longest is None
        else {
            "game_id": longest["game_id"],
            "label": glabel(longest),
            "duration_s": longest["duration_s"],
        },
        "most_kills_game": None
        if most_kills is None
        else {
            "game_id": most_kills["game_id"],
            "label": glabel(most_kills),
            "kills": kills_by_game[most_kills["game_id"]],
        },
        "blue_red": {
            "win_rate": _share(blue_wins, len(decided)),
            "first_blood": _share(fb_blue, len(fb_games)),
            "first_tower": obj_share("firsttower"),
            "first_dragon": obj_share("firstdragon"),
            "first_baron": obj_share("firstbaron"),
        },
        "top_kda": _top_kda(conn, where, params),
    }


def _top_kda(conn: sqlite3.Connection, where: str, params: list) -> list[dict]:
    # Alias-less players are dropped (OE id leak guard, same rule as /api/players)
    # — a raw "oe:player:…" id is noise, not a leaderboard entry.
    rows = conn.execute(
        f"""
        SELECT gp.player_id,
               {canonical_player_name_sql("gp.player_id")} AS name,
               SUM(gp.kills) AS k, SUM(gp.deaths) AS d, SUM(gp.assists) AS a
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE {where}
        GROUP BY gp.player_id
        HAVING COUNT(*) >= 3 AND name IS NOT NULL
        """,
        params,
    ).fetchall()
    scored = []
    for r in rows:
        k, d, a = r["k"] or 0, r["d"] or 0, r["a"] or 0
        scored.append(
            {
                "player_id": r["player_id"],
                "name": r["name"],
                "kda": kda(k, d, a),
                "kills": k,
                "deaths": d,
                "assists": a,
            }
        )
    scored.sort(key=lambda p: p["kda"], reverse=True)
    return scored[:5]


def _ranking(games, gts, names: Mapping[str, str]) -> list[dict]:
    """Per-team record: per-game wins/losses, avg duration, avg gold-diff/min."""
    acc: dict[str, dict] = {}

    def bucket(team_id: str) -> dict:
        return acc.setdefault(team_id, {"wins": 0, "losses": 0, "dur": [], "gdm": []})

    for g in games:
        rows = gts.get(g["game_id"], {})
        for team_id, opp_id in ((g["blue_team"], g["red_team"]), (g["red_team"], g["blue_team"])):
            b = bucket(team_id)
            if g["winner"] == team_id:
                b["wins"] += 1
            elif g["winner"] == opp_id:
                b["losses"] += 1
            if g["duration_s"] is not None:
                b["dur"].append(g["duration_s"])
            tr, orow = rows.get(team_id), rows.get(opp_id)
            if tr and orow and tr["totalgold"] is not None and orow["totalgold"] is not None:
                minutes = (tr["gamelength"] or g["duration_s"] or 0) / 60.0
                if minutes > 0:
                    b["gdm"].append((tr["totalgold"] - orow["totalgold"]) / minutes)

    out = []
    for team_id, b in acc.items():
        played = b["wins"] + b["losses"]
        out.append(
            {
                "team_id": team_id,
                "name": names.get(team_id, team_id),
                "wins": b["wins"],
                "losses": b["losses"],
                "winrate": b["wins"] / played if played else None,
                "avg_duration_s": sum(b["dur"]) / len(b["dur"]) if b["dur"] else None,
                "avg_gdm": sum(b["gdm"]) / len(b["gdm"]) if b["gdm"] else None,
            }
        )
    out.sort(
        key=lambda t: (t["winrate"] if t["winrate"] is not None else -1, t["wins"]), reverse=True
    )
    return out


def _matches(games, names: Mapping[str, str]) -> list[dict]:
    """Series list with per-team map scores + a game1_id link, most recent first."""
    series: dict[str, dict] = {}
    for g in games:
        s = series.get(g["series_id"])
        if s is None:
            # team_a/team_b in canonical string order for a stable orientation.
            a, b = sorted((g["blue_team"], g["red_team"]))
            s = series[g["series_id"]] = {
                "series_id": g["series_id"],
                "date": g["date"],
                "team_a": a,
                "team_b": b,
                "score_a": 0,
                "score_b": 0,
                "best_of": g["best_of"],
                "patch": g["patch"],
                "game1_id": g["game_id"],
                "_game1_no": g["game_no"],
            }
        if g["winner"] == s["team_a"]:
            s["score_a"] += 1
        elif g["winner"] == s["team_b"]:
            s["score_b"] += 1
        # Lowest game_no wins the series link (rows arrive in series-date order,
        # not game order); null game_no sorts last.
        if (g["game_no"] or 999) < (s["_game1_no"] or 999):
            s["game1_id"], s["_game1_no"] = g["game_id"], g["game_no"]
    out = []
    for s in series.values():
        winner = (
            s["team_a"]
            if s["score_a"] > s["score_b"]
            else s["team_b"]
            if s["score_b"] > s["score_a"]
            else None
        )
        out.append(
            {
                "series_id": s["series_id"],
                "date": s["date"],
                "team_a": {"team_id": s["team_a"], "name": names.get(s["team_a"], s["team_a"])},
                "team_b": {"team_id": s["team_b"], "name": names.get(s["team_b"], s["team_b"])},
                "score_a": s["score_a"],
                "score_b": s["score_b"],
                "winner_team_id": winner,
                "best_of": s["best_of"],
                "patch": s["patch"],
                "game1_id": s["game1_id"],
            }
        )
    out.sort(key=lambda m: m["date"], reverse=True)
    return out


def build_tournaments_router(db_path: Path) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["tournaments"])

    @router.get("/leagues")
    def list_leagues(
        *,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
        major_only: Annotated[bool, Query()] = False,
    ):
        default_sd, default_ed = default_date_range()
        sd, ed = start_date or default_sd, end_date or default_ed
        conn = get_connection(db_path)
        try:
            return {"leagues": _leagues_in_range(conn, sd, ed, major_only=major_only)}
        finally:
            conn.close()

    @router.get("/tournaments/{league_id}")
    def tournament_detail(
        league_id: str,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
    ):
        default_sd, default_ed = default_date_range()
        sd, ed = start_date or default_sd, end_date or default_ed
        conn = get_connection(db_path)
        try:
            return _tournament_detail(conn, league_id, sd, ed)
        finally:
            conn.close()

    return router
