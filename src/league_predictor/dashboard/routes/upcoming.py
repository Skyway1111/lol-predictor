"""The Upcoming page's API: ``upcoming_matches`` rows with their pre-match predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query

from league_predictor import clock
from league_predictor.dashboard.common import parse_filters
from league_predictor.db import get_connection
from league_predictor.db.names import canonical_team_name_sql


def _today_range() -> tuple[str, str]:
    """The page's semantic default window: the current day."""
    today = clock.utc_today().isoformat()
    return today, today


def build_upcoming_router(db_path: Path) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["upcoming"])

    @router.get("/upcoming")
    def list_upcoming(
        *,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
        leagues: Annotated[str | None, Query()] = None,
        major_only: Annotated[bool, Query()] = True,
    ):
        default_sd, default_ed = _today_range()
        scope = parse_filters(
            start_date or default_sd,
            end_date or default_ed,
            leagues,
            major_only=major_only,
            series_alias="u",
        )
        name_a = canonical_team_name_sql("u.team_a")
        name_b = canonical_team_name_sql("u.team_b")
        conn = get_connection(db_path)
        try:
            rows = conn.execute(
                f"""
                SELECT u.ps_match_id, u.league_name, u.league_id, u.best_of, u.scheduled_ts,
                       u.ps_status, u.team_a, u.team_b, u.p_a, u.p_a_map, u.predicted_at,
                       u.series_id, u.score_a AS live_a, u.score_b AS live_b,
                       COALESCE({name_a}, u.team_a_name) AS team_a_name,
                       COALESCE({name_b}, u.team_b_name) AS team_b_name,
                       s.winner AS winner_team_id,
                       (SELECT COUNT(*) FROM games g
                         WHERE g.series_id = u.series_id AND g.winner = u.team_a) AS score_a,
                       (SELECT COUNT(*) FROM games g
                         WHERE g.series_id = u.series_id AND g.winner = u.team_b) AS score_b
                FROM upcoming_matches u
                LEFT JOIN series s ON s.series_id = u.series_id
                WHERE {scope.where()}
                ORDER BY u.scheduled_ts ASC
                """,
                scope.params,
            ).fetchall()
        finally:
            conn.close()
        matches = [
            {
                "ps_match_id": r["ps_match_id"],
                "scheduled_ts": r["scheduled_ts"],
                "league_id": r["league_id"],
                "league_name": r["league_name"],
                "best_of": r["best_of"],
                "ps_status": r["ps_status"],
                "team_a": {"team_id": r["team_a"], "name": r["team_a_name"]},
                "team_b": {"team_id": r["team_b"], "name": r["team_b_name"]},
                "p_a": r["p_a"],
                "series_id": r["series_id"],
                # The feed's maps won so far; the prediction is for the next map at this score.
                "score": None if r["live_a"] is None else [r["live_a"], r["live_b"]],
                "result": None
                if r["series_id"] is None
                else {
                    "winner_team_id": r["winner_team_id"],
                    "score_a": r["score_a"],
                    "score_b": r["score_b"],
                },
            }
            for r in rows
        ]
        return {"matches": matches, "window": {"start_date": scope.sd, "end_date": scope.ed}}

    return router
