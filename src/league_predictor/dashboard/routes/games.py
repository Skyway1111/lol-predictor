"""HTTP registration for the games browser and individual game sheets."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Query

from league_predictor.dashboard.common import parse_filters
from league_predictor.dashboard.game_rows import GamePage, fetch_games_page
from league_predictor.dashboard.routes.game_detail import build_game_detail
from league_predictor.db import get_connection


def build_games_router(db_path: Path) -> APIRouter:
    router = APIRouter(prefix="/api/games", tags=["games"])

    @router.get("")
    def list_games(  # sightline-ok: 14 - FilterScope is the clump's home; FastAPI binds by name
        *,
        start_date: Annotated[str | None, Query()] = None,
        end_date: Annotated[str | None, Query()] = None,
        leagues: Annotated[str | None, Query()] = None,
        major_only: Annotated[bool, Query()] = True,
        team_id: Annotated[str | None, Query()] = None,
        q: Annotated[str | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=100)] = 25,
        offset: Annotated[int, Query(ge=0)] = 0,
    ):
        scope = parse_filters(start_date, end_date, leagues, major_only=major_only)
        conn = get_connection(db_path)
        try:
            return fetch_games_page(
                conn, scope, GamePage(team_id=team_id, q=q, limit=limit, offset=offset)
            )
        finally:
            conn.close()

    @router.get("/{game_id}")
    def game_detail(game_id: str):  # sightline-ok: 11 - FastAPI per-request conn lifecycle
        conn = get_connection(db_path)
        try:
            return build_game_detail(conn, game_id)
        finally:
            conn.close()

    return router
