"""FastAPI dashboard app assembled from route modules."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from league_predictor import config
from league_predictor.dashboard.routes.champions import build_champions_router
from league_predictor.dashboard.routes.games import build_games_router
from league_predictor.dashboard.routes.players import build_players_router
from league_predictor.dashboard.routes.predict import build_predict_router
from league_predictor.dashboard.routes.static import register_static_routes
from league_predictor.dashboard.routes.teams import build_teams_router
from league_predictor.dashboard.routes.tournaments import build_tournaments_router
from league_predictor.dashboard.routes.upcoming import build_upcoming_router


def create_app(
    db_path: Path | None = None,
    *,
    on_startup: Callable[[], Awaitable[None]] | None = None,
    on_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    """Build the dashboard app: one router per route module, holding ``db_path`` in closures."""
    resolved_db = db_path or config.DB_PATH

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if on_startup is not None:
            await on_startup()
        try:
            yield
        finally:
            if on_shutdown is not None:
                await on_shutdown()

    app = FastAPI(title="LoL Match Predictor Dashboard", lifespan=lifespan)

    @app.get("/")
    def root() -> RedirectResponse:
        """Redirect the site root to the Games terminal (the landing tab)."""
        return RedirectResponse(url="/games")

    app.include_router(build_games_router(resolved_db))
    app.include_router(build_teams_router(resolved_db))
    app.include_router(build_players_router(resolved_db))
    app.include_router(build_champions_router(resolved_db))
    app.include_router(build_tournaments_router(resolved_db))
    app.include_router(build_upcoming_router(resolved_db))
    app.include_router(build_predict_router(resolved_db))

    static_dir = Path(__file__).resolve().parent / "static"
    register_static_routes(app, static_dir)
    return app
