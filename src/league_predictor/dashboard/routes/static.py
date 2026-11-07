"""Static frontend registration."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def register_static_routes(app: FastAPI, static_dir: Path) -> None:
    """Mount ``/static`` and one GET route per HTML shell; a no-op without the dir."""
    if not static_dir.exists():
        return

    def _make_handler(filename: str):
        def serve_page():
            page_path = static_dir / filename
            if not page_path.exists():
                raise HTTPException(404, f"{filename} not found")
            return FileResponse(page_path)

        return serve_page

    for page in sorted(static_dir.glob("*.html")):
        app.add_api_route(f"/{page.stem}", _make_handler(page.name), methods=["GET"])

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
