"""Runtime entrypoint for the read-only dashboard server."""

import logging
import os
import socket
import threading
import time
import webbrowser
from pathlib import Path

import uvicorn

from league_predictor import config
from league_predictor.db import ensure_database_ready, get_connection, latest_played_date

logger = logging.getLogger(__name__)

_ALL_INTERFACES = "0.0.0.0"  # noqa: S104 - opt-in LAN-facing dashboard bind, never the default


def _resolve_bind_host(*, public: bool) -> str:
    """``0.0.0.0`` when ``public`` or ``LOL_PREDICT_PUBLIC`` opts in, else ``127.0.0.1``."""
    env_public = os.environ.get("LOL_PREDICT_PUBLIC", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return _ALL_INTERFACES if (public or env_public) else "127.0.0.1"


def _check_port_available(port: int, host: str = "127.0.0.1") -> None:
    """Fail fast if another process is already listening on this port."""
    # Probe the loopback alias even when binding 0.0.0.0 — a process already
    # bound to the same port on any interface still blocks our bind.
    probe_host = "127.0.0.1" if host == _ALL_INTERFACES else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex((probe_host, port)) == 0:
            raise SystemExit(
                f"ERROR: Port {port} is already in use. "
                f"Kill the existing process first (netstat -ano | findstr :{port})."
            )


def _open_when_up(port: int) -> None:
    """Open the dashboard in the browser once the server accepts connections."""
    for _ in range(600):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                webbrowser.open(f"http://localhost:{port}")
                return
        time.sleep(0.1)


def run_orchestrator(
    port: int | None = None,
    *,
    public: bool = False,
    db_path: Path | None = None,
    open_browser: bool = False,
) -> None:
    """Boot the read-only FastAPI dashboard; blocks in ``uvicorn.run`` until shutdown."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    resolved_port = config.DASHBOARD_DEFAULT_PORT if port is None else port
    resolved_db = config.DB_PATH if db_path is None else db_path
    bind_host = _resolve_bind_host(public=public)
    _check_port_available(resolved_port, bind_host)

    conn = get_connection(resolved_db)
    try:
        ensure_database_ready(conn)
        if latest_played_date(conn) is None:
            raise SystemExit(
                f"{resolved_db} has no games yet. Run update.bat (lol-predict update) first; "
                "building a new database takes about 40 minutes."
            )
    finally:
        conn.close()

    from league_predictor.dashboard.app_factory import create_app

    base_app = create_app(db_path=resolved_db)
    logger.info("Starting dashboard on port %d", resolved_port)

    if open_browser:
        threading.Thread(target=_open_when_up, args=(resolved_port,), daemon=True).start()
    if bind_host == _ALL_INTERFACES:
        logger.warning("Dashboard binding 0.0.0.0:%d — reachable on all interfaces", resolved_port)
    uvicorn.run(base_app, host=bind_host, port=resolved_port, log_level="info")
