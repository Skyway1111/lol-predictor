"""Transaction and run-log helpers shared by the analytics mart builders."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import date

from league_predictor import clock


@contextmanager
def tracked_builder(
    conn: sqlite3.Connection,
    builder_name: str,
    as_of_iso: str,
    run_log: list[dict],
) -> Iterator[dict]:
    """Run one mart builder in a SAVEPOINT inside the caller's open transaction."""
    # Status is appended to ``run_log`` in memory only; ``_write_run_log`` commits it
    # after the generation resolves. Requires an open transaction, owns no commit.
    result: dict = {"as_of_date": as_of_iso}
    savepoint = f'"builder_{builder_name}"'
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        yield result
    except Exception as exc:
        conn.execute(f"ROLLBACK TO {savepoint}")
        conn.execute(f"RELEASE {savepoint}")
        run_log.append(
            {
                "builder": builder_name,
                "status": "error",
                "details": {"as_of_date": as_of_iso, "error": str(exc)},
            }
        )
        raise
    else:
        conn.execute(f"RELEASE {savepoint}")
        run_log.append({"builder": builder_name, "status": "ok", "details": result})


def _write_run_log(conn: sqlite3.Connection, runs: Iterable[dict]) -> None:
    """Commit ``analytics_runs`` rows separately so an error status survives a rollback."""
    conn.executemany(
        "INSERT INTO analytics_runs (builder, completed_at, status, details_json) "
        "VALUES (?, datetime('now'), ?, ?)",
        [
            (r["builder"], r["status"], json.dumps(r.get("details") or {}, sort_keys=True))
            for r in runs
        ],
    )
    conn.commit()


def _as_of_iso(as_of_date: date | None) -> str:
    """ISO day for a builder run — ``as_of_date`` or today."""
    return (as_of_date or clock.utc_today()).isoformat()


def _safe_json(value) -> str:
    """Deterministic JSON (sorted keys) for snapshot columns."""
    return json.dumps(value, sort_keys=True)
