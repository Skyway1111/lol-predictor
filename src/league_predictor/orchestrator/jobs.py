"""The sequences the CLI runs: update, retrain, and the upcoming-match sync."""

from __future__ import annotations

import contextlib
import os
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from league_predictor import clock, config, db
from league_predictor.db import get_connection
from league_predictor.upcoming import pandascore, service
from league_predictor.upcoming.matching import map_team_manually

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Collection, Iterator, Mapping


# sightline-ok: 11 - this horizon must EXCLUDE unfinished games; live.index_through must not
def _latest_data_horizon(conn: sqlite3.Connection) -> date:
    """Return the exclusive day after the latest finished game, or today for no data."""
    latest = db.latest_played_date(conn)
    return latest + timedelta(days=1) if latest is not None else clock.utc_today()


def _guard_ingestion_quality(ingestion: Mapping[str, Any]) -> None:
    """Raise ``RuntimeError`` naming every failure when the ingest is dirty."""
    errors = ingestion.get("errors") or []
    collisions = ingestion.get("unadjudicated_player_collisions") or []
    if not errors and not collisions:
        return
    problems = []
    if errors:
        problems.append(f"{len(errors)} ingestion error(s): {errors}")
    if collisions:
        problems.append(
            f"{len(collisions)} unadjudicated OE handle collision(s) "
            f"(adjudicate in ingestion/player_reconciler.py): {collisions}"
        )
    raise RuntimeError(
        "Refusing to propagate a dirty ingest into analytics marts — "
        + "; ".join(problems)
        + ". Re-run with --allow-dirty-ingest to proceed deliberately."
    )


def _ingest(
    conn: sqlite3.Connection, *, full: bool, allow_dirty: bool, say: Callable[[str], None]
) -> None:
    """Load new match facts; a dirty ingest stops the run unless ``allow_dirty``."""
    from league_predictor.ingestion.service import run_all, run_current_year

    say("Ingesting every configured year..." if full else f"Ingesting {clock.utc_today().year}...")
    result = run_all(conn) if full else run_current_year(conn)
    say(f"Ingested {result['matches_inserted']} new matches, {len(result['errors'])} errors.")
    if not allow_dirty:
        _guard_ingestion_quality(result)


def _refresh_analytics(conn: sqlite3.Connection, *, say: Callable[[str], None]) -> None:
    """Rebuild the dashboard tables from the current facts and model outputs."""
    from league_predictor.analytics.builders import refresh_analytics_marts

    say("Rebuilding dashboard tables...")
    result = refresh_analytics_marts(conn)
    say(
        f"Done: {result['team_daily_snapshot']} team and "
        f"{result['player_daily_snapshot']} player snapshots."
    )


@contextlib.contextmanager
def _pipeline_lock() -> Iterator[None]:
    """One ``update`` or ``retrain`` at a time: both rewrite the models and the database."""
    path = Path(config.MODELS_DIR) / ".pipeline.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise RuntimeError(f"another update or retrain is running ({path})") from None
        yield  # closing the handle releases the lock


@contextlib.contextmanager
def _ingested(
    db_path: Path | str | None, *, full: bool, allow_dirty: bool, say: Callable[[str], None]
) -> Iterator[tuple[sqlite3.Connection, date]]:
    """Lock, ingest new games, yield (conn, the day after the latest game), then rebuild marts."""
    with _pipeline_lock():
        conn = get_connection(db_path if db_path is not None else config.DB_PATH)
        try:
            db.ensure_database_ready(conn)
            empty = db.latest_played_date(conn) is None
            _ingest(conn, full=full or empty, allow_dirty=allow_dirty, say=say)
            yield conn, _latest_data_horizon(conn)
            _refresh_analytics(conn, say=say)
        finally:
            conn.close()


def update(
    db_path: Path | str | None = None,
    *,
    full: bool = False,
    allow_dirty: bool = False,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Ingest new games, score them with the current models, rebuild the dashboard tables.

    COST: about 8 minutes; 35 the first time on a database, which writes each model's
    walk-forward history once. A fresh database also downloads every configured year.
    """
    from league_predictor import lifecycle

    say = progress or (lambda _msg: None)
    with _ingested(db_path, full=full, allow_dirty=allow_dirty, say=say) as (conn, through):
        for token in lifecycle.ORDER:
            say(f"Scoring games with {token}...")
            written = lifecycle.get(token).backfill(conn, through=through)
            history = " with walk-forward history" if written.full else ""
            say(f"  {written.rows} rows under {written.model_id}{history}.")


#: Retraining a model invalidates these: served pins the exact rating artifact it was fit on.
_DEPENDENTS = {"rating": ("served",)}


def retrain(
    tokens: Collection[str],
    db_path: Path | str | None = None,
    *,
    allow_dirty: bool = False,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Ingest new games, refit ``tokens`` and their dependents, rewrite their outputs.

    COST: about 40 minutes for all four models; a GPU speeds the rating and draft fits.
    """
    from league_predictor import lifecycle

    say = progress or (lambda _msg: None)
    chosen = set(tokens)
    for token in tokens:
        chosen.update(_DEPENDENTS.get(token, ()))
    with _ingested(db_path, full=False, allow_dirty=allow_dirty, say=say) as (conn, through):
        for token in (t for t in lifecycle.ORDER if t in chosen):
            say(f"Retraining {token} on games before {through}...")
            written = lifecycle.get(token).retrain(conn, through=through)
            say(f"  {token} is now {written.model_id} ({written.rows} rows).")


def sync_upcoming(
    db_path: Path | str | None = None, *, progress: Callable[[str], None] | None = None
) -> dict[str, Any]:
    """Sync the PandaScore schedule, pair realized series, score the rest. COST: minutes of CPU."""

    say = progress or (lambda _msg: None)
    if not config.PANDASCORE_TOKEN:
        raise RuntimeError(
            'PandaScore token missing — put PANDA="<key>" in panda.env at the project root'
        )
    now = clock.utc_now().replace(tzinfo=UTC)
    say("Fetching the PandaScore upcoming-matches feed...")
    matches = pandascore.fetch_upcoming(config.PANDASCORE_TOKEN, now=now)
    say(f"Fetched {len(matches)} scheduled matches.")

    conn = get_connection(db_path if db_path is not None else config.DB_PATH)
    try:
        db.ensure_database_ready(conn)
        report = service.sync_schedule(conn, matches, now=now)
        paired = service.pair_realized(conn, now=now)
        # Schedule + pairing commit before scoring, so a model failure never loses the sync.
        conn.commit()
        say(
            f"Schedule synced: {report.upserted} upserted, {paired} paired to played series, "
            f"{report.canceled} canceled."
        )
        predicted = service.predict_unfinished(
            conn, now=now, load_predictor=lambda: service.live_served_predictor(conn)
        )
        conn.commit()
        say(f"Predicted {predicted} unfinished matches.")
    finally:
        conn.close()
    return {
        "fetched": len(matches),
        "upserted": report.upserted,
        "canceled": report.canceled,
        "paired": paired,
        "predicted": predicted,
        "unmatched_teams": report.unmatched,
    }


def map_upcoming_team(ps_team_id: int, team_id: str, db_path: Path | str | None = None) -> None:
    """Hand-map one PandaScore team onto a canonical team id (the unmatched-team remedy)."""

    conn = get_connection(db_path if db_path is not None else config.DB_PATH)
    try:
        db.ensure_database_ready(conn)
        map_team_manually(conn, ps_team_id, team_id)
        conn.commit()
    finally:
        conn.close()
