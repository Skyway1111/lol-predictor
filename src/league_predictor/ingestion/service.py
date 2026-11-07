"""Oracle's Elixir ingestion, identity reconciliation, and collision reporting."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from league_predictor import clock
from league_predictor.db.rebuild_guard import require_rebuild_schema
from league_predictor.ingestion.oracles_elixir import download_csv
from league_predictor.ingestion.player_reconciler import (
    fold_player_ids,
    warn_unadjudicated_collisions,
)
from league_predictor.ingestion.roster_builder import build_rosters_from_player_games

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

# Bump when the OE parse/insert path starts extracting MORE from the same CSV bytes
# (new columns, new derived stats) -- a byte-unchanged CSV must then re-ingest anyway.
_INGEST_PARSER_VERSION = 1


def _ingest_marker_value(csv_path: Path) -> str:
    """What the ``.ingested`` sidecar records: parser version + CSV content hash."""
    from league_predictor.ingestion.oracles_elixir import csv_content_hash

    return f"v{_INGEST_PARSER_VERSION}:{csv_content_hash(csv_path)}"


def _db_has_year(conn: sqlite3.Connection, year: int) -> bool:
    """Whether THIS DB holds a game in ``year`` (a series' ``scheduled_ts`` day joined to games).

    Returns:
        True iff a game falls in the calendar year -- checks the ``.ingested`` sidecar's
        claim against the DB, since a fresh DB with a persistent RAW_DIR must re-ingest.
    """
    row = conn.execute(
        "SELECT 1 FROM series s JOIN games g ON g.series_id = s.series_id "
        "WHERE s.scheduled_ts >= ? AND s.scheduled_ts < ? LIMIT 1",
        (f"{year}-01-01", f"{year + 1}-01-01"),
    ).fetchone()
    return row is not None


def _run_years(conn: sqlite3.Connection, years: list[int]) -> dict[str, Any]:
    """Shared daily ingestion: download OE CSVs and write the facts tables for the years.

    Returns:
        Counts of inserted games, per-year errors (one failed year never blocks the
        others), years skipped as byte-unchanged, and unadjudicated handle collisions.
    """
    from league_predictor.ingestion.oracles_elixir import ingest_oe_results, load_csv

    # Refuse a DB missing the facts tables loudly at the door rather than dying
    # mid-run on a missing table.
    require_rebuild_schema(
        conn, "series", "games", "game_players", purpose="Oracle's Elixir daily ingest"
    )

    total_inserted = 0
    errors = []
    skipped_unchanged: list[int] = []
    # Markers are written only after every year's facts write succeeds, so a crash mid-run
    # leaves no marker and the next run re-ingests (idempotent upsert-by-natural-key).
    pending_markers: list[tuple[Path, str]] = []

    def _download(year: int) -> tuple[int, Path | None, str | None]:
        try:
            logger.info("Downloading Oracle's Elixir data for %d", year)
            return year, download_csv(year), None
        except Exception as exc:
            logger.exception("Failed to download year %d", year)
            return year, None, str(exc)

    if len(years) > 1:
        with ThreadPoolExecutor(max_workers=min(len(years), 4)) as pool:
            download_results = list(pool.map(_download, years))
    else:
        download_results = [_download(year) for year in years]

    for year, csv_path, download_error in download_results:
        if download_error is not None:
            errors.append({"year": year, "error": download_error})
            continue
        assert csv_path is not None
        try:
            # Skip parse + insert when the bytes were already ingested; the sidecar marker is
            # trusted only when the DB actually holds the year it claims (fresh DB, stale RAW_DIR).
            content_hash = _ingest_marker_value(csv_path)
            marker = Path(f"{csv_path}.ingested")
            if (
                marker.exists()
                and marker.read_text().strip() == content_hash
                and _db_has_year(conn, year)
            ):
                logger.info("Year %d: OE CSV unchanged since last ingest; skipping", year)
                skipped_unchanged.append(year)
                continue
            df = load_csv(csv_path)
            inserted = ingest_oe_results(conn, df)
            pending_markers.append((marker, content_hash))
            total_inserted += inserted
            logger.info("Year %d: %d games", year, inserted)
        except Exception as exc:
            logger.exception("Failed to ingest year %d", year)
            errors.append({"year": year, "error": str(exc)})

    # Reconcile the fresh facts only when a year actually (re-)ingested. Fold new remints
    # before rebuilding the roster state the fold purges.
    if pending_markers:
        fold_player_ids(conn)
        build_rosters_from_player_games(conn)
    collisions = warn_unadjudicated_collisions(conn)

    # Markers land only after a clean facts write of the changed years. No post-ingest format
    # repair: best_of / winner / status are derived at ingest by ``ingestion.facts_transform``.
    for marker, content_hash in pending_markers:
        marker.write_text(content_hash)
    return {
        "matches_inserted": total_inserted,
        "errors": errors,
        "years_skipped_unchanged": skipped_unchanged,
        "unadjudicated_player_collisions": collisions,
    }


def run_all(conn: sqlite3.Connection) -> dict[str, Any]:
    """Download and ingest Oracle's Elixir data for all configured years."""
    from league_predictor import config

    return _run_years(conn, list(config.OE_YEARS))


def run_current_year(conn: sqlite3.Connection, year: int | None = None) -> dict[str, Any]:
    """Download and ingest the current Oracle's Elixir year."""
    if year is None:
        year = clock.utc_today().year
    return _run_years(conn, [year])
