"""Compute, rank, and compare the daily region-anchored team power display."""

from __future__ import annotations

import dataclasses
import sqlite3
from datetime import date

from league_predictor.analytics._common import _as_of_iso, tracked_builder
from league_predictor.analytics.row_types import TeamPowerDisplayRow
from league_predictor.probability import ELO_PER_LOGODDS

# Latent team strength → power display: the shared transform centers the skill
# filter's latent strength at 1500, and this adds the league strength anchor.


def compute_team_power_display(strength: float, region_elo: float) -> float:
    """Combine the 1500-centered team display with its league strength anchor."""
    return strength * ELO_PER_LOGODDS + region_elo


def refresh_team_power_display(
    conn: sqlite3.Connection,
    *,
    as_of_date: date | None = None,
    run_log: list[dict],
) -> int:
    """Replace the ``power_rating_snapshots`` slice for ``as_of_date``; returns row count."""
    as_of_iso = _as_of_iso(as_of_date)
    rows = conn.execute(
        """
        SELECT team_id, display_name, region, power_rating
        FROM team_daily_snapshot
        WHERE date = ?
        ORDER BY power_rating DESC, display_name ASC
        """,
        (as_of_iso,),
    ).fetchall()
    rows_to_insert = []

    with tracked_builder(conn, "power_rating_snapshots", as_of_iso, run_log) as result:
        for idx, row in enumerate(rows, start=1):
            team_id = str(row["team_id"])
            rating = float(row["power_rating"])
            rows_to_insert.append(
                TeamPowerDisplayRow(
                    date=as_of_iso,
                    team_id=team_id,
                    display_name=row["display_name"],
                    region=row["region"],
                    rating=rating,
                    global_rank=idx,
                )
            )
        conn.execute("DELETE FROM power_rating_snapshots WHERE date = ?", (as_of_iso,))
        conn.executemany(
            TeamPowerDisplayRow.insert_sql(),
            [dataclasses.astuple(r) for r in rows_to_insert],
        )
        result["rows"] = len(rows_to_insert)
    return result["rows"]
