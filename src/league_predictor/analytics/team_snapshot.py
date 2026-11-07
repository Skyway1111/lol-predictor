"""Build the daily team mart, including the region-anchored power display."""

from __future__ import annotations

import dataclasses
from datetime import date
from typing import TYPE_CHECKING

from league_predictor.analytics._common import (
    _as_of_iso,
    _safe_json,
    tracked_builder,
)
from league_predictor.analytics.row_types import TeamDailySnapshotRow
from league_predictor.analytics.team_power_display import compute_team_power_display
from league_predictor.db.asof import team_state_asof
from league_predictor.db.names import team_name_lookup
from league_predictor.predraft.features.snapshots import SnapshotStore
from league_predictor.predraft.league_tier import (
    INTERNATIONAL_EVENTS,
    resolve_domestic_league,
)
from league_predictor.predraft.ratings.region_rating import fit_region_ratings

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping


def _latent_team_strength(states: Mapping, team_id: str) -> float:
    """Latent strength from a pre-fetched ``team_state`` batch; 0.0 neutral prior if unseen."""
    row = states.get(team_id)
    if row is None or row.strength is None:
        return 0.0
    return float(row.strength)


def _required_group(parent: Mapping | None, name: str, keys: tuple[str, ...]) -> dict:
    """The named group, zeros only when ABSENT; a present group must carry every key.

    Raises:
        RuntimeError: a present group lacks an indexed key — a renamed producer
            field fails the build loudly instead of silently zeroing a dashboard column.
    """
    group = (parent or {}).get(name)
    if group is None:
        return dict.fromkeys(keys, 0.0)
    missing = [key for key in keys if key not in group]
    if missing:
        raise RuntimeError(
            f"team feature snapshot group {name!r} lacks {missing} — the snapshot producer "
            "and this mart disagree on field names; reconcile the mart's key list"
        )
    return group


def _team_regions(conn: sqlite3.Connection, as_of_iso: str) -> dict[str, str]:
    """First recognized domestic league among each team's five most recent series."""
    placeholders = ", ".join("?" * len(INTERNATIONAL_EVENTS))
    rows = conn.execute(
        f"""
        WITH appearances AS (
            SELECT team_a AS team_id, league_id, scheduled_ts FROM series
            UNION ALL
            SELECT team_b, league_id, scheduled_ts FROM series
        ), ranked AS (
            SELECT team_id, league_id, ROW_NUMBER() OVER (
                PARTITION BY team_id ORDER BY scheduled_ts DESC
            ) AS recency
            FROM appearances
            WHERE date(scheduled_ts) <= ? AND league_id NOT IN ({placeholders})
        )
        SELECT team_id, league_id FROM ranked WHERE recency <= 5
        ORDER BY team_id, recency
        """,
        (as_of_iso, *INTERNATIONAL_EVENTS),
    )
    regions: dict[str, str] = {}
    for row in rows:
        league = resolve_domestic_league(row["league_id"])
        if league is not None:
            regions.setdefault(row["team_id"], league)
    return regions


def _all_team_ids(conn: sqlite3.Connection) -> list[str]:
    """Every team id appearing on either seat of any series, sorted."""
    rows = conn.execute(
        """
        SELECT DISTINCT team_id FROM (
            SELECT team_a AS team_id FROM series
            UNION
            SELECT team_b AS team_id FROM series
        )
        WHERE team_id IS NOT NULL
        ORDER BY team_id
        """
    ).fetchall()
    return [str(row["team_id"]) for row in rows]


def refresh_team_daily_snapshot(
    conn: sqlite3.Connection,
    *,
    as_of_date: date | None = None,
    run_log: list[dict],
) -> int:
    """Replace the ``team_daily_snapshot`` slice for ``as_of_date``; returns row count."""
    from league_predictor.predraft.pipeline.model_identity import current_rating_model_id

    as_of_iso = _as_of_iso(as_of_date)
    as_of = date.fromisoformat(as_of_iso)
    team_ids = _all_team_ids(conn)
    region_table = fit_region_ratings(conn, through=as_of)
    team_regions = _team_regions(conn, as_of_iso)
    team_names = team_name_lookup(conn)
    # One as-of batch (state strictly before ``as_of``); the serving replay already ran
    # in ``ensure_team_state_current``, so no index is built here.
    strength_states = team_state_asof(conn, team_ids, current_rating_model_id(), as_of)
    store = SnapshotStore(conn)  # every team at one day: one bulk load, not one per team
    rows_to_insert = []

    with tracked_builder(conn, "team_daily_snapshot", as_of_iso, run_log) as result:
        for team_id in team_ids:
            snapshot = store.team_snapshot(team_id, as_of)
            roster = snapshot.get("roster") or {}
            form = _required_group(snapshot, "form", ("win_rate_10", "win_streak", "loss_streak"))
            momentum = _required_group(snapshot, "momentum", ("win_rate_5",))
            objective = _required_group(
                snapshot,
                "objective",
                (
                    "dragon_rate",
                    "baron_rate",
                    "tower_rate",
                    "first_dragon_rate",
                    "first_baron_rate",
                    "first_tower_rate",
                ),
            )
            composite = snapshot.get("composite")
            avg_laning = _required_group(
                composite, "avg_laning", ("golddiffat15", "csdiffat15", "xpdiffat15")
            )
            region = team_regions.get(team_id)
            region_elo = region_table.elo(region)
            latent_strength = _latent_team_strength(strength_states, team_id)
            team_power = compute_team_power_display(latent_strength, region_elo)

            rows_to_insert.append(
                TeamDailySnapshotRow(
                    team_id=team_id,
                    date=as_of_iso,
                    display_name=team_names.get(team_id, team_id),
                    region=region,
                    roster_json=_safe_json(roster),
                    win_rate_10=float(form["win_rate_10"]),
                    win_rate_5=float(momentum["win_rate_5"]),
                    win_streak=int(form["win_streak"]),
                    loss_streak=int(form["loss_streak"]),
                    dragon_rate=float(objective["dragon_rate"]),
                    baron_rate=float(objective["baron_rate"]),
                    tower_rate=float(objective["tower_rate"]),
                    first_dragon_rate=float(objective["first_dragon_rate"]),
                    first_baron_rate=float(objective["first_baron_rate"]),
                    first_tower_rate=float(objective["first_tower_rate"]),
                    avg_gold_diff_15=float(avg_laning["golddiffat15"]),
                    avg_cs_diff_15=float(avg_laning["csdiffat15"]),
                    avg_xp_diff_15=float(avg_laning["xpdiffat15"]),
                    power_rating=team_power,
                )
            )

        conn.execute("DELETE FROM team_daily_snapshot WHERE date = ?", (as_of_iso,))
        conn.executemany(
            TeamDailySnapshotRow.insert_sql(),
            [dataclasses.astuple(r) for r in rows_to_insert],
        )
        result["rows"] = len(rows_to_insert)
    return result["rows"]
