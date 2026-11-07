"""Load the current served artifact into live as-of feature indexes."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor import clock
from league_predictor.predraft.features import FEATURE_FAMILIES
from league_predictor.predraft.pipeline.chosen_model import ServedModel
from league_predictor.predraft.pipeline.chosen_skill_filter import chosen_rating_dependency

if TYPE_CHECKING:
    from league_predictor.predraft.features import FamilyIndex


# sightline-ok: 11 - a SCHEDULED-series horizon; jobs._latest_data_horizon reads FINISHED games
def index_through(conn: sqlite3.Connection) -> date:
    """Return the day after the latest scheduled series for live index coverage."""
    row = conn.execute(
        "SELECT MAX(scheduled_ts) FROM series WHERE scheduled_ts IS NOT NULL"
    ).fetchone()
    if row is None or row[0] is None:
        return clock.utc_today()
    return date.fromisoformat(str(row[0])[:10]) + timedelta(days=1)


def build_live_indexes(conn: sqlite3.Connection, model: ServedModel) -> dict[str, FamilyIndex]:
    """Build live indexes while keeping the corpus-wide original family lazy."""
    through = index_through(conn)
    families = [FEATURE_FAMILIES[name]() for name in model.families]
    return {
        family.name: (
            family.fallback_index(conn)  # pyright: ignore[reportAttributeAccessIssue]
            if family.name == "original"
            else family.build(conn, through=through, query_dates=[])
        )
        for family in families
    }


def load_live_oracle(conn: sqlite3.Connection, artifact_path: Path) -> tuple[ServedModel, dict]:
    """Load one served artifact and its live as-of indexes (COST: minutes of CPU)."""
    # The original-family index retains ``conn``, so the caller owns its lifetime.
    model = ServedModel.load(artifact_path, actual_rating_dependency=chosen_rating_dependency())
    return model, build_live_indexes(conn, model)
