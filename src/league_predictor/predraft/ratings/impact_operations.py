"""The Impact model's fit and history reconstruction."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from league_predictor import clock, db
from league_predictor.db import registry
from league_predictor.predraft.ratings.impact_corpus import build_impact_corpus
from league_predictor.predraft.ratings.impact_history import (
    ImpactHistoryBuild,
    build_impact_history,
)
from league_predictor.predraft.ratings.player_impact import (
    IMPACT_RECIPE,
    fit_player_impact,
)

if TYPE_CHECKING:
    import sqlite3


def build_current_history(
    conn: sqlite3.Connection, *, through: date | None = None
) -> ImpactHistoryBuild:
    """Build every leak-free historical state through the latest played date in the window."""
    anchor = db.latest_played_date(conn, through=through)
    if anchor is None:
        return ImpactHistoryBuild((), 0, 0, None, None)
    return build_impact_history(build_impact_corpus(conn), through=anchor)


def fit_candidate(conn: sqlite3.Connection, *, through: date | None = None) -> dict[str, object]:
    """Fit the frozen Impact recipe and return a JSON-ready artifact without publishing."""
    latest = db.latest_played_date(conn, through=through)
    as_of = latest + timedelta(days=1) if latest is not None else clock.utc_today()
    impactfit = fit_player_impact(conn, as_of)
    artifact = impactfit.to_artifact()
    artifact["recipe"] = IMPACT_RECIPE.to_dict()
    artifact["code_fingerprint"] = registry.code_fingerprint()
    return artifact
