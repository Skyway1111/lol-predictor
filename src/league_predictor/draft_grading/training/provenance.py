"""Fit provenance the operator reads at decision time: served priors, freshness, the stamp."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path

from league_predictor.db.predictions_read import oos_blue_probs
from league_predictor.draft_grading.champion_attributes.wiki_export import (
    DEFAULT_WIKI_EXPORT_PATH,
)
from league_predictor.draft_grading.grader.artifact import family_meta_path

logger = logging.getLogger(__name__)


def served_oos_probs(conn: sqlite3.Connection, served_id: str) -> dict[str, float]:
    """Blue-oriented walk-forward pre-draft priors from the served model's ``oos`` corpus.

    Returns:
        ``{game_id: p_blue}`` over draft-scorable games (known blue side and a winner). A
        scorable game with no oos row is OMITTED and counted, never substituted with 0.5.
    """
    probs, omitted = oos_blue_probs(conn, served_id)
    if omitted:
        logger.warning(
            "draft: %d/%d scorable games have NO served-oos prior (served id %s) — "
            "OMITTED from the pre-draft prior, never substituted 0.5",
            omitted,
            len(probs) + omitted,
            served_id,
        )
    logger.info(
        "draft: loaded %d served-oos pre-draft priors (served id %s)",
        len(probs),
        served_id,
    )
    return probs


#: The wiki champion export whose file mtime dates the champion knowledge base —
#: the path home is champion_attributes/wiki_export.py.
DEFAULT_CHAMPION_EXPORT_PATH = DEFAULT_WIKI_EXPORT_PATH


def knowledge_freshness(
    through: date,
    *,
    champion_export_path: Path = DEFAULT_CHAMPION_EXPORT_PATH,
) -> str:
    """Provenance line, never raises: wiki export age in days at ``through``."""
    wiki_age = _file_age_days(champion_export_path, through)
    wiki = f"{wiki_age}d old" if wiki_age is not None else "absent"
    return f"knowledge freshness: wiki export {wiki}"


def _file_age_days(path: Path, through: date) -> int | None:
    """Age in days of ``path`` (its mtime date to ``through``); ``None`` when it is absent."""
    if not path.exists():
        return None
    return (through - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).date()).days


def stamp_fit_provenance(
    output_root: Path,
    fit_through: date,
    *,
    code_fingerprint: str,
) -> None:
    """Record the fit cutoff and source provenance."""
    meta_path = family_meta_path(output_root)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["trained_through"] = fit_through.isoformat()
    meta["code_fingerprint"] = code_fingerprint
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
