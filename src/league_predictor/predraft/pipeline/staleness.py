"""The staleness dampener: a serve-path guard that humbles a scheduled match with an idle team."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import TYPE_CHECKING, cast

from league_predictor.probability import logit, sigmoid

if TYPE_CHECKING:
    from league_predictor.predraft.features import FamilyIndex

    # sightline-ok: 32 - cast's quoted target is required by Ruff TC006 and used for type narrowing
    from league_predictor.predraft.features.lineup import _LineupIndex

STALE_AFTER_DAYS = 30  # a team idle longer than this (since its last game) is "stale"
STALE_LOGIT_SCALE = 0.5  # stale -> multiply the logit by this (0.5 = halve the confidence)
_EPS = 1e-9  # logit saturation for the shrink


def dampen_if_stale(p: float, days_since_a: float, days_since_b: float) -> float:
    """Pull ``p`` toward 0.5 when either team is idle past ``STALE_AFTER_DAYS`` (``inf`` = none)."""
    if max(days_since_a, days_since_b) <= STALE_AFTER_DAYS:
        return p
    return sigmoid(logit(p, eps=_EPS) * STALE_LOGIT_SCALE)


def dampen_stale_matchup(
    p: float,
    indexes: Mapping[str, FamilyIndex],
    team_a: str,
    team_b: str,
    *,
    as_of: date,
) -> float:
    """Apply :func:`dampen_if_stale` to a live matchup; identity without a ``lineup`` index."""
    lineup = cast("_LineupIndex | None", indexes.get("lineup"))
    if lineup is None:
        return p
    return dampen_if_stale(
        p,
        lineup.days_since_last_game(team_a, as_of),
        lineup.days_since_last_game(team_b, as_of),
    )
