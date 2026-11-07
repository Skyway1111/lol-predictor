"""The Impact artifact: its persisted recipe and its readers."""

from __future__ import annotations

import json
from datetime import date, timedelta
from typing import TYPE_CHECKING

from league_predictor.db.asof import impact_score_asof_scalar
from league_predictor.predraft.ratings.player_impact import (
    IMPACT_RECIPE,
    ImpactRecipe,
    StatContextKwargs,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

#: The model token -- identical everywhere (module name, CLI arg, registry id kind).
token = "impact"


# ── The persisted recipe ────────────────────────────────────────────────────────


def production_recipe() -> ImpactRecipe:
    """The artifact's recipe, sole source of serving-surface recipe facts; source if none."""
    from league_predictor import artifacts

    raw = artifacts.persisted_recipe(token)
    return IMPACT_RECIPE if raw is None else _validated_recipe(raw, source="the impact artifact")


def production_stat_context_kwargs() -> StatContextKwargs:
    """The artifact recipe's stat-context build kwargs."""
    from league_predictor.predraft.ratings.player_impact.recipe import STAT_CONTEXT_KWARGS

    production_recipe()
    return STAT_CONTEXT_KWARGS.copy()


# ── Materialized output rows ────────────────────────────────────────────────────


#: Exactly the per-player fields ``player_score_rows`` reads into ``player_impact_scores``.
_SCORE_FIELDS = frozenset({"role", "impact_pp", "stats_pp", "results_pp", "percentile", "grade"})
#: Every published Impact field a dashboard payload carries, in ``player_impact_scores``
#: column vocabulary; ``asof_date`` rides along as ``impact_asof``.
IMPACT_PAYLOAD_COLUMNS: tuple[str, ...] = (
    "impact",
    "stats_pp",
    "results_pp",
    "percentile",
    "grade",
)


def impact_asof_payload_select(player_expr: str) -> str:
    """Select-list SQL for the Impact payload; bind with :func:`impact_asof_payload_params`.

    Returns:
        :data:`IMPACT_PAYLOAD_COLUMNS` plus ``asof_date AS impact_asof``, each through
        ``db.asof``'s on-or-before rule.
    """

    selects = [
        f"{impact_score_asof_scalar(column, player_expr)} AS {column}"
        for column in IMPACT_PAYLOAD_COLUMNS
    ]
    selects.append(f"{impact_score_asof_scalar('asof_date', player_expr)} AS impact_asof")
    return ", ".join(selects)


def impact_asof_payload_params(asof_date: str, *, player_id: str | None = None) -> tuple[str, ...]:
    """The payload select's bound params; pass ``player_id`` iff composed with ``'?'``."""
    per_fragment = (asof_date,) if player_id is None else (player_id, asof_date)
    return per_fragment * (len(IMPACT_PAYLOAD_COLUMNS) + 1)


def _require_player_fields(
    artifact_path: Path, players: Mapping[str, dict], fields: frozenset[str]
) -> None:
    """Assert every player entry carries ``fields``.

    Raises:
        ValueError: an entry is short a field this reader indexes. Asserted at
            the read site, because a silently absent field degrades into NULLs or a wrong eta.
    """
    missing = [pid for pid, pdata in players.items() if not fields.issubset(pdata)]
    if missing:
        raise ValueError(
            f"{artifact_path}: {len(missing)} of {len(players)} player entries lack one of "
            f"{sorted(fields)} (first: {missing[0]!r}) — refit the slot"
        )


def player_score_rows(*, model_id: str, artifact_path: Path) -> tuple[tuple, ...]:
    """Return the artifact's rows for lifecycle-owned materialization.

    Raises:
        ValueError: a player entry lacks a published field. Asserted, because
            a silently absent headline field would materialize a table of NULL scores.
    """
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    # A row stamped D means "incorporates games through end of day D" (db/asof.py reads
    # asof_date <= ?); the fit's as_of is the strictly-before cutoff, one day later.
    asof = (date.fromisoformat(artifact["as_of_date"]) - timedelta(days=1)).isoformat()
    players: dict[str, dict] = artifact["players"]
    _require_player_fields(artifact_path, players, _SCORE_FIELDS)
    return tuple(
        (
            pid,
            model_id,
            asof,
            pdata["role"],
            pdata["impact_pp"],
            pdata["stats_pp"],
            pdata["results_pp"],
            pdata["percentile"],
            pdata["grade"],
        )
        for pid, pdata in players.items()
    )


def _validated_recipe(raw, *, source: str) -> ImpactRecipe:
    """Require the supported recipe at the artifact boundary."""
    if raw != IMPACT_RECIPE.to_dict():
        raise ValueError(f"{source}: unsupported Impact recipe {raw!r}; retrain impact")
    return IMPACT_RECIPE
