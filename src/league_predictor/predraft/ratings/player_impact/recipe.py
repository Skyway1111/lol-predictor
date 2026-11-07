"""The frozen Impact recipe and the per-version facets keyed off its version."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TypedDict


@dataclass(frozen=True, slots=True)
class ImpactRecipe:
    """Zero-knob production recipe for corpus influence and named-player qualification."""

    version: int
    influence_window_months: int | None
    qualification_window_months: int | None
    half_life_days: float
    min_games: int
    history_buckets_per_role: int

    def to_dict(self) -> dict[str, int | float | None]:
        """Return the exact JSON artifact representation used by lifecycle provenance."""
        return asdict(self)


# v9 prices every player's level from the learned region-rating table exactly by
# league (v8 read the snapshot regional Elo through the region normalization map).
IMPACT_RECIPE = ImpactRecipe(
    version=9,
    influence_window_months=None,
    qualification_window_months=24,
    half_life_days=450.0,
    min_games=25,
    history_buckets_per_role=64,
)


#: A recipe version's stat-context build kwargs (versioned alongside the recipe).
class StatContextKwargs(TypedDict):
    lane_bt: bool


STAT_CONTEXT_KWARGS: StatContextKwargs = {"lane_bt": True}
PRIOR_BLOCKS = {"base": 60.0}
