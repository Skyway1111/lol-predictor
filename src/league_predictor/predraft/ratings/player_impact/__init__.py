"""Roster-change Impact: one per-player rating in win units; analytical, never served."""

# Flow: recipe (frozen knobs) -> predictor (game_eta) -> design (matrix, solve)
# -> results (fit containers) -> fit (the DB-facing orchestration).

from league_predictor.predraft.ratings.player_impact.design import replacement_key
from league_predictor.predraft.ratings.player_impact.fit import (
    fit_player_impact,
    fit_prepared_impact,
    prepare_impact_fit,
)

# The evaluator's recipe-bound preparation entry: private to the package, but part of the
# surface this facade keeps stable for `impact_operations`.
from league_predictor.predraft.ratings.player_impact.predictor import (
    ROSTER_SIZE,
    ImpactValues,
    RegionLevels,
    game_eta,
)
from league_predictor.predraft.ratings.player_impact.recipe import (
    IMPACT_RECIPE,
    ImpactRecipe,
    StatContextKwargs,
)
from league_predictor.predraft.ratings.player_impact.results import (
    ImpactEstimate,
    ImpactFit,
    ImpactFitContext,
)

__all__ = [
    "IMPACT_RECIPE",
    "ROSTER_SIZE",
    "ImpactEstimate",
    "ImpactFit",
    "ImpactFitContext",
    "ImpactRecipe",
    "ImpactValues",
    "RegionLevels",
    "StatContextKwargs",
    "fit_player_impact",
    "fit_prepared_impact",
    "game_eta",
    "prepare_impact_fit",
    "replacement_key",
]
