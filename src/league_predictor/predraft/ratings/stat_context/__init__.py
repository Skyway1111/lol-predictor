"""The one context-adjusted stat home: walk-forward per-game z-scores and aggregates."""

# Flow: vocab (frozen grid dials) -> appearances (one facts scan) -> grid
# (walk-forward estimation) -> prior (per-player aggregates, team-sum weights).

from league_predictor.predraft.ratings.stat_context.grid import build_game_context_z
from league_predictor.predraft.ratings.stat_context.prior import (
    aggregate_prior,
    load_team_sum_weights,
    team_sum_prior,
    team_sum_weights_path,
    walk_forward_pillar_aggregates,
)
from league_predictor.predraft.ratings.stat_context.vocab import (
    GRID_EPOCH,
    GRID_STEP_DAYS,
    GRID_WINDOW_DAYS,
    PILLARS,
    PRIOR_HALF_LIFE_DAYS,
    PRIOR_SHRINKAGE_K,
    STATS,
    ContextZ,
    PlayerPrior,
    game_kda,
)

__all__ = [
    "GRID_EPOCH",
    "GRID_STEP_DAYS",
    "GRID_WINDOW_DAYS",
    "PILLARS",
    "PRIOR_HALF_LIFE_DAYS",
    "PRIOR_SHRINKAGE_K",
    "STATS",
    "ContextZ",
    "PlayerPrior",
    "aggregate_prior",
    "build_game_context_z",
    "game_kda",
    "load_team_sum_weights",
    "team_sum_prior",
    "team_sum_weights_path",
    "walk_forward_pillar_aggregates",
]
