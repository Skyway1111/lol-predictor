"""The fit's containers: one prepared cutoff, one player estimate, one published artifact."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import TYPE_CHECKING

from league_predictor.probability import sigmoid_array

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    import numpy as np
    from scipy import sparse

    from league_predictor.predraft.ratings.impact_corpus import ImpactGame
    from league_predictor.predraft.ratings.player_impact.predictor import (
        ImpactValues,
        RegionLevels,
    )
    from league_predictor.predraft.ratings.player_impact.recipe import ImpactRecipe


# ── Win-units conversion ────────────────────────────────────────────────────────


def _sigmoid(x: float) -> float:
    """Logistic of ``x``."""
    return float(sigmoid_array(x))


def _value_to_pp(value: float, role_reference: float, premier_mean: float) -> float:
    """Win% this value adds over ``role_reference``, both slotted onto four premier averages."""
    return 100.0 * (_sigmoid(value - premier_mean) - _sigmoid(role_reference - premier_mean))


def _percentile_of(reference: Sequence[float], value: float) -> float:
    """Percent of the sorted reference distribution strictly below ``value``."""
    if not reference:
        return 50.0
    return round(100.0 * bisect_left(reference, value) / len(reference), 1)


# ── Result containers ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ImpactEstimate:
    """One player's fitted value and the display decomposition published beside it."""

    player_id: str
    role: str
    region: str
    level: float
    n_games: int
    n_distinct_teammates: int
    prior_logodds: float
    delta: float
    value_logodds: float
    display_logodds: float
    impact_pp: float
    stats_pp: float
    results_pp: float
    percentile: float
    grade: str


@dataclass(frozen=True)
class ImpactFit:
    """The fitted predictor, its per-player estimates, and the fit's own scalars."""

    values: ImpactValues
    scores: dict[str, ImpactEstimate]
    gamma: float
    alpha: float
    beta: float
    lam: float
    cv_logloss: float
    premier_mean_logodds: float
    premier_role_means: dict[str, float]
    region_exact_fraction: float
    n_games: int
    as_of_date: str

    def to_artifact(self) -> dict[str, object]:
        """JSON-serializable dict for models/impact/impact.json."""
        return {
            "as_of_date": self.as_of_date,
            "lambda": self.lam,
            "cv_logloss": self.cv_logloss,
            "n_games": self.n_games,
            "gamma": self.gamma,
            "alpha": self.alpha,
            "beta": self.beta,
            "premier_mean_logodds": self.premier_mean_logodds,
            "premier_role_means": self.premier_role_means,
            "region_exact_fraction": round(self.region_exact_fraction, 4),
            "default_level": round(self.values.region.default_level, 6),
            "players": {
                pid: {
                    "role": r.role,
                    "region": r.region,
                    "level": round(r.level, 4),
                    "n_games": r.n_games,
                    "n_distinct_teammates": r.n_distinct_teammates,
                    "prior_logodds": round(r.prior_logodds, 4),
                    "delta": round(r.delta, 4),
                    "value_logodds": round(r.value_logodds, 4),
                    "display_logodds": round(r.display_logodds, 4),
                    "impact_pp": round(r.impact_pp, 2),
                    "stats_pp": round(r.stats_pp, 2),
                    "results_pp": round(r.results_pp, 2),
                    "percentile": r.percentile,
                    "grade": r.grade,
                }
                for pid, r in self.scores.items()
            },
        }


@dataclass(frozen=True)
class ImpactFitContext:
    """Expensive cutoff-specific Impact inputs shared by fits that differ only in lambda."""

    as_of_date: date
    recipe: ImpactRecipe
    games: Sequence[ImpactGame]
    player_role: dict[str, str]
    col_index: dict[str, int]
    colkey_of: dict[str, str]
    column_scale_of: dict[str, float]
    n_games: dict[str, int]
    n_distinct: dict[str, int]
    qualified: tuple[str, ...]
    skill_priors: dict[str, float]
    player_region: dict[str, str]
    region: RegionLevels
    region_exact_fraction: float
    X: sparse.csr_matrix
    y: np.ndarray
    sample_weight: np.ndarray
    val_mask: np.ndarray
    penalty_mask: np.ndarray
