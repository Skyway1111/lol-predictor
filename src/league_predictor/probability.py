"""Dependency-leaf probability and log-odds transforms."""

from __future__ import annotations

import math

import numpy as np

ELO_ORIGIN = 1500.0
ELO_PER_LOGODDS = 400.0 / math.log(10.0)  # sightline-ok: 26 - base-10 to natural-log scale


def sigmoid(z: float) -> float:
    """Return the numerically stable logistic transform."""
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    exp_z = math.exp(z)
    return exp_z / (1.0 + exp_z)


def logit(p: float, *, eps: float = 1e-12) -> float:
    """Return probability as log-odds with caller-controlled saturation."""
    bounded = min(1.0 - eps, max(eps, p))
    return math.log(bounded / (1.0 - bounded))


def sigmoid_array(x: np.ndarray | float) -> np.ndarray:
    """Return elementwise ``1/(1+exp(-x))`` over float64 — the non-branching array form."""
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def sigmoid_mirrored(x: np.ndarray | float) -> np.ndarray:
    """Return ``sigmoid(x)`` off the favoured side, so ``f(-x) == 1 - f(x)`` holds bitwise."""
    z = np.asarray(x, dtype=np.float64)
    favoured = sigmoid_array(np.abs(z))
    return np.where(z >= 0.0, favoured, 1.0 - favoured)


def logit_array(p: np.ndarray | float, *, eps: float) -> np.ndarray:
    """Return elementwise log-odds of ``p`` clipped into ``[eps, 1-eps]``, over float64."""
    clipped = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(clipped / (1.0 - clipped))


def orient_blue(p_a: float, blue_team: str, team_a: str, team_b: str) -> float | None:
    """Reorient canonical ``p_a`` (P(team_a)) to P(blue); None when blue is neither series team."""
    if blue_team == team_a:
        return p_a
    if blue_team == team_b:
        return 1.0 - p_a
    return None
