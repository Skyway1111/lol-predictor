"""The scalar logit-temperature transform used by the fixed Served recipe."""

from __future__ import annotations

from league_predictor.probability import logit, sigmoid

# Probability clip used before fitting or applying the scalar logit transform.
_EPS = 1e-6


def apply_temperature(p: float, s: float) -> float:
    """Rescale log-odds by ``s``: ``sigmoid(s * logit(clip(p)))``; exact identity at ``s == 1``."""
    if s == 1.0:
        return p
    return sigmoid(s * logit(p, eps=_EPS))
