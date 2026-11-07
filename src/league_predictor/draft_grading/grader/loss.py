"""Importance weights over covered outcome priors."""

import numpy as np


def compute_importance_weights(
    pre_draft_probs: np.ndarray,
    outcomes: np.ndarray,
    *,
    alpha: float = 2.0,
    gamma: float = 1.5,
    base: float = 0.1,
) -> np.ndarray:
    """Importance weights normalized to mean 1; close games and upsets weigh most."""
    closeness = 1.0 - 2.0 * np.abs(pre_draft_probs - 0.5)
    surprise = np.abs(outcomes - pre_draft_probs)
    raw = (closeness + alpha * surprise) ** gamma + base
    return raw / raw.mean()
