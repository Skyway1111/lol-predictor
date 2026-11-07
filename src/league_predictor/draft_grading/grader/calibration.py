"""Bag logit-temperature calibration for the draft head: one scalar on the averaged logit."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from league_predictor.probability import logit_array, sigmoid_array

# Prob clip for the logit — how extreme a prob's log-odds is allowed to get.
_EPS = 1e-6
# Below this the whole log-odds spread is float noise: the corpus carries no temperature.
_FLAT_LOGIT = 1e-12


def apply_temperature(probs: float | np.ndarray, s: float) -> float | np.ndarray:
    """Rescale log-odds by ``s``: ``sigmoid(s * logit(p))``; ``s == 1.0`` is exact identity."""
    if s == 1.0:
        return probs
    z = logit_array(np.asarray(probs, dtype=float), eps=_EPS)
    out = sigmoid_array(s * z)
    return float(out) if np.isscalar(probs) or np.ndim(probs) == 0 else out


#: Golden-section bracket on the logit temperature, and the bracket width it stops at.
_TEMP_BOUNDS: tuple[float, float] = (0.25, 4.0)
_TEMP_TOL = 1e-4


def fit_temperature(probs: Sequence[float], outcomes: Sequence[float]) -> float:
    """MLE the logit-temperature ``s``; returns ``1.0`` on a degenerate corpus."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0 or p.size != y.size:
        return 1.0
    if y.min() == y.max():  # one-class — no calibration signal
        return 1.0
    z = logit_array(p, eps=_EPS)
    if not np.any(np.abs(z) > _FLAT_LOGIT):  # all probs at 0.5 — flat
        return 1.0

    def nll(s: float) -> float:
        # The objective clips at the SAME _EPS as the serving logit transform above,
        # deliberately not metrics.py's 1e-15 — the optimizer must see serving's surface.
        q = np.clip(sigmoid_array(s * z), _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    # Golden-section search: maintain a bracket [lo, hi] with two interior probes.
    invphi = (np.sqrt(5.0) - 1.0) / 2.0  # 1/phi ≈ 0.618
    lo, hi = _TEMP_BOUNDS
    c = hi - invphi * (hi - lo)
    d = lo + invphi * (hi - lo)
    fc, fd = nll(c), nll(d)
    while hi - lo > _TEMP_TOL:
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - invphi * (hi - lo)
            fc = nll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + invphi * (hi - lo)
            fd = nll(d)
    return float(0.5 * (lo + hi))
