"""Dependency-leaf proper scoring rules — the log-loss owner, one home per computational form."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: The prob-space clip the draft/impact evaluators froze; the Sequence form keeps 1e-12.
_EPS = 1e-15

#: The two-class argmax cut: accuracy credits whichever side the probability leans to.
_ARGMAX_CUT = 0.5


def _n(probs: Sequence[float]) -> int:
    """Length; raises a named error on empty input, where every metric is undefined."""
    n = len(probs)
    if n == 0:
        raise ValueError("metric is undefined on empty input")
    return n


def logloss_logits(labels: np.ndarray, eta: np.ndarray) -> float:
    """Mean logistic NLL at logits ``eta`` — the saturation-free log-sum-exp form."""
    y = labels
    return float(np.mean(-(y * -np.logaddexp(0.0, -eta) + (1.0 - y) * -np.logaddexp(0.0, eta))))


def accuracy(probs: Sequence[float], labels: Sequence[int]) -> float:
    """Argmax accuracy at the 0.5 threshold. Reported, never the selector."""
    n = _n(probs)
    return sum((p >= _ARGMAX_CUT) == (y == 1) for p, y in zip(probs, labels, strict=True)) / n
