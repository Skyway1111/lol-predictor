"""Chronological calibration windows and fitted grade calibration."""

from dataclasses import dataclass

import numpy as np

from league_predictor.draft_grading.grader.grades import EdgeCurve


def calibration_tail_start(n_total: int) -> int:
    """First row index of the held-out calibration tail (walk-forward 80/20)."""
    return n_total - n_total // 5


def fold_windows(tail_start: int, n_rows: int, oof_folds: int) -> list[tuple[int, int]]:
    """Expanding-fold ``[start, end)`` windows partitioning ``[tail_start, n_rows)``."""
    edges = np.linspace(tail_start, n_rows, oof_folds + 1, dtype=int)
    return [
        (int(edges[f]), int(edges[f + 1]))
        for f in range(oof_folds)
        if int(edges[f]) > 0 and int(edges[f + 1]) > int(edges[f])
    ]


@dataclass(frozen=True)
class OofCalibration:
    """Calibration-tail OOF outputs: the selected epoch count and serve-time calibration."""

    chosen_rounds: int
    temperature: float
    thresholds: list[float]
    n_folds: int
    n_pool: int
    edge_curve: EdgeCurve
    #: The UNCALIBRATED symmetrized pool, row-aligned to the chronological calibration
    #: tail; absent when the caller does not request the pool.
    pool_probs: np.ndarray | None = None
