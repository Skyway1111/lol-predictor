"""Fit leak-free expected-edge curves and assign frozen Draft grade bands."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)

Grade = Literal["S", "A", "B", "C", "D"]

# Frozen edge cuts, in percentage points of calibrated expected edge: S promises edge
# >= S_PP, A >= A_PP, C/D mirror them below 0.5. Semantics, not tuning knobs.
EDGE_CUT_S_PP = 8.0
EDGE_CUT_A_PP = 4.0

# Tail-support floor: the curve never promises an edge backed by fewer than this many
# fitted sides. Pools smaller than the floor are left unclipped.
MIN_TAIL_SIDES = 50

#: How far the curve and its grade cuts may drift from exact antisymmetry about 0.5.
_ANTISYMMETRY_TOL = 1e-9

#: The canonical knot grid every :class:`EdgeCurve` is defined on.
EDGE_CURVE_KNOTS = 201
_KNOT_GRID = np.linspace(0.0, 1.0, EDGE_CURVE_KNOTS)
_KNOT_GRID.setflags(write=False)


class EdgeCurve:
    """A frozen isotonic score -> win-rate curve on the canonical knot grid."""

    def __init__(self, win_rates) -> None:
        wr = np.ascontiguousarray(np.asarray(win_rates, dtype=np.float64))
        if wr.shape != (EDGE_CURVE_KNOTS,):
            raise ValueError(
                f"edge curve needs exactly {EDGE_CURVE_KNOTS} knot win rates, got {wr.shape}"
            )
        if np.any(np.diff(wr) < 0.0):
            raise ValueError("edge curve win rates are not monotone non-decreasing")
        if float(np.max(np.abs(wr + wr[::-1] - 1.0))) > _ANTISYMMETRY_TOL:
            raise ValueError("edge curve is not antisymmetric about (0.5, 0.5) to 1e-9")
        wr.setflags(write=False)
        self.win_rates = wr
        # The exactly-antisymmetric edge array the evaluators interpolate:
        # (a - b)/2 == -((b - a)/2) bitwise, so edge[i] == -edge[n-1-i] exactly.
        edge = (wr - wr[::-1]) / 2.0
        edge.setflags(write=False)
        self._edge = edge

    def edge_pp(self, score: float) -> float:
        """Calibrated expected edge at ``score``, in win-rate pp; antisymmetric about 0.5."""
        s = float(score)
        if s >= 0.5:  # noqa: PLR2004 - 0.5 is the curve's antisymmetry pivot
            return float(np.interp(s, _KNOT_GRID, self._edge)) * 100.0
        return -float(np.interp(1.0 - s, _KNOT_GRID, self._edge)) * 100.0

    def win_rate(self, scores: np.ndarray | float) -> np.ndarray | float:
        """Calibrated win rate per score, scalar or ``(N,)``; a mirror reads exactly ``1 -`` it."""
        s = np.asarray(scores, dtype=np.float64)
        favoured = 0.5 + np.interp(np.maximum(s, 1.0 - s), _KNOT_GRID, self._edge)
        out = np.where(s >= 0.5, favoured, 1.0 - favoured)  # noqa: PLR2004 - the pivot
        return float(out) if s.ndim == 0 else out

    @property
    def max_edge(self) -> tuple[float, float]:
        """The ceiling ``(score, edge_pp)``: the LOWEST knot on the largest promise's plateau."""
        i = int(np.argmax(self._edge))
        return float(_KNOT_GRID[i]), float(self._edge[i]) * 100.0

    def thresholds(
        self,
        s_cut_pp: float = EDGE_CUT_S_PP,
        a_cut_pp: float = EDGE_CUT_A_PP,
    ) -> tuple[float, float, float, float]:
        """The ``(d_cut, c_cut, b_cut, a_cut)`` score-space preimages of the edge cuts."""
        if not s_cut_pp >= a_cut_pp > 0.0:
            raise ValueError(
                f"edge cuts must satisfy s_cut_pp >= a_cut_pp > 0, got ({s_cut_pp}, {a_cut_pp})"
            )
        a_cut, d_cut = self._preimage_pair(s_cut_pp, tier="S/D")
        b_cut, c_cut = self._preimage_pair(a_cut_pp, tier="A/C")
        assert abs(d_cut + a_cut - 1.0) <= _ANTISYMMETRY_TOL
        assert abs(c_cut + b_cut - 1.0) <= _ANTISYMMETRY_TOL
        assert d_cut <= c_cut <= b_cut <= a_cut
        return (d_cut, c_cut, b_cut, a_cut)

    def _preimage_pair(self, cut_pp: float, *, tier: str) -> tuple[float, float]:
        """(upper, lower): the smallest knot with edge >= ``cut_pp``/100 and its mirror."""
        hits = np.nonzero(self._edge >= cut_pp / 100.0)[0]
        if hits.size == 0:
            logger.warning(
                "edge curve never reaches +%.1fpp - the %s tiers are EMPTY by "
                "construction (promise unreachable for this fit)",
                cut_pp,
                tier,
            )
            return 1.0, 0.0
        i = int(hits[0])
        return float(_KNOT_GRID[i]), float(_KNOT_GRID[EDGE_CURVE_KNOTS - 1 - i])

    def to_meta(self) -> dict[str, list[float]]:
        """The JSON-ready artifact form: knot scores + win rates (floats round-trip exactly)."""
        return {
            "scores": [float(s) for s in _KNOT_GRID],
            "win_rates": [float(w) for w in self.win_rates],
        }

    @classmethod
    def from_meta(cls, meta: Mapping) -> EdgeCurve:
        """Rebuild from artifact meta; refuses a curve on a non-canonical knot grid."""
        scores = np.asarray(meta["scores"], dtype=np.float64)
        if scores.shape != (EDGE_CURVE_KNOTS,) or not np.allclose(
            scores, _KNOT_GRID, rtol=0.0, atol=1e-12
        ):
            raise ValueError(
                "edge curve meta carries a non-canonical knot grid - the curve is only "
                f"defined on {EDGE_CURVE_KNOTS} evenly spaced knots in [0, 1]"
            )
        return cls(meta["win_rates"])


def fit_edge_curve(scores: np.ndarray, outcomes: np.ndarray) -> EdgeCurve:
    """Fit the antisymmetrized centered isotonic win-rate curve on an OOF-scored pool."""
    from sklearn.isotonic import IsotonicRegression

    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if s.size == 0:
        logger.warning(
            "edge curve fit on an EMPTY pool - serving a flat 0.5 curve (every edge 0, "
            "every side grades B)"
        )
        return EdgeCurve(np.full(EDGE_CURVE_KNOTS, 0.5))
    x_pool = np.concatenate([s, 1.0 - s])
    y_pool = np.concatenate([y, 1.0 - y])
    iso = IsotonicRegression(out_of_bounds="clip").fit(x_pool, y_pool)
    fitted = np.asarray(iso.predict(x_pool), dtype=np.float64)

    # Tail-support clip: the largest promise backed by >= MIN_TAIL_SIDES fitted sides
    # (the pool is reflected, so the low tail is its exact mirror).
    if fitted.size >= MIN_TAIL_SIDES:
        cap = float(np.sort(fitted)[-MIN_TAIL_SIDES])
        fitted = np.clip(fitted, 1.0 - cap, cap)

    # Centering: one node per constant block at (mean support score, block value),
    # linearly interpolated -> strictly increasing across distinct blocks.
    order = np.argsort(x_pool, kind="stable")
    xs, fs = x_pool[order], fitted[order]
    node_x, node_y = [], []
    i = 0
    while i < len(fs):
        j = i
        while j < len(fs) and fs[j] == fs[i]:
            j += 1
        node_x.append(float(xs[i:j].mean()))
        node_y.append(float(fs[i]))
        i = j
    f = np.interp(_KNOT_GRID, node_x, node_y)

    g = 0.5 + (f - f[::-1]) / 2.0
    g = np.maximum.accumulate(np.clip(g, 0.0, 1.0))
    return EdgeCurve(g)


def assign_grade(score: float, thresholds: tuple[float, float, float, float]) -> Grade:
    """Bucket a continuous score into a grade letter at the artifact's four frozen cut-points."""
    d_cut, c_cut, b_cut, a_cut = thresholds
    if score <= d_cut:
        return "D"
    if score <= c_cut:
        return "C"
    if score <= b_cut:
        return "B"
    if score <= a_cut:
        return "A"
    return "S"
