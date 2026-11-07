"""Tests for draft.grader.grades — thresholds, grade assignment, the edge curve, bins."""

from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from league_predictor.draft_grading.grader.grades import (
    _KNOT_GRID,
    EDGE_CURVE_KNOTS,
    EDGE_CUT_A_PP,
    EDGE_CUT_S_PP,
    EdgeCurve,
    assign_grade,
    fit_edge_curve,
)


class TestAssignGrade:
    def test_extreme_scores(self):
        thresholds = (0.35, 0.45, 0.55, 0.65)
        assert assign_grade(0.10, thresholds) == "D"
        assert assign_grade(0.90, thresholds) == "S"

    def test_boundary(self):
        thresholds = (0.35, 0.45, 0.55, 0.65)
        assert assign_grade(0.35, thresholds) == "D"
        assert assign_grade(0.351, thresholds) == "C"


def _identity_curve() -> EdgeCurve:
    """The identity calibration g(s) = s — edge(s) = s - 0.5, hand-checkable."""
    return EdgeCurve(np.linspace(0.0, 1.0, EDGE_CURVE_KNOTS))


class TestEdgeCurveValidation:
    def test_rejects_wrong_length(self):
        with pytest.raises(ValueError, match="knot"):
            EdgeCurve(np.linspace(0.0, 1.0, 50))

    def test_rejects_non_monotone(self):
        wr = np.linspace(0.0, 1.0, EDGE_CURVE_KNOTS)
        wr[100], wr[101] = wr[101] + 0.05, wr[100]
        with pytest.raises(ValueError, match="monotone"):
            EdgeCurve(wr)

    def test_rejects_asymmetric(self):
        wr = np.clip(np.linspace(0.0, 1.0, EDGE_CURVE_KNOTS) + 0.1, 0.0, 1.0)
        with pytest.raises(ValueError, match="antisymmetric"):
            EdgeCurve(wr)


class TestEdgeCurveEvaluation:
    def test_identity_curve_edge_and_win_rate(self):
        curve = _identity_curve()
        assert curve.edge_pp(0.75) == pytest.approx(25.0)
        assert curve.edge_pp(0.5) == 0.0
        assert curve.win_rate(0.62) == pytest.approx(0.62)
        # win_rate and edge_pp read the same knot edges; edge_pp's percentage round-trip
        # may differ by an ulp.
        for s in (0.13, 0.4441, 0.5, 0.777):
            assert curve.win_rate(s) == pytest.approx(0.5 + curve.edge_pp(s) / 100.0, abs=1e-15)

    def test_win_rate_is_vectorised_and_exactly_antisymmetric(self):
        curve = _identity_curve()
        scores = np.array([0.13, 0.4441, 0.5, 0.777, 1.0, 0.0])
        rates = curve.win_rate(scores)
        assert isinstance(rates, np.ndarray)
        assert rates.tolist() == [curve.win_rate(float(s)) for s in scores]
        high = scores >= 0.5
        assert np.array_equal(curve.win_rate(1.0 - scores[high]), 1.0 - rates[high])
        assert rates[4] == curve.win_rates[-1]
        assert rates[5] == curve.win_rates[0]

    def test_edge_pp_negation_symmetry_is_exact(self):
        """edge_pp(1-p) == -edge_pp(p) bitwise, so red can display the stored blue edge negated."""
        rng = np.random.default_rng(3)
        curve = fit_edge_curve(rng.uniform(0.2, 0.8, 400), rng.integers(0, 2, 400))
        for p in (0.5, 0.123456789, 0.4999999, 0.500001, 0.9, 1e-9, *rng.uniform(0, 1, 50)):
            assert curve.edge_pp(1.0 - p) == -curve.edge_pp(p)

    def test_out_of_range_scores_clamp_to_endpoints(self):
        curve = _identity_curve()
        assert curve.edge_pp(1.5) == curve.edge_pp(1.0)
        assert curve.edge_pp(-0.5) == curve.edge_pp(0.0)


class TestEdgeThresholds:
    def test_identity_curve_preimages(self):
        """On g(s) = s, the preimage of +X pp is the smallest knot ≥ 0.5 + X/100."""
        # Cuts sit off the knot grid (X/100 not a multiple of 0.005): a cut landing exactly on
        # a knot may tie either way.
        d, c, b, a = _identity_curve().thresholds(s_cut_pp=10.2, a_cut_pp=4.2)
        assert a == pytest.approx(0.605)
        assert b == pytest.approx(0.545)
        assert d + a == pytest.approx(1.0, abs=1e-9)
        assert c + b == pytest.approx(1.0, abs=1e-9)
        assert d <= c <= b <= a
        # The cut set drives assign_grade unchanged.
        assert assign_grade(0.65, (d, c, b, a)) == "S"
        assert assign_grade(0.57, (d, c, b, a)) == "A"
        assert assign_grade(0.50, (d, c, b, a)) == "B"
        assert assign_grade(0.43, (d, c, b, a)) == "C"
        assert assign_grade(0.35, (d, c, b, a)) == "D"

    def test_saturation_when_promise_unreachable(self, caplog):
        """A curve too flat to reach X_S saturates the S cut to 1.0 — S and D empty, logged."""
        flat = EdgeCurve(np.full(EDGE_CURVE_KNOTS, 0.5))
        with caplog.at_level(logging.WARNING):
            d, c, b, a = flat.thresholds(s_cut_pp=11.0, a_cut_pp=4.0)
        assert (d, c, b, a) == (0.0, 0.0, 1.0, 1.0)
        assert "unreachable" in caplog.text
        assert assign_grade(0.9, (d, c, b, a)) == "B"  # everything grades B

    def test_rejects_misordered_cuts(self):
        with pytest.raises(ValueError, match="cut"):
            _identity_curve().thresholds(s_cut_pp=4.0, a_cut_pp=10.0)

    def test_frozen_cuts_are_ordered_positive(self):
        """The pre-registered semantic constants: S promises more than A, both > 0."""
        assert EDGE_CUT_S_PP > EDGE_CUT_A_PP > 0.0

    def test_frozen_cut_values(self):
        """The exact pre-registered constants (operator-set 2026-07-09) must not drift."""
        assert EDGE_CUT_S_PP == 8.0
        assert EDGE_CUT_A_PP == 4.0


class TestFitEdgeCurve:
    def test_fit_is_monotone_antisymmetric_and_tracks_calibration(self):
        """y ~ Bernoulli(s): the fit is monotone, antisymmetric about (0.5, 0.5), near-diagonal."""
        rng = np.random.default_rng(11)
        s = rng.uniform(0.05, 0.95, 20_000)
        y = (rng.random(20_000) < s).astype(float)
        curve = fit_edge_curve(s, y)
        wr = curve.win_rates
        assert np.all(np.diff(wr) >= 0.0)
        assert np.max(np.abs(wr + wr[::-1] - 1.0)) < 1e-9
        assert curve.win_rate(0.5) == pytest.approx(0.5, abs=1e-12)
        assert curve.win_rate(0.75) == pytest.approx(0.75, abs=0.03)

    def test_reflection_makes_a_skewed_pool_symmetric(self):
        """A blue-skewed pool still yields an antisymmetric curve; the fit reflects (1-s, 1-y)."""
        rng = np.random.default_rng(5)
        s = rng.uniform(0.5, 0.95, 5_000)  # blue-favored scores only
        y = (rng.random(5_000) < s).astype(float)
        wr = fit_edge_curve(s, y).win_rates
        assert np.max(np.abs(wr + wr[::-1] - 1.0)) < 1e-9

    def test_empty_pool_yields_flat_curve_loudly(self, caplog):
        with caplog.at_level(logging.WARNING):
            curve = fit_edge_curve(np.array([]), np.array([]))
        assert np.array_equal(curve.win_rates, np.full(EDGE_CURVE_KNOTS, 0.5))
        assert "EMPTY" in caplog.text

    def test_meta_roundtrip_is_exact(self):
        """to_meta → json → from_meta reproduces the curve bitwise (float repr round-trips)."""
        rng = np.random.default_rng(7)
        curve = fit_edge_curve(rng.uniform(0.2, 0.8, 500), rng.integers(0, 2, 500))
        meta = json.loads(json.dumps(curve.to_meta()))
        again = EdgeCurve.from_meta(meta)
        assert np.array_equal(again.win_rates, curve.win_rates)
        assert set(meta) == {"scores", "win_rates"}
        assert len(meta["scores"]) == EDGE_CURVE_KNOTS

    def test_from_meta_rejects_foreign_knot_grid(self):
        meta = {
            "scores": list(np.linspace(0.1, 0.9, EDGE_CURVE_KNOTS)),
            "win_rates": list(np.linspace(0.0, 1.0, EDGE_CURVE_KNOTS)),
        }
        with pytest.raises(ValueError, match="knot grid"):
            EdgeCurve.from_meta(meta)

    def test_centering_leaves_no_interior_plateaus(self):
        """The centered fit is strictly increasing across the data's score range."""
        # Plateaus make edge-cut preimages cliff-prone: a cut on one empties or floods a tier.
        rng = np.random.default_rng(3)
        # Coarse scores (heavy ties) force wide PAVA blocks in a raw isotonic fit.
        s = np.round(rng.uniform(0.2, 0.8, 4_000), 1)
        y = (rng.random(4_000) < s).astype(float)
        wr = fit_edge_curve(s, y).win_rates
        interior = (np.asarray(_KNOT_GRID) > 0.25) & (np.asarray(_KNOT_GRID) < 0.75)
        assert np.all(np.diff(wr[interior]) > 0.0)

    def test_tail_clip_bounds_thin_support_promises(self):
        """A lone extreme side cannot promise a +-50pp edge: clipped to MIN_TAIL_SIDES range."""
        rng = np.random.default_rng(9)
        s = np.concatenate([rng.uniform(0.35, 0.65, 2_000), [0.999]])
        y = np.concatenate([(rng.random(2_000) < 0.5).astype(float), [1.0]])
        curve = fit_edge_curve(s, y)
        assert curve.edge_pp(0.999) < 45.0  # not the saturated +49.9pp
        assert curve.edge_pp(0.999) == -curve.edge_pp(0.001)

    def test_small_pool_is_not_clipped(self):
        """Pools under MIN_TAIL_SIDES (fixtures, degenerate fits) skip the clip."""
        s = np.array([0.1, 0.9] * 10)
        y = np.array([0.0, 1.0] * 10)
        curve = fit_edge_curve(s, y)
        assert curve.win_rate(0.9) > 0.9  # unclipped extreme survives
