"""Tests for symmetric series-probability aggregation: f(p) + f(1-p) == 1 exactly."""

import itertools

import pytest

from league_predictor.predraft.pipeline.series_math import simulate_series, symmetric_series_prob

_PROBS = [0.02, 0.18, 0.30, 0.45, 0.50, 0.55, 0.62, 0.74, 0.88, 0.97]
_MULTI = ["Bo3", "Bo5"]


def test_swap_symmetric():
    for fmt in _MULTI:
        for p in _PROBS:
            a = symmetric_series_prob(p, fmt)
            b = symmetric_series_prob(1.0 - p, fmt)
            assert a + b == pytest.approx(1.0, abs=1e-9), (fmt, p)


def test_half_maps_to_half():
    for fmt in _MULTI:
        assert symmetric_series_prob(0.5, fmt) == pytest.approx(0.5, abs=1e-12), fmt


def test_bo1_is_identity():
    for p in _PROBS:
        # A Bo1 series IS the map; series-level damping must never touch it.
        assert symmetric_series_prob(p, "Bo1") == p
        assert symmetric_series_prob(p, None) == p


def test_monotonic_in_map_prob():
    xs = [i / 100 for i in range(1, 100)]
    for fmt in _MULTI:
        ys = [symmetric_series_prob(x, fmt) for x in xs]
        assert all(b > a for a, b in itertools.pairwise(ys)), fmt


def test_aggregation_is_orientation_symmetric():
    for fmt in _MULTI:
        for p in _PROBS:
            expected = 0.5 * (simulate_series(p, fmt) + (1.0 - simulate_series(1.0 - p, fmt)))
            assert symmetric_series_prob(p, fmt) == pytest.approx(expected, abs=1e-12), (fmt, p)


def test_naive_aggregation_amplifies_favorite():
    # Sanity: a 0.65 map favorite is a bigger series favorite (best-of effect).
    assert symmetric_series_prob(0.65, "Bo3") > 0.65
    assert symmetric_series_prob(0.65, "Bo5") > symmetric_series_prob(0.65, "Bo3")
