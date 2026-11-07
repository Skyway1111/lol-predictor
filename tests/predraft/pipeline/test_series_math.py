"""Tests for the Markov series simulator (pipeline/series_math.py)."""

import math

import pytest

from league_predictor.probability import logit
from league_predictor.predraft.pipeline.series_math import (
    SERIES_MOMENTUM,
    simulate_series,
    symmetric_series_prob,
)

# simulate_series


def test_simulate_series_bo1():
    """Bo1 series prob equals game win prob."""
    assert simulate_series(0.6, "Bo1") == pytest.approx(0.6)
    assert simulate_series(0.0, "Bo1") == pytest.approx(0.0)
    assert simulate_series(1.0, "Bo1") == pytest.approx(1.0)


def test_simulate_series_bo3_50pct():
    """Bo3 at 50% game win prob → 50% series win prob."""
    assert simulate_series(0.5, "Bo3") == pytest.approx(0.5, abs=1e-9)


def test_simulate_series_bo3_higher_prob():
    """Better game win prob amplifies in Bo3."""
    p_bo1 = 0.6
    p_bo3 = simulate_series(p_bo1, "Bo3")
    assert p_bo3 > p_bo1  # amplification effect


def test_simulate_series_bo5_50pct():
    """Bo5 at 50% game win prob → 50% series win prob."""
    assert simulate_series(0.5, "Bo5") == pytest.approx(0.5, abs=1e-9)


def test_simulate_series_bo5_higher_prob():
    """Better game win prob amplifies more in Bo5 than Bo3."""
    p = 0.6
    p_bo3 = simulate_series(p, "Bo3")
    p_bo5 = simulate_series(p, "Bo5")
    assert p_bo5 > p_bo3  # Bo5 amplifies more


def test_simulate_series_bo5_lower_prob():
    """Worse game win prob is penalized more in Bo5."""
    p = 0.4
    p_bo3 = simulate_series(p, "Bo3")
    p_bo5 = simulate_series(p, "Bo5")
    assert p_bo5 < p_bo3  # Bo5 penalizes underdogs more


def test_simulate_series_unknown_format():
    """Unknown format falls back to Bo1 behavior."""
    assert simulate_series(0.7, "Bo7") == pytest.approx(0.7)


# game_prob_at_state


# series probability from a state


def test_series_from_state_is_decided_at_match_point_taken():
    assert simulate_series(0.6, "Bo3", state=(2, 0)) == pytest.approx(1.0)
    assert simulate_series(0.6, "Bo3", state=(0, 2)) == pytest.approx(0.0)
    assert symmetric_series_prob(0.6, "Bo5", state=(3, 1)) == pytest.approx(1.0)


def test_series_from_state_treats_p_as_the_next_maps_probability():
    """At (1,0) in a Bo3, team1 needs one map: p(next) + (1-p(next)) * p(decider)."""
    p_next = 0.6
    got = simulate_series(p_next, "Bo3", state=(1, 0))
    # The decider at (1,1) reads the pre-series odds: the next map's minus one map of lead.
    p_decider = 1 / (1 + math.exp(-(logit(p_next) - SERIES_MOMENTUM)))
    assert got == pytest.approx(p_next + (1 - p_next) * p_decider, abs=1e-12)
    assert simulate_series(0.6, "Bo5", state=(2, 0)) > simulate_series(0.6, "Bo5", state=(0, 2))
    coin = simulate_series(0.5, "Bo3", state=(1, 0), momentum=0.0)
    assert coin == pytest.approx(0.75, abs=1e-12)


def test_series_from_the_pre_series_state_is_the_plain_series_prob():
    for p in (0.3, 0.5, 0.64):
        for fmt in ("Bo3", "Bo5"):
            assert simulate_series(p, fmt, state=(0, 0)) == simulate_series(p, fmt)
            assert symmetric_series_prob(p, fmt, state=(0, 0)) == symmetric_series_prob(p, fmt)


def test_symmetric_series_prob_from_state_is_swap_symmetric():
    for state in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (0, 2)):
        for p in (0.35, 0.5, 0.8):
            mirrored = (state[1], state[0])
            total = symmetric_series_prob(p, "Bo5", state=state) + symmetric_series_prob(
                1.0 - p, "Bo5", state=mirrored
            )
            assert total == pytest.approx(1.0, abs=1e-12)
