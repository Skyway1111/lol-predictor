"""Tests for draft_grading/grader/calibration.py — the bag logit-temperature scalar."""

from __future__ import annotations

import numpy as np

from league_predictor.draft_grading.grader.calibration import apply_temperature, fit_temperature


def _logloss(probs: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _corpus(scale: float, seed: int, n: int = 20000):
    """Outcomes from a true logit reported as ``sigmoid(scale * true_logit)``."""
    rng = np.random.default_rng(seed)
    true_logit = rng.normal(0.0, 2.0, n)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-true_logit))).astype(float)
    probs = 1.0 / (1.0 + np.exp(-scale * true_logit))
    return probs, y


def test_under_confident_gives_s_above_one():
    """A model whose probs sit too close to 0.5 needs sharpening (s>1)."""
    probs, y = _corpus(scale=0.5, seed=0)
    s = fit_temperature(probs, y)  # pyright: ignore[reportArgumentType]  # ndarray duck-types Sequence[float]
    assert s > 1.05, s


def test_over_confident_gives_s_below_one():
    """A model whose probs are too extreme needs softening (s<1)."""
    probs, y = _corpus(scale=2.0, seed=1)
    s = fit_temperature(probs, y)  # pyright: ignore[reportArgumentType]  # ndarray duck-types Sequence[float]
    assert s < 0.95, s


def test_fit_recovers_the_inverse_scale():
    """The optimum for ``sigmoid(scale*z)`` is ``s ~= 1/scale``."""
    probs, y = _corpus(scale=0.5, seed=2)
    s = fit_temperature(probs, y)  # pyright: ignore[reportArgumentType]  # ndarray duck-types Sequence[float]
    assert abs(s - 2.0) < 0.25, s


def test_temperature_never_worsens_logloss():
    probs, y = _corpus(scale=0.5, seed=3)
    s = fit_temperature(probs, y)  # pyright: ignore[reportArgumentType]  # ndarray duck-types Sequence[float]
    assert _logloss(apply_temperature(probs, s), y) <= _logloss(probs, y) + 1e-9  # pyright: ignore[reportArgumentType]  # apply_temperature(array) returns array


def test_apply_identity_at_one_is_exact():
    probs = np.array([0.1, 0.5, 0.73, 0.99])
    out = apply_temperature(probs, 1.0)
    assert out is probs or np.array_equal(out, probs)


def test_apply_temperature_vectorized_matches_scalar():
    probs = np.array([0.2, 0.4, 0.8])
    s = 1.6
    vec = apply_temperature(probs, s)
    scal = np.array([apply_temperature(float(p), s) for p in probs])
    np.testing.assert_allclose(vec, scal, atol=1e-12)


def test_degenerate_corpora_return_identity():
    # empty
    assert fit_temperature([], []) == 1.0
    # one-class outcomes
    assert fit_temperature([0.3, 0.6, 0.4], [1, 1, 1]) == 1.0
    # mismatched lengths
    assert fit_temperature([0.3, 0.6], [1]) == 1.0
    # all probs at 0.5 (flat — nothing to scale)
    assert fit_temperature([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 1.0
