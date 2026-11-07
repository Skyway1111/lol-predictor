"""Behavioural tests for the LightGBM seed-bag backend, the served pre-draft base model."""

from __future__ import annotations

import math
import random

import numpy as np

from sklearn.metrics import log_loss
from league_predictor.predraft.pipeline.lightgbm_bag import (
    LightGBMBagBackend,
    LightGBMParams,
    to_frame,
)
from league_predictor.predraft.pipeline.model_contract import predict_symmetric

_NAMES = ["diff", "region"]
_CATS = ["region"]
_REGION_BUMP = {"LCK": 0.6, "LPL": 0.0, "LEC": -0.6}


def _synth(n: int, *, seed: int = 0):
    """Label depends on both a numeric diff and a categorical region bump."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        x = rng.uniform(-3.0, 3.0)
        region = rng.choice(list(_REGION_BUMP))
        p = 1.0 / (1.0 + math.exp(-(x + _REGION_BUMP[region])))
        X.append({"diff": x, "region": region})
        y.append(1 if rng.random() < p else 0)
    return X, y


def test_bag_learns_monotone_signal_and_outputs_probabilities():
    X, y = _synth(500)
    fitted = LightGBMBagBackend(seeds=(42, 123, 7)).fit(
        X, y, sample_weight=None, feature_names=_NAMES, categorical_names=_CATS
    )
    probs = fitted.predict_proba([{"diff": 3.0, "region": "LCK"}, {"diff": -3.0, "region": "LEC"}])
    assert all(0.0 < p < 1.0 for p in probs)
    assert probs[0] > probs[1]  # strong matchup scored above the weak one


def _xor(n: int, *, seed: int):
    """An XOR signal (label = 1 iff the features disagree in sign): needs depth >= 2 to model."""
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        a = rng.uniform(-1.0, 1.0)
        b = rng.uniform(-1.0, 1.0)
        X.append({"a": a, "b": b})
        y.append(1 if (a > 0) != (b > 0) else 0)
    return X, y


def test_bag_honours_max_depth_param():
    """``max_depth`` reaches LightGBM: a depth-3 bag scores XOR far better than a depth-1 bag."""
    X, y = _xor(1500, seed=3)
    Xte, yte = _xor(600, seed=99)

    def _ll_at_depth(depth: int) -> float:
        fitted = LightGBMBagBackend(
            LightGBMParams(max_depth=depth, n_estimators=200), seeds=(1, 2, 3)
        ).fit(X, y, sample_weight=None, feature_names=["a", "b"], categorical_names=[])
        return log_loss(yte, fitted.predict_proba(Xte))

    assert _ll_at_depth(3) < _ll_at_depth(1) - 0.05


def test_bag_combines_seeds_in_logit_space():
    """The bag averages per-seed log-odds: it equals sigmoid(mean(logit(member))) exactly."""
    X, y = _synth(400, seed=5)
    Xte = [{"diff": 1.3, "region": "LPL"}, {"diff": -0.7, "region": "LCK"}]

    def _single_seed(s: int) -> list[float]:
        return (
            LightGBMBagBackend(seeds=(s,))
            .fit(X, y, sample_weight=None, feature_names=_NAMES, categorical_names=_CATS)
            .predict_proba(Xte)
        )

    bag = (
        LightGBMBagBackend(seeds=(7, 999))
        .fit(X, y, sample_weight=None, feature_names=_NAMES, categorical_names=_CATS)
        .predict_proba(Xte)
    )
    p7, p999 = _single_seed(7), _single_seed(999)
    for got, a, b in zip(bag, p7, p999, strict=True):
        mean_logit = (math.log(a / (1 - a)) + math.log(b / (1 - b))) / 2
        assert abs(got - 1.0 / (1.0 + math.exp(-mean_logit))) < 1e-9


def test_bag_predict_is_float_identical_to_vectorised_reference():
    """The vectorised bag predict is bit-for-bit the explicit per-member numpy reference."""
    X, y = _synth(400, seed=11)
    Xte = [
        {"diff": 1.3, "region": "LPL"},
        {"diff": -0.7, "region": "LCK"},
        {"diff": 0.0, "region": "LEC"},
    ]
    fitted = LightGBMBagBackend(seeds=(7, 99, 123)).fit(
        X, y, sample_weight=None, feature_names=_NAMES, categorical_names=_CATS
    )
    frame = to_frame(Xte, feature_names=_NAMES, categorical_names=_CATS)
    per = np.clip(
        np.vstack([m.predict_proba_frame(frame) for m in fitted._members]), 1e-15, 1 - 1e-15
    )
    avg_logit = np.log(per / (1 - per)).mean(axis=0)
    expected = [float(p) for p in 1.0 / (1.0 + np.exp(-avg_logit))]
    assert fitted.predict_proba(Xte) == expected  # exact equality, not approx


def test_predict_symmetric_one_call_equals_two_call_on_fitted_bag():
    """One concatenated ``predict_symmetric`` call is float-identical to two separate calls."""
    X, y = _synth(400, seed=13)
    X_ab = [{"diff": 1.3, "region": "LPL"}, {"diff": -0.7, "region": "LCK"}]
    X_ba = [{"diff": -1.3, "region": "LCK"}, {"diff": 0.7, "region": "LPL"}]
    fitted = LightGBMBagBackend(seeds=(7, 999)).fit(
        X, y, sample_weight=None, feature_names=_NAMES, categorical_names=_CATS
    )
    p_ab = fitted.predict_proba(X_ab)
    p_ba = fitted.predict_proba(X_ba)
    two_call = [0.5 * (a + (1.0 - b)) for a, b in zip(p_ab, p_ba, strict=True)]
    assert predict_symmetric(fitted, X_ab, X_ba) == two_call  # exact equality
