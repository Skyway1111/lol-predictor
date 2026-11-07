"""Tests for side-agnostic symmetric prediction, driven by tiny real FittedModel stand-ins."""

from __future__ import annotations

import pytest

from league_predictor.predraft.pipeline.model_contract import predict_symmetric


class _Stub:
    """Row-independent FittedModel: each row carries its own probability in ``p``."""

    def predict_proba(self, X):
        return [row["p"] for row in X]


def test_predict_symmetric_averages_both_orientations():
    """P_sym = 0.5*(P(a beats b) + (1 - P(b beats a))) per row."""
    X_ab = [{"p": 0.8}, {"p": 0.3}]
    X_ba = [{"p": 0.5}, {"p": 0.9}]
    out = predict_symmetric(_Stub(), X_ab, X_ba)
    assert out[0] == pytest.approx(0.65)  # 0.5*(0.8 + (1-0.5))
    assert out[1] == pytest.approx(0.20)  # 0.5*(0.3 + (1-0.9))


class _CallCounter:
    """FittedModel that records the row count of every predict_proba call."""

    def __init__(self) -> None:
        self.call_lengths: list[int] = []

    def predict_proba(self, X):
        rows = list(X)
        self.call_lengths.append(len(rows))
        return [row["p"] for row in rows]


def test_predict_symmetric_scores_both_orientations_in_one_call():
    """One native call (the perf win), row-for-row equal to the explicit two-call combine."""
    X_ab = [{"p": 0.8}, {"p": 0.3}, {"p": 0.55}]
    X_ba = [{"p": 0.5}, {"p": 0.9}, {"p": 0.40}]
    counter = _CallCounter()
    out = predict_symmetric(counter, X_ab, X_ba)
    assert counter.call_lengths == [len(X_ab) + len(X_ba)]
    two_call = _Stub()
    p_ab = two_call.predict_proba(X_ab)
    p_ba = two_call.predict_proba(X_ba)
    expected = [0.5 * (a + (1.0 - b)) for a, b in zip(p_ab, p_ba, strict=True)]
    assert out == expected
