"""Tests for the fixed Served recipe's pure temperature transform."""

from __future__ import annotations

import itertools
import math

import pytest

from league_predictor.predraft.pipeline.temperature import apply_temperature


def test_apply_is_bit_exact_identity_at_one():
    """s == 1.0 returns the input prob byte-for-byte, not a logit round-trip."""
    for p in (0.0, 1e-12, 0.5, 0.7333333333333333, 1.0 - 1e-9, 1.0):
        assert apply_temperature(p, 1.0) is p or apply_temperature(p, 1.0) == p


def test_apply_is_antisymmetric():
    """apply(1 - p, s) == 1 - apply(p, s): the serve path is side-agnostic."""
    s = 1.5
    for p in (0.05, 0.3, 0.5, 0.62, 0.91):
        assert apply_temperature(1.0 - p, s) == pytest.approx(
            1.0 - apply_temperature(p, s), abs=1e-15
        )


def test_apply_is_monotone_in_p():
    """Increasing p yields an increasing calibrated prob (s > 0)."""
    for s in (0.5, 1.5, 3.0):
        probs = [apply_temperature(p, s) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert probs == sorted(probs)
        assert all(a < b for a, b in itertools.pairwise(probs))


def test_apply_sharpens_when_temperature_exceeds_one():
    """s > 1 pushes a confident prob further from 0.5 (the under-confidence fix)."""
    assert apply_temperature(0.7, 1.5) > 0.7
    assert apply_temperature(0.3, 1.5) < 0.3
    # and the transform is exactly sigmoid(s * logit(p))
    assert apply_temperature(0.7, 1.5) == pytest.approx(
        1.0 / (1.0 + math.exp(-1.5 * math.log(0.7 / 0.3))), abs=1e-15
    )
