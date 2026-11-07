"""Tests for the walk-forward dataset stage: recency weights, corpus, ColumnarRows."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import pytest

from league_predictor.predraft.features.families.recency import RecencyFamily
from league_predictor.predraft.pipeline.dataset import (
    ColumnarRows,
    recency_weights,
    rows_in_window,
)
from league_predictor.predraft.pipeline.walk_forward import FeaturizedCorpus

# ColumnarRows -- the lazy dict-row facade over the corpus feature matrix


def _rows3():
    return ColumnarRows(np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]), ["a", "b"])


def test_columnar_rows_len_and_single_row_materializes_a_plain_dict():
    rows = _rows3()
    assert len(rows) == 3
    assert rows[1] == {"a": 3.0, "b": 4.0}
    assert type(rows[1]) is dict  # a REAL dict, safe for pandas/consumers
    assert list(rows) == [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}, {"a": 5.0, "b": 6.0}]


def test_columnar_rows_slice_is_a_view_not_a_copy():
    """window() slicing must not duplicate the corpus matrix."""
    rows = _rows3()
    sliced = rows[1:3]
    assert isinstance(sliced, ColumnarRows)
    assert len(sliced) == 2
    assert sliced[0] == {"a": 3.0, "b": 4.0}
    assert sliced.matrix.base is rows.matrix  # numpy view, shared memory
    assert sliced.names == rows.names


def test_columnar_rows_empty_is_falsy_and_nonempty_truthy():
    """``fit_window`` gates on ``if not ds.X`` -- a raw ndarray would raise here."""
    rows = _rows3()
    assert rows
    assert not rows[0:0]


def test_recency_weights_decay_then_floor():
    """Schedule is max(exp(-days_back/tau), floor); negative age clamps to 1.0."""
    ref = date(2026, 1, 1)
    dates = [date(2026, 1, 1), date(2025, 12, 2), date(2025, 1, 1), date(2026, 2, 1)]
    w = recency_weights(dates, reference=ref, tau=30.0, floor=0.05)
    assert w[0] == pytest.approx(1.0)  # 0 days back -> exp(0)
    assert w[1] == pytest.approx(math.exp(-1.0))  # 30 days back -> exp(-1)
    assert w[2] == pytest.approx(0.05)  # 365 days back -> floored
    assert w[3] == pytest.approx(1.0)  # future row -> clamped to 1.0


def test_recency_weights_floor_zero_is_a_hard_cutoff_shape():
    """floor=0 decays old rows toward zero; a row one tau back still carries exp(-1)."""
    ref = date(2026, 1, 1)
    w = recency_weights([date(2025, 1, 1), date(2025, 12, 2)], reference=ref, tau=30.0, floor=0.0)
    assert w[0] == pytest.approx(math.exp(-365.0 / 30.0))
    assert w[1] == pytest.approx(math.exp(-1.0))


def test_corpus_augments_both_orientations_with_flipped_labels(skill_db):
    """Each eligible map becomes two rows (a-vs-b, b-vs-a): labels flip, diffs negate."""
    conn = skill_db(start=date(2026, 1, 1), end=date(2026, 4, 15))
    n_rows = len(rows_in_window(conn, date.min, date(2026, 4, 15)))
    fam = RecencyFamily()
    index = fam.build(conn, through=date(2026, 4, 15), query_dates=[])
    ds = FeaturizedCorpus(
        conn,
        start=date.min,
        end=date(2026, 4, 15),
        families=[fam],
        indexes={"recency": index},
        drop_features=frozenset(),
    ).window(date.min, date(2026, 4, 15))

    assert ds.feature_names == [
        "recency_winrate20_diff",
        "recency_ema_diff",
        "recency_momentum_diff",
    ]
    assert ds.categorical_names == []
    assert len(ds.X) == len(ds.y) == 2 * n_rows  # both orientations
    # rows come in (a,b),(b,a) pairs: labels flip, each diff negates.
    assert ds.y[0] + ds.y[1] == 1
    assert ds.X[0]["recency_winrate20_diff"] == pytest.approx(-ds.X[1]["recency_winrate20_diff"])
    assert ds.meta[0]["date"] == ds.meta[1]["date"]  # same map, two orientations


def test_corpus_window_excludes_rows_outside_bounds(skill_db):
    """[start, end) bounds row selection -- the window cap is just a tighter start."""
    conn = skill_db(start=date(2026, 1, 1), end=date(2026, 4, 15))
    fam = RecencyFamily()
    index = fam.build(conn, through=date(2026, 4, 15), query_dates=[])
    ds = FeaturizedCorpus(
        conn,
        start=date(2026, 3, 1),
        end=date(2026, 4, 15),
        families=[fam],
        indexes={"recency": index},
        drop_features=frozenset(),
    ).window(date(2026, 3, 1), date(2026, 4, 15))
    assert ds.X  # non-empty
    assert all(m["date"] >= date(2026, 3, 1) for m in ds.meta)
