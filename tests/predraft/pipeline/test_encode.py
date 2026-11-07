"""Tests for the shared X -> model-ready DataFrame encoder (column order, dtypes)."""

from __future__ import annotations

import pandas as pd
import pytest

from league_predictor.predraft.pipeline.lightgbm_bag import to_frame


def test_to_frame_orders_columns_and_types_them():
    X = [{"b": "LCK", "a": 1.5}, {"a": 2.5, "b": "LPL"}]
    df = to_frame(X, feature_names=["a", "b"], categorical_names=["b"])
    assert list(df.columns) == ["a", "b"]  # feature_names order, not dict order
    assert df["a"].dtype == float  # numeric stays float
    assert isinstance(df["b"].dtype, pd.CategoricalDtype)  # declared categorical -> category
    assert df["a"].tolist() == [1.5, 2.5]
    assert df["b"].tolist() == ["LCK", "LPL"]


def test_to_frame_raises_on_missing_feature():
    """A row missing a declared feature is a bug, not a silent zero-fill."""
    with pytest.raises(KeyError):
        to_frame([{"a": 1.0}], feature_names=["a", "b"], categorical_names=[])


def test_to_frame_empty_rows_keeps_columns():
    df = to_frame([], feature_names=["a", "b"], categorical_names=["b"])
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 0


def test_to_frame_columnar_fast_path_equals_dict_path():
    """A ColumnarRows X produces the exact frame the equivalent list-of-dicts produces."""
    import numpy as np

    from league_predictor.predraft.pipeline.dataset import ColumnarRows

    matrix = np.array([[1.5, 10.0], [2.5, 20.0], [-0.5, 30.0]])
    rows = ColumnarRows(matrix, ["a", "c"])
    got = to_frame(rows, feature_names=["a", "c"], categorical_names=[])
    ref = to_frame(
        [{"a": 1.5, "c": 10}, {"a": 2.5, "c": 20}, {"a": -0.5, "c": 30}],
        feature_names=["a", "c"],
        categorical_names=[],
    )
    pd.testing.assert_frame_equal(got, ref)


def test_to_frame_columnar_missing_feature_and_categoricals_are_loud():
    """Fast path raises too: missing feature -> KeyError, declared categorical -> ValueError."""
    import numpy as np

    from league_predictor.predraft.pipeline.dataset import ColumnarRows

    rows = ColumnarRows(np.array([[1.0]]), ["a"])
    with pytest.raises(KeyError):
        to_frame(rows, feature_names=["a", "b"], categorical_names=[])
    with pytest.raises(ValueError, match="categorical"):
        to_frame(rows, feature_names=["a"], categorical_names=["a"])


def test_to_frame_empty_columnar_keeps_columns():
    import numpy as np

    from league_predictor.predraft.pipeline.dataset import ColumnarRows

    rows = ColumnarRows(np.empty((0, 2)), ["a", "b"])
    df = to_frame(rows, feature_names=["a", "b"], categorical_names=[])
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 0


def test_to_frame_consolidated_coercion_matches_per_column_reference():
    """The consolidated astype produces the same dtypes and values as per-column astype."""
    cols = ["a", "b", "c"]
    cats = ["b"]
    X = [
        {"a": 1.5, "b": "LCK", "c": 10},
        {"a": 2.5, "b": "LPL", "c": 20},
        {"a": -0.5, "b": "LCK", "c": 30},
    ]
    got = to_frame(X, feature_names=cols, categorical_names=cats)

    ref = pd.DataFrame([{n: row[n] for n in cols} for row in X], columns=cols)
    for n in cols:
        ref[n] = ref[n].astype("category") if n in cats else ref[n].astype(float)

    assert list(got.columns) == list(ref.columns)
    assert [str(d) for d in got.dtypes] == [str(d) for d in ref.dtypes]
    pd.testing.assert_frame_equal(got, ref)
