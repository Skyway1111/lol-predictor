"""Infer played series format from realized maps."""

import pytest

from league_predictor.ingestion.facts_transform import infer_series_format


@pytest.mark.parametrize(
    ("numbers", "labels", "expected"),
    [
        ([1], [1], "Bo1"),
        ([1, 2], [1, 1], "Bo3"),
        ([1, 2, 3], [1, 0, 1], "Bo3"),
        ([1, 2, 3], [1, 1, 1], "Bo5"),
        ([1, 2, 3, 4, 5], [1, 0, 1, 0, 1], "Bo5"),
        ([3, 4], [1, 0], "Bo5"),
    ],
)
def test_realized_format(numbers, labels, expected):
    assert infer_series_format(numbers, labels) == expected
