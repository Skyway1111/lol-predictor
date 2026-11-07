"""The served model's exact rating-artifact dependency contract."""

from dataclasses import replace

import pytest

from league_predictor.predraft.ratings.skill_filter.artifact import (
    RatingDependency,
    require_rating_dependency,
)


def _dependency() -> RatingDependency:
    return RatingDependency(
        artifact_sha="a" * 64,
        member_count=7,
        ordered_member_digest="b" * 64,
    )


def test_rating_dependency_round_trips_and_has_stable_identity():
    dependency = _dependency()
    assert RatingDependency.from_meta(dependency.to_meta()) == dependency
    assert RatingDependency.from_meta(dependency.to_meta()).identity == dependency.identity


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_sha", "c" * 64),
        ("member_count", 6),
        ("ordered_member_digest", "d" * 64),
        ("stage_version", "rating-order-v0"),
    ],
)
def test_rating_dependency_mismatch_names_retrain_remedy(field, value):
    expected = _dependency()
    actual = replace(expected, **{field: value})

    with pytest.raises(RuntimeError, match="retrain served"):
        require_rating_dependency(actual, expected)


def test_rating_dependency_accepts_exact_match():
    require_rating_dependency(_dependency(), _dependency())
