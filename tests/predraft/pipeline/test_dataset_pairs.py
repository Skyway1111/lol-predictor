"""Hard pair assertions for orientation-augmented training rows."""

from datetime import date

import pytest

from league_predictor.predraft.pipeline.dataset import (
    validate_orientation_pairs,
)


def _pair():
    return [
        {"match_id": "g1", "date": date(2026, 1, 1), "team_a": "A", "team_b": "B"},
        {"match_id": "g1", "date": date(2026, 1, 1), "team_a": "B", "team_b": "A"},
    ]


def test_orientation_pair_accepts_exact_reverse_and_preserved_weight():
    validate_orientation_pairs(_pair(), [1, 0], [0.4, 0.4])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[1].update(match_id="g2"), "match id/date"),
        (lambda rows: rows[1].update(date=date(2026, 1, 2)), "match id/date"),
        (lambda rows: rows[1].update(team_a="A", team_b="B"), "reverse teams"),
    ],
)
def test_orientation_pair_rejects_identity_date_and_team_corruption(mutate, message):
    rows = _pair()
    mutate(rows)
    with pytest.raises(ValueError, match=message):
        validate_orientation_pairs(rows, [1, 0])


def test_orientation_pair_rejects_noncomplementary_labels_and_weights():
    with pytest.raises(ValueError, match="complementary"):
        validate_orientation_pairs(_pair(), [1, 1])
    with pytest.raises(ValueError, match="preserve match weight"):
        validate_orientation_pairs(_pair(), [1, 0], [0.4, 0.5])
