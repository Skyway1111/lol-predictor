"""The shared scoring error budget: the rate gate every scoring pass crosses before commit."""

from __future__ import annotations

import pytest

from league_predictor.draft_grading.draft_scoring import enforce_error_budget


def test_under_budget_does_not_raise() -> None:
    """1 predictor failure in 1000 attempted (0.1%) is isolated bad rows, not a fault."""
    enforce_error_budget(
        {"scored": 999, "skipped_predict_error": 1, "skipped_no_draft": 0}, scope="x"
    )


def test_over_budget_raises_and_names_the_rate_and_scope() -> None:
    """10 failures in 100 attempted (10%) is systematic — raise, naming the scope and rate."""
    with pytest.raises(RuntimeError) as ei:
        enforce_error_budget(
            {"scored": 90, "skipped_predict_error": 10, "skipped_no_draft": 3},
            scope="whole-draft scoring",
        )
    msg = str(ei.value)
    assert "whole-draft scoring" in msg
    assert "10/100" in msg
    assert "10.0%" in msg
    assert "NOT committed" in msg


def test_expected_small_skips_never_trip_the_gate() -> None:
    """No-draft and degenerate skips sit outside the rate, so an all-skip window never breaches."""
    enforce_error_budget(
        {
            "scored": 0,
            "skipped_predict_error": 0,
            "skipped_no_draft": 500,
            "skipped_degenerate": 500,
        },
        scope="x",
    )


def test_degenerate_skips_are_outside_the_denominator() -> None:
    """Degenerate skips stay out of the denominator: 5 errors + 5 scored is a 50% breach."""
    with pytest.raises(RuntimeError, match="5/10 attempted predictions failed"):
        enforce_error_budget(
            {"scored": 5, "skipped_predict_error": 5, "skipped_degenerate": 900}, scope="x"
        )


def test_no_attempted_predictions_is_never_a_breach() -> None:
    """An empty pass (nothing scored, nothing errored) is a no-op, not a divide-by-zero."""
    enforce_error_budget({"scored": 0, "skipped_predict_error": 0}, scope="x")
