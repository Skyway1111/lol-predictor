"""Tests for the strictly-before team-state temporal seam."""

from __future__ import annotations

from datetime import date

from league_predictor.db.asof import team_state_asof, team_state_coverage
from league_predictor.db.rows import TeamStateRow
from league_predictor.db.writes import replace_team_state
from tests.conftest import register_model


def _seed(rebuild_db, model_id: str) -> None:
    register_model(rebuild_db, model_id)
    replace_team_state(
        rebuild_db,
        model_id,
        [
            TeamStateRow("A", model_id, "2026-01-05", strength=1.0),
            TeamStateRow("A", model_id, "2026-01-10", strength=2.0),
            TeamStateRow("B", model_id, "2026-01-05", strength=3.0),
            TeamStateRow("B", model_id, "2026-01-10", strength=4.0),
        ],
    )


def test_asof_excludes_same_day_rows(rebuild_db):
    """A row stamped D is NOT visible when pricing an event on D."""
    _seed(rebuild_db, "rating:m")
    result = team_state_asof(rebuild_db, ["A", "B"], "rating:m", date(2026, 1, 5))
    assert result == {}


def test_asof_includes_the_prior_day_row(rebuild_db):
    """A row stamped D IS visible for an event on D+1."""
    _seed(rebuild_db, "rating:m")
    result = team_state_asof(rebuild_db, ["A", "B"], "rating:m", date(2026, 1, 6))
    assert set(result) == {"A", "B"}
    assert result["A"].asof_date == "2026-01-05"
    assert result["A"].strength == 1.0
    assert result["B"].asof_date == "2026-01-05"


def test_asof_picks_the_latest_row_strictly_before(rebuild_db):
    """The resolved row is the latest strictly before the event date — same-day excluded."""
    _seed(rebuild_db, "rating:m")

    later = team_state_asof(rebuild_db, ["A", "B"], "rating:m", date(2026, 1, 15))
    assert later["A"].asof_date == "2026-01-10"
    assert later["A"].strength == 2.0
    assert later["B"].asof_date == "2026-01-10"

    on_appearance_day = team_state_asof(rebuild_db, ["A", "B"], "rating:m", date(2026, 1, 10))
    assert on_appearance_day["A"].asof_date == "2026-01-05"
    assert on_appearance_day["B"].asof_date == "2026-01-05"


def test_asof_empty_team_ids_returns_empty(rebuild_db):
    _seed(rebuild_db, "rating:m")
    assert team_state_asof(rebuild_db, [], "rating:m", date(2026, 1, 15)) == {}


def test_asof_is_scoped_to_the_model_id(rebuild_db):
    """A different model's rows never leak into another model's as-of read."""
    _seed(rebuild_db, "rating:m")
    register_model(rebuild_db, "rating:other")
    replace_team_state(
        rebuild_db,
        "rating:other",
        [
            TeamStateRow("A", "rating:other", "2026-01-08", strength=99.0),
        ],
    )
    result = team_state_asof(rebuild_db, ["A"], "rating:m", date(2026, 1, 9))
    assert result["A"].strength == 1.0  # the rating:m 01-05 row, not rating:other


def test_team_state_coverage_reports_count_and_max(rebuild_db):
    _seed(rebuild_db, "rating:m")
    count, max_asof = team_state_coverage(rebuild_db, "rating:m")
    assert count == 4
    assert max_asof == "2026-01-10"


def test_team_state_coverage_empty(rebuild_db):
    register_model(rebuild_db, "rating:empty")
    count, max_asof = team_state_coverage(rebuild_db, "rating:empty")
    assert count == 0
    assert max_asof is None
