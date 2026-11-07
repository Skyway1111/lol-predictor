"""Schema-level invariants: table set, orientation CHECK, FK enforcement, natural keys."""

from __future__ import annotations

import sqlite3

import pytest

from league_predictor.db.rows import PredictionRow
from league_predictor.db.writes import append_predictions
from tests.conftest import register_model
from tests.db._schema_helpers import CANONICAL_TABLES, _table_names


def _seed_teams(conn: sqlite3.Connection) -> None:
    conn.executemany("INSERT INTO teams (team_id) VALUES (?)", [("A",), ("B",)])


def test_schema_creates_the_pinned_table_set(rebuild_db):
    assert _table_names(rebuild_db) == sorted(CANONICAL_TABLES)


def test_series_orientation_check_rejects_non_canonical(rebuild_db):
    """team_a must sort strictly before team_b; team_a >= team_b is rejected."""
    _seed_teams(rebuild_db)
    with pytest.raises(sqlite3.IntegrityError):
        rebuild_db.execute("INSERT INTO series (series_id, team_a, team_b) VALUES ('s1', 'B', 'A')")
    with pytest.raises(sqlite3.IntegrityError):
        rebuild_db.execute("INSERT INTO series (series_id, team_a, team_b) VALUES ('s2', 'A', 'A')")


def test_series_orientation_check_accepts_canonical(rebuild_db):
    _seed_teams(rebuild_db)
    rebuild_db.execute("INSERT INTO series (series_id, team_a, team_b) VALUES ('s1', 'A', 'B')")
    count = rebuild_db.execute("SELECT COUNT(*) FROM series").fetchone()[0]
    assert count == 1


def test_prediction_with_unknown_model_id_fails_fk(rebuild_db):
    """A derived row must reference a registered model — FK enforced."""
    row = PredictionRow(
        subject_kind="series",
        subject_id="s1",
        model_id="rating:doesnotexist",
        p_a=0.6,
        asof_date="2026-07-01",
        context="backfill",
        created_at="2026-07-01T10:00:00+00:00",
    )
    with pytest.raises(sqlite3.IntegrityError):
        append_predictions(rebuild_db, [row])


def test_predictions_natural_key_is_unique(rebuild_db):
    """(subject_kind, subject_id, model_id, context, created_at) is unique — append-only."""
    register_model(rebuild_db, "rating:abc")
    row = PredictionRow(
        subject_kind="series",
        subject_id="s1",
        model_id="rating:abc",
        p_a=0.6,
        asof_date="2026-07-01",
        context="backfill",
        created_at="2026-07-01T10:00:00+00:00",
    )
    append_predictions(rebuild_db, [row])
    with pytest.raises(sqlite3.IntegrityError):
        append_predictions(rebuild_db, [row])


def test_predictions_asof_date_is_required_and_format_checked(rebuild_db):
    """asof_date is NOT NULL and 'YYYY-MM-DD': lexicographic order carries the AS-OF reads."""
    register_model(rebuild_db, "rating:abc")
    base = (
        "INSERT INTO predictions (subject_kind, subject_id, model_id, p_a, "
        "asof_date, context, created_at) VALUES ('series', 's1', 'rating:abc', "
        "0.5, ?, 'backfill', '2026-07-01T10:00:00+00:00')"
    )
    with pytest.raises(sqlite3.IntegrityError):
        rebuild_db.execute(base, (None,))
    with pytest.raises(sqlite3.IntegrityError):
        rebuild_db.execute(base, ("07/01/2026",))
    rebuild_db.execute(base, ("2026-07-01",))


def test_team_state_asof_date_format_checked(rebuild_db):
    register_model(rebuild_db, "rating:abc")
    with pytest.raises(sqlite3.IntegrityError):
        rebuild_db.execute(
            "INSERT INTO team_state (team_id, model_id, asof_date) "
            "VALUES ('A', 'rating:abc', '20260701')"
        )
    rebuild_db.execute(
        "INSERT INTO team_state (team_id, model_id, asof_date) "
        "VALUES ('A', 'rating:abc', '2026-07-01')"
    )


def test_games_unique_per_series_and_game_no(rebuild_db):
    _seed_teams(rebuild_db)
    rebuild_db.execute("INSERT INTO series (series_id, team_a, team_b) VALUES ('s1', 'A', 'B')")
    rebuild_db.execute("INSERT INTO games (game_id, series_id, game_no) VALUES ('g1', 's1', 1)")
    with pytest.raises(sqlite3.IntegrityError):
        rebuild_db.execute("INSERT INTO games (game_id, series_id, game_no) VALUES ('g2', 's1', 1)")


def test_pick_scores_side_is_checked(rebuild_db):
    _seed_teams(rebuild_db)
    rebuild_db.execute("INSERT INTO series (series_id, team_a, team_b) VALUES ('s1', 'A', 'B')")
    rebuild_db.execute("INSERT INTO games (game_id, series_id, game_no) VALUES ('g1', 's1', 1)")
    register_model(rebuild_db, "draft:v8", kind="draft")
    with pytest.raises(sqlite3.IntegrityError):
        rebuild_db.execute(
            "INSERT INTO pick_scores (game_id, side, role, model_id) "
            "VALUES ('g1', 'Blue', 'mid', 'draft:v8')"
        )
    rebuild_db.execute(
        "INSERT INTO pick_scores (game_id, side, role, model_id) "
        "VALUES ('g1', 'blue', 'mid', 'draft:v8')"
    )
