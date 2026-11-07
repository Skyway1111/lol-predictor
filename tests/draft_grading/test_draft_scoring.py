"""Verify whole-draft score stamping, coverage, and version reconciliation."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from league_predictor.draft_grading.draft_scoring import (
    DraftScope,
    _missing_draft_games,
    rescore_stale_drafts,
    score_new_drafts,
)
from tests.conftest import register_model
from tests.draft_grading.conftest import insert_draft_row, seed_game


def _scope(*, year: int = 2026, max_games: int | None = None) -> DraftScope:
    return DraftScope(start=date(year, 1, 1), max_games=max_games)


@pytest.mark.parametrize("size", [1, 3])
def test_malformed_batch_refuses_to_write_partial_results(draft_db, predictor, monkeypatch, size):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    seed_game(draft_db, "g2")
    score = predictor.predict(None)
    monkeypatch.setattr(predictor, "predict_batch", lambda *_a, **_k: [score] * size)
    with pytest.raises(RuntimeError, match=r"returned.*scores.*drafts"):
        score_new_drafts(draft_db, predictor, "draft:v1", _scope())
    assert draft_db.execute("SELECT COUNT(*) FROM draft_scores").fetchone()[0] == 0


def test_score_new_drafts_stamps_model_id(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")

    counts = score_new_drafts(draft_db, predictor, "draft:v1", _scope())

    assert counts["scored"] == 1
    row = draft_db.execute(
        "SELECT model_id, p_blue, grade_blue, grade_red, edge_pp_blue FROM draft_scores"
    ).fetchone()
    assert row["model_id"] == "draft:v1"
    assert row["p_blue"] == pytest.approx(0.55)
    assert (row["grade_blue"], row["grade_red"]) == ("B", "C")
    # An analytical quantile predictor writes NULL edge, never a fake 0.
    assert row["edge_pp_blue"] is None


def test_score_new_drafts_writes_edge_pp_when_the_predictor_has_a_curve(draft_db):
    """DraftScore.edge_pp_blue lands in the row; red is its exact negation, never stored."""
    from tests.draft_grading.conftest import MockPredictor

    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")

    edgy = MockPredictor(edge_pp_blue=4.7)
    counts = score_new_drafts(draft_db, edgy, "draft:v1", _scope())

    assert counts["scored"] == 1
    row = draft_db.execute("SELECT edge_pp_blue FROM draft_scores").fetchone()
    assert row["edge_pp_blue"] == pytest.approx(4.7)


def test_only_completed_games_with_full_picks_are_scorable(draft_db):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g_ok")
    seed_game(draft_db, "g_no_winner", winner=None)
    seed_game(draft_db, "g_incomplete", complete_picks=False)

    games = _missing_draft_games(draft_db, "draft:v1", _scope())

    assert games == ["g_ok"]


def test_rows_of_two_models_coexist(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    register_model(draft_db, "draft:v2", kind="draft")
    seed_game(draft_db, "g1")

    score_new_drafts(draft_db, predictor, "draft:v1", _scope())
    score_new_drafts(draft_db, predictor, "draft:v2", _scope())

    rows = draft_db.execute("SELECT model_id FROM draft_scores").fetchall()
    assert sorted(r["model_id"] for r in rows) == ["draft:v1", "draft:v2"]


def test_rescore_stale_drafts_replaces_and_returns_count(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    register_model(draft_db, "draft:v2", kind="draft")
    seed_game(draft_db, "g_stale")
    seed_game(draft_db, "g_fresh", scheduled_ts="2026-04-01T12:00:00+00:00")
    insert_draft_row(draft_db, "g_stale", "draft:v1")
    insert_draft_row(draft_db, "g_fresh", "draft:v2", p_blue=0.99)

    touched = rescore_stale_drafts(draft_db, predictor, "draft:v2", _scope())

    assert touched == 1  # only g_stale was under a different model
    assert (
        draft_db.execute(
            "SELECT COUNT(*) FROM draft_scores WHERE model_id = 'draft:v1'"
        ).fetchone()[0]
        == 0
    )
    # g_fresh's distinctive score is untouched.
    assert (
        draft_db.execute("SELECT p_blue FROM draft_scores WHERE game_id = 'g_fresh'").fetchone()[
            "p_blue"
        ]
        == 0.99
    )


def test_force_treats_all_games_as_stale(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    score_new_drafts(draft_db, predictor, "draft:v1", _scope())

    assert rescore_stale_drafts(draft_db, predictor, "draft:v1", _scope()) == 0
    assert rescore_stale_drafts(draft_db, predictor, "draft:v1", _scope(), force=True) == 1


def test_guard_raises_on_predraft_db(predictor):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(RuntimeError, match="current-schema database"):
            score_new_drafts(conn, predictor, "draft:v1", _scope())
    finally:
        conn.close()
