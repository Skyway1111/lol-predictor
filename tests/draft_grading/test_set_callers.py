"""A format-4 family driven through the real scorers: the war room, the radar, the backfill."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from league_predictor.draft_grading import coverage
from league_predictor.draft_grading.draft_scoring import score_new_drafts
from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.player_pick_scoring import (
    PickScope,
    begin_pick_reconcile,
    spot_check_pick_rows,
)
from league_predictor.draft_grading.suggest import DraftSuggestEngine
from tests.conftest import register_model
from tests.draft_grading._set_helpers import patch_wiring, stub_index, write_set_family
from tests.draft_grading.conftest import BLUE_PICKS, RED_PICKS, role_pools, seed_game

MODEL_ID = "draft:set4"


@pytest.fixture(autouse=True)
def _stub_wiring(monkeypatch):
    """Every load in this file rebuilds the stub encoder, never 4b's production pair."""
    patch_wiring(monkeypatch)


@pytest.fixture
def corpus(draft_db):
    """A seeded corpus with its model registered; the caller paths read both."""
    register_model(draft_db, MODEL_ID, kind="draft")
    for i in range(4):
        seed_game(draft_db, f"g{i}", scheduled_ts=f"2026-03-1{i + 1}T12:00:00+00:00")
    # Every seeded game is won by T_A (blue); the stat cache reads per-row results.
    draft_db.execute("UPDATE game_players SET result = (team_id = 'T_A')")
    draft_db.commit()
    return draft_db


@pytest.fixture
def artifact(tmp_path) -> Path:
    return write_set_family(tmp_path / "draft", stub_index())


def _cache(conn) -> BulkStatsCache:
    return BulkStatsCache(conn, baseline="team_form")


def _scope() -> PickScope:
    return PickScope(year_start=date(2026, 1, 1), role_pools=role_pools())


def test_the_war_room_loads_and_scores_a_format_4_family(corpus, artifact):
    """`suggest` loads through the dispatching loader and its score path takes the cache."""
    engine = DraftSuggestEngine.load(corpus, artifact)
    blue, red = list(BLUE_PICKS.values()), list(RED_PICKS.values())
    score = engine.score(blue, red, [], [])
    assert engine._score_cache is not None  # the score cache reaches the single-draft path
    assert 0.0 < score.prob_blue_wins < 1.0
    assert score.prob_blue_wins != 0.5
    assert engine.score(red, blue, [], []).prob_blue_wins == 1.0 - score.prob_blue_wins


def test_the_pick_radar_and_its_spot_check_run_on_a_format_4_family(corpus, artifact):
    """The pick pass writes the radar and the D6 replay reproduces every row bitwise."""
    family = DraftSuggestEngine.load(corpus, artifact)._predictor
    cache = _cache(corpus)
    counts = begin_pick_reconcile(corpus, family, MODEL_ID, _scope(), stats_cache=cache).finish(
        corpus
    )
    assert counts["scored"] == 40  # four games, ten slots each
    checked = spot_check_pick_rows(corpus, family, MODEL_ID, _scope(), stats_cache=cache)
    assert checked > 0


def test_the_draft_pass_writes_a_format_4_grade(corpus, artifact):
    family = DraftSuggestEngine.load(corpus, artifact)._predictor
    counts = score_new_drafts(
        corpus,
        family,
        MODEL_ID,
        coverage.DraftScope(start=date(2026, 1, 1)),
        stats_cache=_cache(corpus),
    )
    assert counts["scored"] == 4
    rows = corpus.execute(
        "SELECT p_blue, grade_blue FROM draft_scores WHERE model_id = ?", (MODEL_ID,)
    ).fetchall()
    assert len(rows) == 4
    assert all(0.0 < row[0] < 1.0 and row[1] for row in rows)


def test_the_backfill_threads_the_cache_and_spot_checks_a_format_4_family(
    corpus, artifact, monkeypatch
):
    """The gate the review named: a set family is cache-mandatory, so the replay is not skipped."""
    family = DraftSuggestEngine.load(corpus, artifact)._predictor
    monkeypatch.setattr(coverage, "load_grader", lambda _conn, *, artifact: family)  # noqa: ARG005 - load_grader(conn, artifact=...) binds artifact by keyword
    seen: dict[str, object] = {}

    def fake_spot_check(_conn, predictor, model_id, scope, *, stats_cache, sample_size):
        seen.update(predictor=predictor, model_id=model_id, scope=scope, cache=stats_cache)
        return sample_size

    monkeypatch.setattr(coverage.player_pick_scoring, "spot_check_pick_rows", fake_spot_check)

    counts = coverage.rescore_all(corpus, MODEL_ID, model_path=artifact)

    assert counts["scored"] == 4
    assert counts["spot_checked"] == coverage.SPOT_CHECK_ROWS
    assert seen["predictor"] is family
    assert seen["cache"] is not None
