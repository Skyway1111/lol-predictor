"""Backfill-engine dispatch and verification wiring (blend design D6)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from league_predictor.draft_grading import coverage
from tests.conftest import register_model
from tests.draft_grading.conftest import MockPredictor, seed_game

# ---- the dispatching loader seam ----------------------------------------


@pytest.fixture
def backfill(draft_db, monkeypatch):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    predictor = MockPredictor()
    monkeypatch.setattr(coverage, "load_grader", lambda *_a, **_k: predictor)
    monkeypatch.setattr(coverage, "BulkStatsCache", lambda *_a, **_k: object())
    return predictor


def test_backfill_shell_loads_through_the_dispatching_loader(draft_db, monkeypatch):
    """The shell loads the explicit format-4 artifact through live.load_grader."""
    seen: dict[str, Path] = {}

    def fake_load_grader(_conn, *, artifact: Path):
        seen["artifact"] = artifact
        return MockPredictor()

    monkeypatch.setattr(coverage, "load_grader", fake_load_grader)
    predictor, draft_scope, pick_scope = coverage._build_backfill_shell(
        draft_db, model_path=Path("models/draft/draft.json")
    )
    assert seen["artifact"] == Path("models/draft/draft.json")
    assert isinstance(predictor, MockPredictor)
    assert draft_scope.start == coverage._BACKFILL_START
    assert pick_scope.year_start == coverage._BACKFILL_START


def test_rescore_all_spot_checks_the_family(draft_db, monkeypatch, backfill):
    blend = backfill
    # The mock sides cannot run the blend's real batch surfaces; pin only the wiring —
    # rescore_all calls the spot check with the run's predictor, id, scope, cache, and count.
    seen: dict[str, object] = {}

    def fake_spot_check(_conn, predictor, model_id, scope, *, stats_cache, sample_size):
        seen.update(
            predictor=predictor,
            model_id=model_id,
            scope=scope,
            stats_cache=stats_cache,
            sample_size=sample_size,
        )
        return 7

    monkeypatch.setattr(coverage.player_pick_scoring, "spot_check_pick_rows", fake_spot_check)

    monkeypatch.setattr(
        coverage,
        "begin_pick_reconcile",
        lambda *_a, **_k: SimpleNamespace(finish=lambda _conn: {"scored": 0}),
    )
    monkeypatch.setattr(coverage, "rescore_stale_drafts", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        coverage,
        "score_new_drafts",
        lambda *_a, **_k: {"scored": 0},
    )

    counts = coverage.rescore_all(draft_db, "draft:v1", model_path=Path("x"))

    assert counts["spot_checked"] == 7
    assert seen["predictor"] is blend
    assert seen["model_id"] == "draft:v1"
    assert seen["sample_size"] == coverage.SPOT_CHECK_ROWS
    assert seen["stats_cache"] is not None
    assert isinstance(seen["scope"], coverage.PickScope)


@pytest.mark.usefixtures("backfill")
def test_rescore_all_overlaps_draft_passes_with_the_pick_pool(draft_db, monkeypatch):
    """The draft (GPU) passes run between pick dispatch and pick join (retrain-wall overlap)."""
    order: list[str] = []

    class _FakePass:
        def finish(self, _conn):
            order.append("pick_finish")
            return {"scored": 0}

    monkeypatch.setattr(
        coverage,
        "begin_pick_reconcile",
        lambda *_a, **_k: order.append("pick_begin") or _FakePass(),
    )
    monkeypatch.setattr(
        coverage, "rescore_stale_drafts", lambda *_a, **_k: order.append("draft") or 0
    )
    monkeypatch.setattr(
        coverage, "score_new_drafts", lambda *_a, **_k: order.append("draft_new") or {"scored": 0}
    )

    coverage.rescore_all(draft_db, "draft:v1", model_path=Path("x"))

    assert order == ["pick_begin", "draft", "draft_new", "pick_finish"]


# ---- rescore_all is a full recompute: a same-id row is not proof of freshness ----

# A refresh can revise facts under stored rows, and one revision shifts the walk-forward
# features of every later game — so coverage under the current model id proves nothing.


@pytest.mark.usefixtures("backfill")
def test_rescore_all_recomputes_draft_rows_under_the_current_model_id(draft_db):

    coverage.rescore_all(draft_db, "draft:v1", model_path=Path("x"))
    fresh = draft_db.execute("SELECT p_blue FROM draft_scores WHERE game_id = 'g1'").fetchone()[0]

    # Simulate the revision: the stored value diverges from scoring over the current facts.
    draft_db.execute("UPDATE draft_scores SET p_blue = -1.0 WHERE game_id = 'g1'")

    coverage.rescore_all(draft_db, "draft:v1", model_path=Path("x"))

    got = draft_db.execute("SELECT p_blue FROM draft_scores WHERE game_id = 'g1'").fetchone()[0]
    assert got == fresh


@pytest.mark.usefixtures("backfill")
def test_rescore_all_forces_both_passes(draft_db, monkeypatch):
    """Both score passes run version-safe full re-scores (``force=True``)."""
    forced: dict[str, object] = {}

    monkeypatch.setattr(
        coverage,
        "begin_pick_reconcile",
        lambda *_a, **k: (
            forced.__setitem__("pick", k.get("force"))
            or SimpleNamespace(finish=lambda _conn: {"scored": 0})
        ),
    )
    monkeypatch.setattr(
        coverage,
        "rescore_stale_drafts",
        lambda *_a, **k: forced.__setitem__("draft", k.get("force")) or 0,
    )
    monkeypatch.setattr(coverage, "score_new_drafts", lambda *_a, **_k: {"scored": 0})

    coverage.rescore_all(draft_db, "draft:v1", model_path=Path("x"))

    assert forced == {"pick": True, "draft": True}
