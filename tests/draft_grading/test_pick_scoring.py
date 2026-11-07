"""Pick-score versioning against the rebuilt schema, with a mocked predictor."""

from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path

import pytest

import league_predictor.draft_grading.player_pick_scoring as pps
from league_predictor.draft_grading.player_pick_scoring import (
    PickScope,
    begin_pick_reconcile,
    spot_check_pick_rows,
)
from tests.conftest import register_model
from tests.draft_grading.conftest import insert_pick_row, role_pools, seed_game


def _scope(*, year: int = 2026) -> PickScope:
    return PickScope(year_start=date(year, 1, 1), role_pools=role_pools())


def _reconcile(conn, predictor, model_id, scope, **kwargs) -> dict[str, int]:
    return begin_pick_reconcile(conn, predictor, model_id, scope, **kwargs).finish(conn)


# --- stamping ---------------------------------------------------------------


def test_reconcile_stamps_model_id_on_every_slot(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")

    counts = _reconcile(draft_db, predictor, "draft:v1", _scope())

    assert counts["scored"] == 10
    assert counts["skipped_predict_error"] == 0
    rows = draft_db.execute("SELECT model_id FROM pick_scores").fetchall()
    assert len(rows) == 10
    assert {r["model_id"] for r in rows} == {"draft:v1"}


def test_reconcile_is_idempotent(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    _reconcile(draft_db, predictor, "draft:v1", _scope())

    again = _reconcile(draft_db, predictor, "draft:v1", _scope())

    assert again["scored"] == 0
    assert draft_db.execute("SELECT COUNT(*) FROM pick_scores").fetchone()[0] == 10


# --- missing-slot detection over draft_actions ------------------------------


def test_ban_holes_do_not_block_detection(draft_db, predictor):
    """Detection counts only pick actions, so ban holes leave the game fully scorable."""
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g_ok")

    counts = _reconcile(draft_db, predictor, "draft:v1", _scope())

    assert counts["scored"] == 10
    games = draft_db.execute("SELECT DISTINCT game_id FROM pick_scores").fetchall()
    assert {g["game_id"] for g in games} == {"g_ok"}


def test_incomplete_picks_are_excluded(draft_db, predictor):
    """A game with 9 pick actions (not 10) is not scorable."""
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g_ok")
    seed_game(draft_db, "g_bad", complete_picks=False)

    _reconcile(draft_db, predictor, "draft:v1", _scope())

    games = draft_db.execute("SELECT DISTINCT game_id FROM pick_scores").fetchall()
    assert {g["game_id"] for g in games} == {"g_ok"}


def test_only_unscored_slots_are_picked_up(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    for side in ("blue", "red"):
        for role in ("top", "jng", "mid", "bot", "sup"):
            if (side, role) in {("blue", "jng"), ("red", "sup")}:
                continue
            insert_pick_row(draft_db, "g1", side, role, "draft:v1")

    slots = pps._reconcile_slots(draft_db, "draft:v1", _scope(), force=False)
    assert {(m["side"], m["role"]) for m in slots} == {("blue", "jng"), ("red", "sup")}

    counts = _reconcile(draft_db, predictor, "draft:v1", _scope())
    assert counts["scored"] == 2


def test_slots_outside_year_start_are_excluded(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g_old", scheduled_ts="2025-12-31T12:00:00+00:00")
    seed_game(draft_db, "g_new", scheduled_ts="2026-01-15T12:00:00+00:00")

    _reconcile(draft_db, predictor, "draft:v1", _scope())

    games = draft_db.execute("SELECT DISTINCT game_id FROM pick_scores").fetchall()
    assert {g["game_id"] for g in games} == {"g_new"}


# --- version reconcile ------------------------------------------------------


def test_reconcile_replaces_stale_rows(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    register_model(draft_db, "draft:v2", kind="draft")
    seed_game(draft_db, "g1")
    _reconcile(draft_db, predictor, "draft:v1", _scope())

    counts = _reconcile(draft_db, predictor, "draft:v2", _scope())

    assert counts["scored"] == 10
    assert (
        draft_db.execute("SELECT COUNT(*) FROM pick_scores WHERE model_id = 'draft:v1'").fetchone()[
            0
        ]
        == 0
    )
    assert (
        draft_db.execute("SELECT COUNT(*) FROM pick_scores WHERE model_id = 'draft:v2'").fetchone()[
            0
        ]
        == 10
    )

    # Idempotent: nothing stale remains.
    assert _reconcile(draft_db, predictor, "draft:v2", _scope())["scored"] == 0


def test_reconcile_leaves_fresh_target_rows_untouched(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    register_model(draft_db, "draft:v2", kind="draft")
    seed_game(draft_db, "g_stale")
    seed_game(draft_db, "g_fresh", scheduled_ts="2026-04-01T12:00:00+00:00")
    # g_stale carries v1 rows (to be converted); g_fresh already carries v2 rows
    # at a distinctive score that must survive unchanged.
    for role in ("top", "jng", "mid", "bot", "sup"):
        for side in ("blue", "red"):
            insert_pick_row(draft_db, "g_stale", side, role, "draft:v1")
            insert_pick_row(draft_db, "g_fresh", side, role, "draft:v2", actual=0.99)

    counts = _reconcile(draft_db, predictor, "draft:v2", _scope())

    assert counts["scored"] == 10  # only g_stale's slots were stale
    assert (
        draft_db.execute(
            "SELECT COUNT(*) FROM pick_scores WHERE game_id = 'g_stale' AND model_id = 'draft:v1'"
        ).fetchone()[0]
        == 0
    )
    assert (
        draft_db.execute(
            "SELECT COUNT(*) FROM pick_scores WHERE game_id = 'g_stale' AND model_id = 'draft:v2'"
        ).fetchone()[0]
        == 10
    )
    fresh = draft_db.execute(
        "SELECT actual_score FROM pick_scores WHERE game_id = 'g_fresh'"
    ).fetchall()
    assert len(fresh) == 10
    assert all(r["actual_score"] == 0.99 for r in fresh)  # never re-scored


def test_force_treats_all_rows_as_stale(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    _reconcile(draft_db, predictor, "draft:v1", _scope())

    # Nothing is missing or under a different model, so the plain reconcile no-ops...
    assert _reconcile(draft_db, predictor, "draft:v1", _scope())["scored"] == 0
    # ...but --force re-scores every in-scope slot.
    assert _reconcile(draft_db, predictor, "draft:v1", _scope(), force=True)["scored"] == 10
    assert draft_db.execute("SELECT COUNT(*) FROM pick_scores").fetchone()[0] == 10


# --- blind/counter dispatch -------------------------------------------------


def test_blind_pick_uses_marginalized_when_prior_present(draft_db, predictor, monkeypatch):
    monkeypatch.setattr(
        pps,
        "cached_enemy_response_distribution",
        lambda db, role, as_of, top_k=10: [("Ahri", 1.0)],  # noqa: ARG005 - mirrors cached_enemy_response_distribution(db, role, as_of, top_k=...)
    )
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")

    _reconcile(draft_db, predictor, "draft:v1", _scope())

    dispatch = {(side, role): enemy for (_gid, side, role, enemy) in predictor.swap_calls}
    # blue top picks before red top → blind: marginalizes over the meta prior.
    assert dispatch[("blue", "top")] == [("Ahri", 1.0)]
    # red top responds → counter: scores against the KNOWN enemy (blue top = Gnar).
    assert dispatch[("red", "top")] == [("Gnar", 1.0)]


# --- the overlapped parent-cache future (blend D2/D6) ------------------------


def test_reconcile_resolves_stats_cache_future_on_the_parent_path(draft_db, predictor, monkeypatch):
    """A cache future is resolved before the in-process pass; workers build their own."""
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    resolved = object()

    class _Future:
        def result(self):
            return resolved

    seen: dict[str, object] = {}
    real_process = pps._process

    def spy(conn, pred, model_id, slots, scope, stats_cache, **kwargs):
        seen["stats_cache"] = stats_cache
        return real_process(conn, pred, model_id, slots, scope, None, **kwargs)

    monkeypatch.setattr(pps, "_process", spy)
    _reconcile(
        draft_db, predictor, "draft:v1", _scope(), stats_cache=None, stats_cache_future=_Future()
    )
    assert seen["stats_cache"] is resolved


# --- the deferred-join pick pass (retrain-wall overlap) -----------------------


def test_small_corpus_scores_at_begin_and_finish_returns_counts(draft_db, predictor):
    """An in-process pass scores during begin; finish only hands back the counts."""
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")

    pick_pass = pps.begin_pick_reconcile(draft_db, predictor, "draft:v1", _scope())
    counts = pick_pass.finish(draft_db)

    assert counts["scored"] == 10
    assert draft_db.execute("SELECT COUNT(*) FROM pick_scores").fetchone()[0] == 10


def test_parallel_begin_defers_the_flush_to_finish(tmp_path, monkeypatch):
    """The parallel path's rows land only at finish(); workers stay read-only."""
    from league_predictor.db.schema import apply_current_schema
    from tests.draft_grading.conftest import ROLES, MockPredictor

    draft_db = sqlite3.connect(tmp_path / "facts.db")
    draft_db.row_factory = sqlite3.Row
    apply_current_schema(draft_db)
    draft_db.executemany("INSERT INTO teams (team_id) VALUES (?)", [("T_A",), ("T_B",)])
    players = [f"bp_{r}" for r in ROLES] + [f"rp_{r}" for r in ROLES]
    draft_db.executemany("INSERT INTO players (player_id) VALUES (?)", [(p,) for p in players])
    draft_db.commit()
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    monkeypatch.setattr(pps.resources, "bounded_workers", lambda *_a, **_k: 1)
    monkeypatch.setattr(
        pps,
        "_parallel_pick_worker",
        lambda _db, _model, model_id, slots, **_kw: (
            [
                (s["game_id"], s["side"], s["role"], model_id, "p", 0.5, 3, 0.7, 0.3, 0.5)
                for s in slots
            ],
            {
                "scored": len(slots),
                "skipped_no_draft": 0,
                "skipped_predict_error": 0,
                "skipped_degenerate": 0,
            },
        ),
    )
    # Force the parallel branch regardless of slot count.
    monkeypatch.setattr(pps, "_PARALLEL_MIN_SLOTS", 1)

    pick_pass = pps.begin_pick_reconcile(
        draft_db,
        MockPredictor(),
        "draft:v1",
        _scope(),
        parallel_model_path=Path("models/draft/draft.json"),
    )
    assert draft_db.execute("SELECT COUNT(*) FROM pick_scores").fetchone()[0] == 0
    counts = pick_pass.finish(draft_db)
    assert counts["scored"] == 10
    assert draft_db.execute("SELECT COUNT(*) FROM pick_scores").fetchone()[0] == 10


# --- the D6 reproducibility spot check ----------------------------------------


def test_spot_check_passes_on_freshly_scored_rows(draft_db, predictor):
    """Every stored row re-scored on the singular path must match bitwise."""
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    _reconcile(draft_db, predictor, "draft:v1", _scope())

    checked = spot_check_pick_rows(draft_db, predictor, "draft:v1", _scope(), stats_cache=None)

    assert checked == 10  # sample_size (20) caps at the population


def test_spot_check_raises_on_a_drifted_row(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    _reconcile(draft_db, predictor, "draft:v1", _scope())
    draft_db.execute(
        "UPDATE pick_scores SET percentile = percentile + 1e-9 "
        "WHERE game_id = 'g1' AND side = 'blue' AND role = 'top'"
    )
    draft_db.commit()

    with pytest.raises(RuntimeError, match="g1/blue/top"):
        spot_check_pick_rows(draft_db, predictor, "draft:v1", _scope(), stats_cache=None)


def test_spot_check_on_an_empty_table_checks_nothing(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    assert spot_check_pick_rows(draft_db, predictor, "draft:v1", _scope(), stats_cache=None) == 0


def test_spot_check_sample_size_bounds_the_work(draft_db, predictor):
    register_model(draft_db, "draft:v1", kind="draft")
    seed_game(draft_db, "g1")
    _reconcile(draft_db, predictor, "draft:v1", _scope())

    checked = spot_check_pick_rows(
        draft_db, predictor, "draft:v1", _scope(), stats_cache=None, sample_size=3
    )

    assert checked == 3


# --- guard ------------------------------------------------------------------


def test_guard_raises_on_predraft_db(predictor):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    try:
        with pytest.raises(RuntimeError, match="current-schema database"):
            begin_pick_reconcile(conn, predictor, "draft:v1", _scope())
    finally:
        conn.close()
