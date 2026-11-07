"""The post-draft rebuild: the stack fit corpus, the scored population, freshness, no commits."""

from __future__ import annotations

import sqlite3

import pytest

from league_predictor.postdraft import coverage
from league_predictor.probability import logit, sigmoid

SERVED = "served:s_v1_aaaa"
DRAFT = "draft:dddd"


def _model(conn: sqlite3.Connection, model_id: str, kind: str) -> None:
    conn.execute("INSERT INTO models (model_id, kind) VALUES (?, ?)", (model_id, kind))


def _toy_db() -> sqlite3.Connection:
    from league_predictor.db.schema import apply_current_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    apply_current_schema(conn)
    conn.execute("INSERT INTO teams (team_id) VALUES ('A'), ('B')")
    conn.execute(
        "INSERT INTO series (series_id, team_a, team_b, scheduled_ts) "
        "VALUES ('s', 'A', 'B', '2026-06-01')"
    )
    _model(conn, SERVED, "served")
    _model(conn, DRAFT, "draft")
    return conn


def _game(conn: sqlite3.Connection, gid: str, blue: str, winner: str = "A") -> None:
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner) "
        "VALUES (?, 's', ?, ?, ?, ?)",
        (gid, int(gid[1:]), blue, "B" if blue == "A" else "A", winner),
    )


def _served(conn: sqlite3.Connection, gid: str, context: str, p_a: float, model_id=SERVED) -> None:
    conn.execute(
        "INSERT INTO predictions (subject_kind, subject_id, model_id, p_a, asof_date, "
        "context, created_at) VALUES ('game', ?, ?, ?, '2026-06-01', ?, ?)",
        (gid, model_id, p_a, context, f"2026-06-01T00:00:0{context[0]}"),
    )


def _draft(
    conn: sqlite3.Connection, gid: str, p_blue: float, model_id=DRAFT, context: str = "oos"
) -> None:
    conn.execute(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red, context) "
        "VALUES (?, ?, ?, 'B', 'B', ?)",
        (gid, model_id, p_blue, context),
    )


def _fittable(conn: sqlite3.Connection, n: int = 6, *, blue: str = "A") -> None:
    """Plant ``n`` games both parents cover out-of-sample — the minimum the stack fit needs."""
    for i in range(n):
        gid = f"f{10 + i}"
        _game(conn, gid, blue, winner="A" if i % 2 else "B")
        _served(conn, gid, "oos", 0.55)
        _draft(conn, gid, 0.52)


def _rows(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {r["game_id"]: r for r in conn.execute("SELECT * FROM postdraft_scores")}


def _stack_rows(conn: sqlite3.Connection) -> dict[str, sqlite3.Row]:
    return {r["team_id"]: r for r in conn.execute("SELECT * FROM postdraft_stack")}


def _expected(conn: sqlite3.Connection, served_blue: float, draft_blue: float, blue: str) -> float:
    """The stack's own p_blue under the coefficients the rebuild just wrote."""
    stack = _stack_rows(conn)
    coefficient = stack["*"]["beta"] + stack[blue]["u"] + stack["B" if blue == "A" else "A"]["u"]
    return sigmoid(stack["*"]["a"] * logit(served_blue) + coefficient * logit(draft_blue))


def test_only_both_parent_games_get_rows():
    conn = _toy_db()
    _fittable(conn)
    _game(conn, "g1", "A")
    _served(conn, "g1", "oos", 0.6)
    _draft(conn, "g1", 0.55, context="backfill")
    _game(conn, "g2", "A")  # served only, no draft
    _served(conn, "g2", "oos", 0.6)
    _game(conn, "g3", "A")  # draft only, no served
    _draft(conn, "g3", 0.55, context="backfill")

    n = coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    rows = _rows(conn)
    assert n == 7
    assert set(rows) == {"g1", *(f"f{10 + i}" for i in range(6))}
    assert rows["g1"]["p_blue"] == pytest.approx(_expected(conn, 0.6, 0.55, "A"))


def test_oos_preferred_over_backfill_on_both_parents():
    conn = _toy_db()
    _fittable(conn)
    _game(conn, "g1", "A")
    _served(conn, "g1", "backfill", 0.4)
    _served(conn, "g1", "oos", 0.7)
    _draft(conn, "g1", 0.3, context="backfill")
    _draft(conn, "g1", 0.55, context="oos")
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    assert _rows(conn)["g1"]["p_blue"] == pytest.approx(_expected(conn, 0.7, 0.55, "A"))


def test_backfill_only_game_still_covered():
    conn = _toy_db()
    _fittable(conn)
    _game(conn, "g1", "A")
    _served(conn, "g1", "backfill", 0.4)
    _draft(conn, "g1", 0.55, context="backfill")
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    assert _rows(conn)["g1"]["p_blue"] == pytest.approx(_expected(conn, 0.4, 0.55, "A"))


def test_blue_orientation_when_blue_is_team_b():
    conn = _toy_db()
    _fittable(conn)
    _game(conn, "g1", "B")  # blue is team_b -> served_blue = 1 - p_a
    _served(conn, "g1", "oos", 0.7)
    _draft(conn, "g1", 0.55, context="backfill")
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    assert _rows(conn)["g1"]["p_blue"] == pytest.approx(_expected(conn, 1 - 0.7, 0.55, "B"))


def test_the_stack_row_set_is_the_global_row_plus_every_fit_team():
    conn = _toy_db()
    _fittable(conn)
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    stack = _stack_rows(conn)
    assert set(stack) == {"*", "A", "B"}
    assert stack["*"]["n_oos"] == 6
    assert stack["*"]["u"] == 0.0
    assert stack["*"]["penalty"] is not None
    assert stack["A"]["a"] is None
    assert stack["A"]["beta"] is None
    assert stack["A"]["n_oos"] == 6


def test_a_backfill_only_pair_refuses_to_fit():
    """No oos corpus means no stack; the rebuild raises instead of writing a default."""
    conn = _toy_db()
    _game(conn, "g1", "A")
    _served(conn, "g1", "backfill", 0.6)
    _draft(conn, "g1", 0.55, context="backfill")
    with pytest.raises(RuntimeError, match="at least 2 rows"):
        coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    assert _rows(conn) == {}
    assert _stack_rows(conn) == {}


def test_stamps_are_the_parent_ids_used():
    conn = _toy_db()
    _fittable(conn)
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    for row in (*_rows(conn).values(), *_stack_rows(conn).values()):
        assert row["served_model_id"] == SERVED
        assert row["draft_model_id"] == DRAFT


def test_rebuild_is_idempotent():
    conn = _toy_db()
    _fittable(conn)
    n1 = coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    first = dict(_rows(conn)["f10"])
    first_stack = dict(_stack_rows(conn)["*"])
    n2 = coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    assert n1 == n2 == 6
    assert dict(_rows(conn)["f10"]) == first
    assert dict(_stack_rows(conn)["*"]) == first_stack


def test_assert_fresh_passes_then_raises_on_id_change():
    conn = _toy_db()
    _fittable(conn)
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    coverage.assert_fresh(conn, served_model_id=SERVED, draft_model_id=DRAFT)  # no raise
    with pytest.raises(RuntimeError, match="current model pair"):
        coverage.assert_fresh(conn, served_model_id="served:other", draft_model_id=DRAFT)


def test_assert_fresh_sees_a_stale_stack_even_when_the_scores_are_current():
    """The stack is stamped like the scores, so a stale coefficient row cannot hide."""
    conn = _toy_db()
    _fittable(conn)
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    conn.execute("UPDATE postdraft_stack SET draft_model_id = 'draft:stale' WHERE team_id = '*'")
    with pytest.raises(RuntimeError, match="postdraft_stack"):
        coverage.assert_fresh(conn, served_model_id=SERVED, draft_model_id=DRAFT)


def test_defaults_resolve_from_parent_tables():
    conn = _toy_db()
    _fittable(conn)
    n = coverage.rebuild(conn)  # no explicit ids: resolve from predictions + draft_scores
    assert n == 6
    assert _rows(conn)["f10"]["served_model_id"] == SERVED
    assert _rows(conn)["f10"]["draft_model_id"] == DRAFT
    coverage.assert_fresh(conn)  # default resolution agrees


def test_multiple_draft_ids_raises():
    conn = _toy_db()
    _model(conn, "draft:other", "draft")
    _fittable(conn)
    _draft(conn, "f10", 0.55, model_id="draft:other")
    with pytest.raises(RuntimeError, match="multiple ids"):
        coverage.rebuild(conn)


def test_explicit_served_id_used_amid_a_superseded_corpus():
    """With two served ids in predictions, an explicit served_model_id stacks that id alone."""
    conn = _toy_db()
    _model(conn, "served:old", "served")
    _fittable(conn)
    _game(conn, "g1", "A")
    _served(conn, "g1", "oos", 0.9, model_id="served:old")  # superseded, must be ignored
    _served(conn, "g1", "oos", 0.6, model_id=SERVED)  # the new model
    _draft(conn, "g1", 0.55, context="backfill")
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    assert _rows(conn)["g1"]["p_blue"] == pytest.approx(_expected(conn, 0.6, 0.55, "A"))


def test_rebuild_does_not_commit():
    conn = _toy_db()
    _fittable(conn)
    coverage.rebuild(conn, served_model_id=SERVED, draft_model_id=DRAFT)
    conn.rollback()
    assert _rows(conn) == {}
    assert _stack_rows(conn) == {}
