"""Tests for state_export.ensure_team_state_current, with the serving replay stubbed."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from league_predictor.analytics import team_snapshot
from league_predictor.db.asof import team_state_asof
from league_predictor.db.rows import ModelRow, TeamStateRow
from league_predictor.db.schema import apply_current_schema
from league_predictor.predraft.ratings.skill_filter import state_export
from tests.predraft.conftest import insert_game

RATING_ID = "rating:testrating"
LATEST_DAY = "2026-03-10"


@pytest.fixture
def facts_db():
    """Facts DB with one completed game (latest corpus day = LATEST_DAY)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    insert_game(conn, game_id="g1", day=LATEST_DAY, team1="A", team2="B", winner="A")
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def spied(monkeypatch):
    """Stub the rating identity and serving replay; return the export-call recorder."""
    export_calls: list = []

    def _fake_export(_conn, _backend, model_id):
        export_calls.append((model_id,))
        return [
            TeamStateRow(
                team_id="A",
                model_id=model_id,
                asof_date=LATEST_DAY,
                strength=1.5,
            ),
            TeamStateRow(
                team_id="B",
                model_id=model_id,
                asof_date=LATEST_DAY,
                strength=-0.3,
            ),
        ]

    monkeypatch.setattr(
        "league_predictor.predraft.pipeline.model_identity.current_rating_model_id",
        lambda: RATING_ID,
    )
    monkeypatch.setattr(
        "league_predictor.predraft.pipeline.model_identity.model_row",
        lambda kind: ModelRow(model_id=RATING_ID, kind=kind, artifact_sha="0" * 64),
    )
    monkeypatch.setattr(
        "league_predictor.predraft.pipeline.chosen_skill_filter.chosen_skill_filter_backend",
        object,
    )
    monkeypatch.setattr(
        "league_predictor.predraft.ratings.skill_filter.state_export.export_team_state",
        _fake_export,
    )
    return export_calls


def _register(conn, model_id):
    conn.execute(
        "INSERT OR IGNORE INTO models (model_id, kind) VALUES (?, 'rating')",
        (model_id,),
    )


def _seed_state(conn, team_id, model_id, asof, strength):
    conn.execute(
        "INSERT INTO team_state (team_id, model_id, asof_date, strength) VALUES (?, ?, ?, ?)",
        (team_id, model_id, asof, strength),
    )


def test_stale_coverage_triggers_rebuild(facts_db, spied):
    """No coverage under the served rating id -> the replay runs, rows land in team_state."""
    assert state_export.ensure_team_state_current(facts_db) is True

    assert len(spied) == 1, "the serving replay must run when coverage is stale"
    rows = {
        (r["team_id"], r["model_id"])
        for r in facts_db.execute("SELECT team_id, model_id FROM team_state")
    }
    assert rows == {("A", RATING_ID), ("B", RATING_ID)}


def test_fresh_coverage_no_ops(facts_db, spied):
    """A row at or after the latest corpus day skips the replay entirely."""
    _register(facts_db, RATING_ID)
    _seed_state(facts_db, "A", RATING_ID, LATEST_DAY, 1.2)
    facts_db.commit()

    assert state_export.ensure_team_state_current(facts_db) is False

    assert spied == [], "the expensive replay must not run on fresh coverage"
    # the pre-seeded row is untouched
    assert facts_db.execute(
        "SELECT strength FROM team_state WHERE team_id='A' AND model_id=?", (RATING_ID,)
    ).fetchone()[0] == pytest.approx(1.2)


def test_foreign_id_rows_deleted(facts_db, spied):
    """Rows under any non-current model id are dropped — on the fresh no-op path too."""
    _register(facts_db, RATING_ID)
    _register(facts_db, "rating:retired_old")
    _seed_state(facts_db, "A", RATING_ID, LATEST_DAY, 1.2)  # fresh served coverage
    _seed_state(facts_db, "A", "rating:retired_old", "2026-01-01", 9.9)
    facts_db.commit()

    state_export.ensure_team_state_current(facts_db)

    assert spied == [], "coverage was fresh — no rebuild"
    models_left = {r[0] for r in facts_db.execute("SELECT DISTINCT model_id FROM team_state")}
    assert models_left == {RATING_ID}, "foreign-id rows must be deleted"


@pytest.mark.usefixtures("spied")
def test_downstream_strength_non_default_after_reconcile(facts_db):
    """After a rebuild the served strength flows to the downstream as-of read, not 0.0."""
    # Before: no state -> the downstream readout is the 0.0 default.
    before = team_state_asof(facts_db, ["A"], RATING_ID, date(2026, 3, 20))
    assert team_snapshot._latent_team_strength(before, "A") == 0.0

    state_export.ensure_team_state_current(facts_db)

    after = team_state_asof(facts_db, ["A"], RATING_ID, date(2026, 3, 20))
    assert team_snapshot._latent_team_strength(after, "A") == pytest.approx(1.5)
