"""Each month's refit cuts at its own start, and the walk covers served's game set exactly."""

from __future__ import annotations

from datetime import date

import pytest

from league_predictor.draft_grading import oos_walk
from league_predictor.draft_grading.grader.predictor import DraftScore
from tests.draft_grading.grader._facts_helpers import BLUE_COMP, RED_COMP, _game


def _corpus(tmp_path):
    """A file-backed corpus: five games a month over April..June 2026, one in July (unseen)."""
    import sqlite3

    from tests.draft_grading.grader.conftest import GRADER_TEST_SCHEMA

    path = tmp_path / "corpus.db"
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(GRADER_TEST_SCHEMA)
    for month in (4, 5, 6):
        for i in range(5):
            day = f"2026-{month:02d}-{1 + i:02d}"
            _game(db, f"g{month}_{i}", day, "LCK", BLUE_COMP, RED_COMP, i % 2 == 0)
    blue_won = True
    _game(db, "late", "2026-07-03", "LCK", BLUE_COMP, RED_COMP, blue_won)
    # A nine-player game: finished, but no draft path can load it, so no walk scores it.
    _game(db, "torn", "2026-05-20", "LCK", BLUE_COMP, RED_COMP, blue_won)
    db.execute(
        "DELETE FROM game_players WHERE game_id = 'torn' AND role = 'sup' AND team_id = 'red'"
    )
    db.commit()
    return db


def _source_is_the_artifact(monkeypatch) -> None:
    """The one state a walk is legal in: source and the artifact stamp are one recipe."""
    monkeypatch.setattr(oos_walk, "assert_artifact_is_source", lambda: None)


@pytest.mark.parametrize(
    ("stamped", "raises"),
    [
        ({"version": 7, "kind": "set"}, False),
        ({"version": 6, "kind": "set"}, True),
        ({"version": 6}, True),
        (None, False),
    ],
)
def test_the_walk_refuses_a_source_that_is_not_the_artifact_s_definition(
    stamped, raises, monkeypatch
):
    """Refitting source would write `oos` rows under another model's id."""
    from league_predictor.draft_grading import recipe as draft_recipe

    monkeypatch.setattr(oos_walk.artifacts, "persisted_recipe", lambda _t: stamped)
    monkeypatch.setattr(draft_recipe, "DRAFT_RECIPE", draft_recipe.DRAFT_RECIPE)
    if raises:
        with pytest.raises(RuntimeError, match="retrain draft"):
            oos_walk.assert_artifact_is_source()
    else:
        oos_walk.assert_artifact_is_source()


class _FakeGrader:
    pro_excess_baseline = "team_form"

    def __init__(self, cutoff):
        self.cutoff = cutoff

    def predict_batch(self, drafts, stats_cache=None):
        del stats_cache
        return [
            DraftScore(prob_blue_wins=0.6, grade_blue="A", grade_red="D", edge_pp_blue=4.5)
            for _ in drafts
        ]


def test_each_month_refits_at_its_own_start_and_the_walk_covers_the_expected_set(
    tmp_path, monkeypatch
):
    conn = _corpus(tmp_path)
    fits: list[tuple[date, str]] = []

    def fake_fit(fit_conn, root, *, fit_through, production_probs, code_fingerprint):
        del production_probs, code_fingerprint
        # The worker's connection is read-only: a refit can never write the corpus.
        with pytest.raises(Exception, match=r"readonly|read-only"):
            fit_conn.execute("DELETE FROM games")
        later = fit_conn.execute(
            "SELECT COUNT(*) FROM games g JOIN series s ON s.series_id = g.series_id "
            "WHERE date(s.scheduled_ts) >= ?",
            (fit_through.isoformat(),),
        ).fetchone()[0]
        fits.append((fit_through, str(root)))
        root.write_text("{}", encoding="utf-8")
        assert later > 0  # the games the refit must NOT see exist in the corpus...

    monkeypatch.setattr(oos_walk, "fit_candidate", fake_fit)
    monkeypatch.setattr(oos_walk, "served_oos_probs", lambda _conn, _sid: {})
    monkeypatch.setattr(oos_walk, "load_grader", lambda _conn, *, artifact: _FakeGrader(artifact))
    monkeypatch.setattr(oos_walk, "walk_worker_count", lambda _n: 1)
    _source_is_the_artifact(monkeypatch)

    through = date(2026, 7, 1)
    scores = oos_walk.walk_draft_oos(
        conn, through=through, served_id="served:x", code_fingerprint="f" * 8, months=4
    )
    # ...and the dataset the real fit builds cuts at `outcome_until = fit_through`,
    # exclusive (tests/draft_grading/test_training_candidate.py pins that cut).
    assert [f[0] for f in fits] == [date(2026, 4, 1), date(2026, 5, 1), date(2026, 6, 1)]
    assert {s.game_id for s in scores} == oos_walk.expected_oos_game_ids(
        conn, through=through, months=4
    )
    assert len(scores) == 15
    assert {"late", "torn"}.isdisjoint(s.game_id for s in scores)
    assert scores[0].edge_pp_blue == 4.5


def test_a_month_that_loses_games_refuses_the_write(tmp_path, monkeypatch):
    conn = _corpus(tmp_path)
    real_score_window = oos_walk._score_window

    def dropping_score_window(predictor, score_conn, *, since, until):
        return real_score_window(predictor, score_conn, since=since, until=until)[:-1]

    def fake_fit(fit_conn, root, **_kwargs):
        del fit_conn
        root.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(oos_walk, "fit_candidate", fake_fit)
    monkeypatch.setattr(oos_walk, "served_oos_probs", lambda _conn, _sid: {})
    monkeypatch.setattr(oos_walk, "load_grader", lambda _conn, *, artifact: _FakeGrader(artifact))
    monkeypatch.setattr(oos_walk, "_score_window", dropping_score_window)
    monkeypatch.setattr(oos_walk, "walk_worker_count", lambda _n: 1)
    _source_is_the_artifact(monkeypatch)
    with pytest.raises(RuntimeError, match="partial oos write"):
        oos_walk.walk_draft_oos(
            conn, through=date(2026, 7, 1), served_id="s", code_fingerprint="f" * 8, months=4
        )


def test_the_walk_refuses_a_card_that_cannot_hold_one_refit(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (int(1.0e9), int(16e9)))
    with pytest.raises(RuntimeError, match="GPU memory"):
        oos_walk.walk_worker_count(12)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda: (int(3.5 * oos_walk.REFIT_VRAM_GB * 1e9), int(16e9))
    )
    monkeypatch.setattr(oos_walk.resources, "bounded_workers", lambda _gb, *, reserve_gb: 6)  # noqa: ARG005 - mirrors bounded_workers' signature
    assert oos_walk.walk_worker_count(12) == 3
    assert oos_walk.walk_worker_count(2) == 2
