"""Predict endpoint tests — the wiring contract only, over stubbed (never real) model handles."""

import threading
from datetime import date

import pytest

from league_predictor import clock
from league_predictor.asof import MatchContext
from league_predictor.dashboard.model_serving import DraftScorer, SeriesPrediction, ServedOracle
from tests.dashboard.conftest import make_games_client


class _FakeScore:
    prob_blue_wins = 0.61
    grade_blue = "A"
    grade_red = "C"
    edge_pp_blue = 4.5


# ── matchup route ────────────────────────────────────────────────────────


def test_matchup_warming(tmp_path, db_template_path, monkeypatch):
    monkeypatch.setattr(ServedOracle, "predict_series", lambda *_, **__: None)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/matchup", json={"team_a_id": "oe:team:t1", "team_b_id": "oe:team:t2"}
    ).json()
    assert body["available"] is False
    assert body["status"] == "warming"
    assert body["p_a"] is None
    assert body["note"]


def test_matchup_error_state(tmp_path, db_template_path, monkeypatch):
    monkeypatch.setattr(ServedOracle, "predict_series", lambda *_, **__: None)
    monkeypatch.setattr(ServedOracle, "status", property(lambda _self: "error"))
    monkeypatch.setattr(ServedOracle, "error", property(lambda _self: "boom"))
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/matchup", json={"team_a_id": "oe:team:t1", "team_b_id": "oe:team:t2"}
    ).json()
    assert body["available"] is False
    assert body["status"] == "error"
    assert body["note"] == "boom"


def test_matchup_ready(tmp_path, db_template_path, monkeypatch):
    seen = {}

    def fake_predict(_self, team_a, team_b, as_of, *, context=None):
        seen["args"] = (team_a, team_b, as_of, context)
        return SeriesPrediction(p_map=0.60, p_bo3=0.65, p_bo5=0.68, as_of=as_of)

    monkeypatch.setattr(ServedOracle, "predict_series", fake_predict)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/matchup",
        json={"team_a_id": "oe:team:t1", "team_b_id": "oe:team:t2", "as_of": "2026-03-06"},
    ).json()
    assert seen["args"] == ("oe:team:t1", "oe:team:t2", date(2026, 3, 6), None)
    assert body["available"] is True
    assert body["as_of"] == "2026-03-06"
    assert (body["p_a"], body["p_a_bo3"], body["p_a_bo5"]) == (0.60, 0.65, 0.68)
    assert body["score"] == [0, 0]


def test_matchup_with_a_series_score_prices_the_next_map(tmp_path, db_template_path, monkeypatch):
    """A format plus maps won reaches the oracle as the match context, and echoes back."""
    seen = {}

    def fake_predict(_self, team_a, team_b, as_of, *, context=None):
        seen["context"] = context
        state = (context.wins_for(team_a), context.wins_for(team_b))
        return SeriesPrediction(p_map=0.55, p_bo3=1.0, p_bo5=0.9, as_of=as_of, state=state)

    monkeypatch.setattr(ServedOracle, "predict_series", fake_predict)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/matchup",
        json={
            "team_a_id": "oe:team:t1",
            "team_b_id": "oe:team:t2",
            "best_of": 5,
            "score_a": 0,
            "score_b": 2,
            "stage": "Playoffs",
        },
    ).json()
    assert seen["context"] == MatchContext.of("Bo5", "Playoffs", {"oe:team:t1": 0, "oe:team:t2": 2})
    assert body["score"] == [0, 2]
    assert body["p_a_bo5"] == 0.9
    # A score above any format's reach is refused before the oracle.
    bad = client.post(
        "/api/predict/matchup",
        json={"team_a_id": "a", "team_b_id": "b", "best_of": 3, "score_a": 3},
    )
    assert bad.status_code == 422


def test_matchup_defaults_to_today(tmp_path, db_template_path, monkeypatch):
    seen = {}

    def fake_predict(_self, _team_a, _team_b, as_of, **_kwargs):
        seen["as_of"] = as_of
        return SeriesPrediction(p_map=0.5, p_bo3=0.5, p_bo5=0.5, as_of=as_of)

    monkeypatch.setattr(ServedOracle, "predict_series", fake_predict)
    client = make_games_client(tmp_path, db_template_path)
    client.post("/api/predict/matchup", json={"team_a_id": "a", "team_b_id": "b"})
    assert seen["as_of"] == clock.utc_today()


def test_matchup_validation(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    # Missing required field -> 422 (never reaches the oracle).
    assert client.post("/api/predict/matchup", json={"team_a_id": "x"}).status_code == 422


# ── draft route ──────────────────────────────────────────────────────────


def test_draft_scored(tmp_path, db_template_path, monkeypatch):
    seen = {}

    def fake_score(_self, blue_picks, red_picks, blue_bans, red_bans):
        seen["args"] = (blue_picks, red_picks, blue_bans, red_bans)
        return _FakeScore()

    monkeypatch.setattr(DraftScorer, "score", fake_score)
    monkeypatch.setattr(
        DraftScorer,
        "assigned_roles",
        lambda *_: {
            "blue": {"Ahri": "mid", "Lee Sin": "jng"},
            "red": {"Orianna": "mid"},
        },
    )
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/draft",
        json={
            "blue": {"picks": ["Ahri", "Lee Sin"]},
            "red": {"picks": ["Orianna"]},
            "bans": {"blue": ["Zed"], "red": []},
        },
    ).json()
    assert seen["args"] == (["Ahri", "Lee Sin"], ["Orianna"], ["Zed"], [])
    assert body["available"] is True
    assert body["p_blue"] == pytest.approx(0.61)
    assert (body["grade_blue"], body["grade_red"]) == ("A", "C")
    assert body["edge_pp_blue"] == pytest.approx(4.5)
    assert body["roles"]["blue"] == {"Ahri": "mid", "Lee Sin": "jng"}


def test_draft_illegal_draft_is_noted(tmp_path, db_template_path, monkeypatch):
    def raising_score(*_):
        raise ValueError("champion used twice: Ahri")

    monkeypatch.setattr(DraftScorer, "score", raising_score)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post("/api/predict/draft", json={"blue": {}, "red": {}}).json()
    assert body["available"] is False
    assert "Ahri" in body["note"]
    assert body["p_blue"] is None
    assert body["grade_blue"] is None
    assert body["grade_red"] is None


def test_draft_minimal_shape(tmp_path, db_template_path, monkeypatch):
    monkeypatch.setattr(DraftScorer, "score", lambda *_: _FakeScore())
    client = make_games_client(tmp_path, db_template_path)
    # Picks/bans optional; only blue/red required.
    assert client.post("/api/predict/draft", json={"blue": {}, "red": {}}).status_code == 200


# ── draft suggest route ──────────────────────────────────────────────────


def test_draft_suggest_route(tmp_path, db_template_path, monkeypatch):
    seen = {}

    def fake_suggest(_self, blue_picks, red_picks, blue_bans, red_bans, side, top_n=5):
        seen["args"] = (blue_picks, red_picks, blue_bans, red_bans, side, top_n)
        return _FakeScore(), [{"champion": "Ahri", "role": "mid", "p_blue": 0.63}]

    monkeypatch.setattr(DraftScorer, "suggest", fake_suggest)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/draft/suggest",
        json={
            "blue": {"picks": ["Aatrox"]},
            "red": {"picks": []},
            "bans": {"blue": ["Zed"], "red": []},
            "side": "red",
        },
    ).json()
    assert seen["args"] == (["Aatrox"], [], ["Zed"], [], "red", 5)
    assert body["available"] is True
    assert body["side"] == "red"
    assert body["p_blue"] == pytest.approx(0.61)
    assert body["suggestions"] == [{"champion": "Ahri", "role": "mid", "p_blue": 0.63}]


def test_draft_suggest_rollout_route(tmp_path, db_template_path, monkeypatch):
    seen = {}

    def fake_rollouts(_self, blue_picks, red_picks, blue_bans, red_bans, side, champions):
        seen["args"] = (blue_picks, red_picks, blue_bans, red_bans, side, champions)
        return [0.508, 0.492]

    monkeypatch.setattr(DraftScorer, "rollout_values", fake_rollouts)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/draft/suggest/rollout",
        json={
            "blue": {"picks": []},
            "red": {"picks": ["Aatrox"]},
            "bans": {"blue": ["Zed"], "red": []},
            "side": "blue",
            "champions": ["Maokai", "Bard"],
        },
    ).json()
    assert seen["args"] == ([], ["Aatrox"], ["Zed"], [], "blue", ["Maokai", "Bard"])
    assert body["available"] is True
    assert [v["champion"] for v in body["values"]] == ["Maokai", "Bard"]
    assert body["values"][0]["p_blue_end"] == pytest.approx(0.508)
    assert body["values"][1]["p_blue_end"] == pytest.approx(0.492)


def test_draft_suggest_rollout_illegal_candidate_is_noted(tmp_path, db_template_path, monkeypatch):
    def raising_rollouts(*_, **__):
        raise ValueError("champion used twice: Aatrox")

    monkeypatch.setattr(DraftScorer, "rollout_values", raising_rollouts)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post(
        "/api/predict/draft/suggest/rollout",
        json={"blue": {}, "red": {}, "side": "blue", "champions": ["Aatrox"]},
    ).json()
    assert body["available"] is False
    assert "Aatrox" in body["note"]
    assert body["values"] == []


def test_draft_suggest_illegal_draft_is_noted(tmp_path, db_template_path, monkeypatch):
    def raising_suggest(*_, **__):
        raise ValueError("champion used twice: Ahri")

    monkeypatch.setattr(DraftScorer, "suggest", raising_suggest)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post("/api/predict/draft/suggest", json={"blue": {}, "red": {}}).json()
    assert body["available"] is False
    assert "Ahri" in body["note"]
    assert body["suggestions"] == []


def test_draft_routes_surface_an_artifact_failure(tmp_path, db_template_path, monkeypatch):
    """Artifact failures reach the operator with the remedy needed to restore serving."""

    def raising(*_, **__):
        raise RuntimeError("draft artifact has no set weights; retrain draft")

    monkeypatch.setattr(DraftScorer, "suggest", raising)
    client = make_games_client(tmp_path, db_template_path)
    body = client.post("/api/predict/draft/suggest", json={"blue": {}, "red": {}}).json()
    assert body["available"] is False
    assert "RuntimeError" in body["note"]
    assert "retrain draft" in body["note"]


# ── DraftScorer handle: illegal drafts rejected before any load ──────────


def test_handle_rejects_illegal_drafts_before_load():
    # db_path=None: any load attempt would crash, so a raised ValueError proves the
    # handle validates the request before touching the artifact or the DB.
    scorer = DraftScorer(db_path=None)
    with pytest.raises(ValueError, match="more than 5"):
        scorer.score(["A", "B", "C", "D", "E", "F"], [], [], [])
    with pytest.raises(ValueError, match="used twice"):
        scorer.suggest(["Ahri"], ["Ahri"], [], [], side="blue")
    # An illegal CANDIDATE is also caught pre-load, not just an illegal board.
    with pytest.raises(ValueError, match="used twice"):
        scorer.rollout_values(["Ahri"], [], [], [], side="red", champions=["Ahri"])


def test_handle_suggest_is_always_the_fast_path(monkeypatch):
    # The handle must never hold its lock for a whole board's rollout (~20s), so it pins
    # rollout=False on every engine call; the blocking mode is an engine-only concern.
    seen = {}

    class _RecordingEngine:
        def suggest(self, *_args, **kwargs):
            seen.update(kwargs)
            return object(), []

    engine = _RecordingEngine()
    monkeypatch.setattr(DraftScorer, "_loaded_engine", lambda _self: engine)
    DraftScorer(db_path=None).suggest([], [], [], [], side="blue")
    assert seen["rollout"] is False


# ── the draft-slot dev seam ──────────────────────────────────────────────


# ── ServedOracle state machine (build stubbed) ───────────────────────────


def test_oracle_kicks_off_one_warmup_and_reports_warming(monkeypatch):
    calls = []
    started: list[threading.Thread] = []
    real_start = threading.Thread.start

    def _record_start(thread):
        started.append(thread)
        real_start(thread)

    monkeypatch.setattr(threading.Thread, "start", _record_start)
    monkeypatch.setattr(ServedOracle, "_build", lambda _self: calls.append(1))
    oracle = ServedOracle(db_path=None)
    assert oracle.predict_series("a", "b", clock.utc_today()) is None
    assert oracle.status == "warming"
    # Subsequent calls while warming do not start another build.
    assert oracle.predict_series("a", "b", clock.utc_today()) is None
    assert len(started) == 1
    started[0].join(timeout=5)
    assert len(calls) == 1
