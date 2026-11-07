"""The /api/upcoming endpoint: window, league lenses, names, and paired results."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi.testclient import TestClient

from league_predictor.dashboard.app_factory import create_app
from league_predictor.db.schema import apply_current_schema


@pytest.fixture
def client(tmp_path):
    db = tmp_path / "upcoming_api.db"
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    _seed(conn)
    conn.close()
    return TestClient(create_app(db_path=db))


def _seed(conn: sqlite3.Connection) -> None:
    conn.executemany(
        "INSERT INTO teams (team_id, name) VALUES (?, ?)",
        [("oe:team:aa", "Gen.G"), ("oe:team:bb", "T1"), ("oe:team:cc", "BLG")],
    )
    conn.executemany(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES (?, ?)",
        [("oe:team:aa", "Gen.G"), ("oe:team:bb", "T1"), ("oe:team:cc", "BLG")],
    )
    conn.executemany(
        "INSERT INTO leagues (league_id, name) VALUES (?, ?)",
        [("LCK", "LCK"), ("HLL", "HLL")],
    )
    conn.execute("INSERT INTO models (model_id, kind) VALUES ('served:x', 'served')")
    rows = [
        # matched + predicted LCK match on the 16th
        (
            900,
            "LCK",
            "LCK",
            3,
            "2026-07-16T09:00:00+00:00",
            "not_started",
            "oe:team:aa",
            "oe:team:bb",
            "Gen.G",
            "T1",
            0.71,
            0.64,
            "served:x",
            "2026-07-15T12:00:00+00:00",
            None,
            "2026-07-15T12:00:00+00:00",
        ),
        # unmatched minor-league match, same day
        (
            901,
            "Baltic Masters",
            None,
            3,
            "2026-07-16T15:00:00+00:00",
            "not_started",
            None,
            None,
            "Mystery Org",
            "Other Org",
            None,
            None,
            None,
            None,
            None,
            "2026-07-15T12:00:00+00:00",
        ),
        # HLL (non-major, mapped) match a day later
        (
            902,
            "HLL",
            "HLL",
            3,
            "2026-07-17T15:00:00+00:00",
            "not_started",
            "oe:team:aa",
            "oe:team:cc",
            "Gen.G",
            "BLG",
            0.66,
            0.6,
            "served:x",
            "2026-07-15T12:00:00+00:00",
            None,
            "2026-07-15T12:00:00+00:00",
        ),
        # played + paired LCK match on the 14th
        (
            903,
            "LCK",
            "LCK",
            3,
            "2026-07-14T09:00:00+00:00",
            "finished",
            "oe:team:aa",
            "oe:team:bb",
            "Gen.G",
            "T1",
            0.7,
            0.62,
            "served:x",
            "2026-07-14T08:00:00+00:00",
            "ser_x",
            "2026-07-15T12:00:00+00:00",
        ),
    ]
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts,"
        " best_of, winner, status) VALUES ('ser_x', 'LCK', 'oe:team:aa', 'oe:team:bb',"
        " '2026-07-14T09:05:00+00:00', 3, 'oe:team:bb', 'completed')"
    )
    conn.executemany(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner)"
        " VALUES (?, 'ser_x', ?, 'oe:team:aa', 'oe:team:bb', ?)",
        [("gx1", 1, "oe:team:bb"), ("gx2", 2, "oe:team:aa"), ("gx3", 3, "oe:team:bb")],
    )
    conn.executemany(
        "INSERT INTO upcoming_matches (ps_match_id, league_name, league_id, best_of, scheduled_ts,"
        " ps_status, team_a, team_b, team_a_name, team_b_name, p_a, p_a_map, model_id,"
        " predicted_at, series_id, fetched_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        rows,
    )
    # a running Bo5 on the 18th, T1 (team_b) up 2-0, priced for its next map
    conn.execute(
        "INSERT INTO upcoming_matches (ps_match_id, league_name, league_id, best_of, scheduled_ts,"
        " ps_status, team_a, team_b, team_a_name, team_b_name, score_a, score_b, p_a, p_a_map,"
        " model_id, predicted_at, fetched_at) VALUES (904, 'LCK', 'LCK', 5,"
        " '2026-07-18T09:00:00+00:00', 'running', 'oe:team:aa', 'oe:team:bb', 'Gen.G', 'T1',"
        " 0, 2, 0.12, 0.48, 'served:x', '2026-07-18T10:30:00+00:00', '2026-07-18T10:30:00+00:00')"
    )
    conn.commit()


def _get(client, query: str) -> list[dict]:
    resp = client.get(f"/api/upcoming?{query}")
    assert resp.status_code == 200
    return resp.json()["matches"]


def test_window_returns_matches_ascending_with_names_and_predictions(client):
    matches = _get(client, "start_date=2026-07-16&end_date=2026-07-17&major_only=false")
    assert [m["ps_match_id"] for m in matches] == [900, 901, 902]
    lck = matches[0]
    assert lck["team_a"] == {"team_id": "oe:team:aa", "name": "Gen.G"}
    assert lck["team_b"] == {"team_id": "oe:team:bb", "name": "T1"}
    assert lck["p_a"] == 0.71
    assert lck["best_of"] == 3
    assert lck["league_id"] == "LCK"
    unmatched = matches[1]
    assert unmatched["team_a"] == {"team_id": None, "name": "Mystery Org"}
    assert unmatched["p_a"] is None
    assert unmatched["league_name"] == "Baltic Masters"


def test_major_only_keeps_major_league_matches_only(client):
    matches = _get(client, "start_date=2026-07-16&end_date=2026-07-17&major_only=true")
    assert [m["ps_match_id"] for m in matches] == [900]


def test_explicit_league_selection_wins(client):
    matches = _get(client, "start_date=2026-07-16&end_date=2026-07-17&leagues=HLL&major_only=true")
    assert [m["ps_match_id"] for m in matches] == [902]


def test_paired_match_carries_the_realized_result(client):
    matches = _get(client, "start_date=2026-07-14&end_date=2026-07-14&major_only=true")
    (played,) = matches
    assert played["series_id"] == "ser_x"
    assert played["result"] == {"winner_team_id": "oe:team:bb", "score_a": 1, "score_b": 2}


def test_unpaired_match_has_no_result(client):
    matches = _get(client, "start_date=2026-07-16&end_date=2026-07-16&major_only=true")
    assert matches[0]["result"] is None
    assert matches[0]["score"] is None  # not started: no maps won yet


def test_running_match_carries_its_live_score(client):
    """A running row exposes the feed's maps won; its prediction is for the next map."""
    (live,) = _get(client, "start_date=2026-07-18&end_date=2026-07-18&major_only=true")
    assert live["ps_status"] == "running"
    assert live["score"] == [0, 2]
    assert live["result"] is None
    assert live["p_a"] == 0.12
