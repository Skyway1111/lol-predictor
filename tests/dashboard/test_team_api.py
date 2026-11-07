"""Team terminal API tests — list/overview/roster/rating-history over seeded marts."""

import shutil
import sqlite3

import pytest
from fastapi.testclient import TestClient

from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.dashboard.app_factory import create_app
from tests.dashboard.conftest import _TERMINAL_AS_OF, make_terminal_client, seed_terminal_data
from tests.predraft.conftest import insert_game


def test_teams_list_gates_on_recent_activity(tmp_path, db_template_path):
    """Rankings anchor on the snapshot date: a team outside the activity window never ranks."""
    db = tmp_path / "terminal_api.db"
    shutil.copy(db_template_path, db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    # A team whose only game is ~26 months before the snapshot (2026-03-20).
    insert_game(
        conn,
        game_id="g_dead",
        day="2024-01-10",
        team1="oe:team:dead",
        team2="oe:team:t1",
        winner="oe:team:t1",
        tournament="LCK",
        blue_team="oe:team:dead",
    )
    conn.execute(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES ('oe:team:dead', 'Old Sponsor Org')"
    )
    # A recent scheduled-but-gameless series (upcoming-sync residue) must not
    # resurrect the dead id: activity means played games, not series seats.
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, best_of, "
        "winner, status) VALUES ('ser_ghost', 'LCK', 'oe:team:dead', 'oe:team:t1', "
        "'2026-03-15T12:00:00+00:00', 3, NULL, 'completed')"
    )
    refresh_analytics_marts(conn, as_of_date=_TERMINAL_AS_OF)
    conn.close()
    client = TestClient(create_app(db_path=db))

    data = client.get(
        "/api/teams?major_only=false&start_date=2024-01-01&end_date=2026-12-31"
    ).json()
    ids = {t["team_id"] for t in data["teams"]}
    assert "oe:team:t1" in ids
    assert "oe:team:dead" not in ids


def test_team_endpoints_return_terminal_data(tmp_path, db_template_path):
    client = make_terminal_client(tmp_path, db_template_path)

    resp = client.get("/api/teams?major_only=false&min_games=0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["as_of_date"] == "2026-03-20"
    assert any(team["team_id"] == "oe:team:t1" for team in data["teams"])

    resp = client.get("/api/teams/oe:team:t1/overview")
    assert resp.status_code == 200
    overview = resp.json()
    assert overview["display_name"] == "T1"
    # power.rating is the skill-filter-derived team power.
    assert isinstance(overview["power"]["rating"], float)

    resp = client.get("/api/teams/oe:team:t1/roster")
    assert resp.status_code == 200
    roster = resp.json()
    assert len(roster["players"]) == 5
    assert {player["role"] for player in roster["players"]} == {"top", "jng", "mid", "bot", "sup"}
    # Roster surfaces live Impact per player (as-of the snapshot date), matching /api/players.
    for player in roster["players"]:
        assert "impact" in player

    # Power-rating trace for the team-page sparkline, oldest-first. days=365 keeps the fixed
    # 2026-03-20 mart snapshot inside the window relative to "now".
    resp = client.get("/api/teams/oe:team:t1/rating-history?days=365")
    assert resp.status_code == 200
    history = resp.json()["history"]
    assert history, "expected at least the seeded snapshot"
    assert history[0]["date"] == "2026-03-20"
    assert history[0]["rating"] is not None
    assert [h["date"] for h in history] == sorted(h["date"] for h in history)


def test_team_overview_carries_draft_utilization_from_the_stack(tmp_path, db_template_path):
    """The team page reads beta + u and its oos count straight from postdraft_stack."""
    db = tmp_path / "terminal_api.db"
    shutil.copy(db_template_path, db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    conn.executemany(
        "INSERT INTO postdraft_stack (team_id, a, beta, u, n_oos, served_model_id, draft_model_id) "
        "VALUES (?, ?, ?, ?, ?, 'served:abc', 'draft:test')",
        [("*", 1.1, 0.9, 0.0, 40), ("oe:team:t1", None, None, 0.4, 12)],
    )
    refresh_analytics_marts(conn, as_of_date=_TERMINAL_AS_OF)
    conn.close()
    client = TestClient(create_app(db_path=db))

    util = client.get("/api/teams/oe:team:t1/overview").json()["draft_utilization"]
    assert util == {"value": pytest.approx(1.3), "n_oos": 12}
    # A team the fit never saw shows no number rather than the global coefficient.
    conn = sqlite3.connect(str(db))
    conn.execute("DELETE FROM postdraft_stack WHERE team_id = 'oe:team:t1'")
    conn.commit()
    conn.close()
    assert client.get("/api/teams/oe:team:t1/overview").json()["draft_utilization"] is None


def test_team_overview_utilization_is_null_without_a_stack(tmp_path, db_template_path):
    """An unfitted stack leaves the field null; the page dashes it, never defaults it."""
    client = make_terminal_client(tmp_path, db_template_path)
    assert client.get("/api/teams/oe:team:t1/overview").json()["draft_utilization"] is None
