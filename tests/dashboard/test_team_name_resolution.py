"""Canonical team-name resolution across the route joins (Issue 18): newest name wins, once."""

import shutil
import sqlite3
from datetime import date
from urllib.parse import quote

from fastapi.testclient import TestClient

from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.dashboard.app_factory import create_app
from tests.dashboard.conftest import (
    seed_games_fixture,
    seed_terminal_data,
)

RANGE = {"start_date": "2026-01-01", "end_date": "2026-12-31"}
_NEW_NAME = "T1 Rebrand"  # inserted after "T1", so it holds the higher rowid → newest


def _terminal_client_with_renamed_t1(tmp_path, db_template_path) -> TestClient:
    """Terminal marts + a second oe_team_names row for t1 (a rename)."""
    db_path = tmp_path / "renamed_terminal.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))
    conn.execute(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES ('oe:team:t1', ?)",
        (_NEW_NAME,),
    )
    conn.commit()
    conn.close()
    return TestClient(create_app(db_path=db_path))


def test_player_list_no_duplicate_rows_for_renamed_team(tmp_path, db_template_path):
    """A renamed team must not duplicate its players (the LEFT-JOIN fan-out bug)."""
    client = _terminal_client_with_renamed_t1(tmp_path, db_template_path)
    resp = client.get("/api/players", params={**RANGE, "major_only": "false"})
    assert resp.status_code == 200
    players = resp.json()["players"]
    ids = [p["player_id"] for p in players]
    assert len(ids) == len(set(ids)), "renamed team duplicated its players"
    fakers = [p for p in players if p["player_id"] == "oe:player:faker"]
    assert len(fakers) == 1


def test_player_list_uses_newest_team_name(tmp_path, db_template_path):
    client = _terminal_client_with_renamed_t1(tmp_path, db_template_path)
    resp = client.get("/api/players", params={**RANGE, "major_only": "false"})
    faker = next(p for p in resp.json()["players"] if p["player_id"] == "oe:player:faker")
    assert faker["team_display_name"] == _NEW_NAME


def test_player_overview_uses_newest_team_name(tmp_path, db_template_path):
    client = _terminal_client_with_renamed_t1(tmp_path, db_template_path)
    resp = client.get("/api/players/oe:player:faker/overview")
    assert resp.status_code == 200
    assert resp.json()["team_display_name"] == _NEW_NAME


def test_tournament_detail_dedupes_and_uses_newest_team_name(tmp_path, db_template_path):
    """The league sheet resolves one newest name per team — no arbitrary pick, no duplicates."""
    client = _terminal_client_with_renamed_t1(tmp_path, db_template_path)
    resp = client.get(f"/api/tournaments/{quote('LCK')}", params=RANGE)
    assert resp.status_code == 200
    body = resp.json()
    t1_rows = [r for r in body["ranking"] if r["team_id"] == "oe:team:t1"]
    assert len(t1_rows) == 1
    assert t1_rows[0]["name"] == _NEW_NAME


def test_games_list_uses_newest_team_name(tmp_path, db_template_path):
    """Game rows use the canonical newest-name SQL."""
    db_path = tmp_path / "renamed_games.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_games_fixture(conn)
    conn.execute(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES ('oe:team:t1', ?)", (_NEW_NAME,)
    )
    conn.commit()
    conn.close()
    client = TestClient(create_app(db_path=db_path))
    resp = client.get("/api/games", params={**RANGE, "major_only": "false"})
    assert resp.status_code == 200
    names = set()
    for g in resp.json()["games"]:
        names.add(g["blue"]["name"])
        names.add(g["red"]["name"])
    assert _NEW_NAME in names
    assert "T1" not in names  # the stale name must not leak through
