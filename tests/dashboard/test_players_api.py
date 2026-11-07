"""Player terminal API tests — list filters, overview pillars, draft profile."""

import shutil
import sqlite3
from datetime import date

from fastapi.testclient import TestClient

from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.analytics.player_draft import RADAR_AXES
from league_predictor.dashboard.app_factory import create_app
from tests.dashboard.conftest import make_terminal_client, seed_terminal_data


def test_player_overview_has_team_display_name(tmp_path, db_template_path):
    client = make_terminal_client(tmp_path, db_template_path)
    resp = client.get("/api/players/oe:player:faker/overview")
    assert resp.status_code == 200
    data = resp.json()
    assert "team_display_name" in data
    assert data["team_display_name"] != ""


def test_player_list_and_overview(tmp_path, db_template_path):
    client = make_terminal_client(tmp_path, db_template_path)

    resp = client.get("/api/players?major_only=false&min_games=0")
    assert resp.status_code == 200
    players = resp.json()["players"]
    assert any(player["player_id"] == "oe:player:faker" for player in players)

    resp = client.get("/api/players/oe:player:faker/overview")
    assert resp.status_code == 200
    overview = resp.json()
    assert overview["role"] == "mid"
    # Role-generic pillars[] payload; mid carries the 3-pillar shape.
    assert isinstance(overview["pillars"], list)
    pillar_names = [p["name"] for p in overview["pillars"]]
    assert pillar_names == ["laning", "combat", "vision"]
    # Every pillar entry carries a data_quality annotation so the UI can flag partials.
    assert all("data_quality" in p for p in overview["pillars"])


def test_sup_overview_returns_four_pillars(tmp_path, db_template_path):
    """The support overview returns its four role-specific pillars."""
    client = make_terminal_client(tmp_path, db_template_path)
    resp = client.get("/api/players/oe:player:keria/overview")
    assert resp.status_code == 200
    overview = resp.json()
    assert overview["role"] == "sup"
    names = {p["name"] for p in overview["pillars"]}
    assert names == {"vision", "playmaking", "lane", "utility"}


def test_player_overview_returns_draft_profile_block(tmp_path, db_template_path):
    """The overview's draft_profile mirrors the snapshot columns; the radar has five axes."""
    client = make_terminal_client(tmp_path, db_template_path)
    resp = client.get("/api/players/oe:player:faker/overview")
    assert resp.status_code == 200
    profile = resp.json()["draft_profile"]

    # Shape check: the gameplay anchor is gone — the radar is draft-derived only.
    assert set(profile["axes"]) == {
        "blind_quality",
        "counter_quality",
        "counter_rate",
        "pool_depth",
        "consistency",
    }
    for ax in profile["axes"].values():
        assert "value" in ax
        assert "sample" in ax

    # No pick_score rows seeded → soft axes are NULL but the block still renders.
    assert profile["axes"]["blind_quality"]["value"] is None
    assert profile["axes"]["counter_quality"]["value"] is None
    # Year is parsed from snapshot_date prefix.
    assert profile["year"] == 2026
    # Weights are exposed so the front-end states the blend without its own copy.
    assert profile["weights"] == {
        "blind_quality": 0.30,
        "counter_quality": 0.30,
        "pool_depth": 0.20,
        "consistency": 0.20,
        "counter_rate": 0.0,
    }


def test_draft_profile_axes_come_from_the_one_axis_home(tmp_path, db_template_path):
    """Spoke order, labels and weights are served from ``RADAR_AXES``; the page keeps no copy."""
    client = make_terminal_client(tmp_path, db_template_path)
    profile = client.get("/api/players/oe:player:faker/overview").json()["draft_profile"]

    assert list(profile["axes"]) == [axis.key for axis in RADAR_AXES]
    assert [ax["label"] for ax in profile["axes"].values()] == [ax.label for ax in RADAR_AXES]
    assert profile["weights"] == {axis.key: axis.weight for axis in RADAR_AXES}


def test_player_overview_states_the_impact_qualification_bar(tmp_path, db_template_path):
    """A player with no Impact row serves NULL plus the bar the UI must state."""
    client = make_terminal_client(tmp_path, db_template_path)
    overview = client.get("/api/players/oe:player:faker/overview").json()
    assert overview["impact"] is None
    assert overview["grade"] is None
    assert overview["impact_threshold"] == {"min_games": 25, "window_months": 24}


def test_player_list_excludes_inactive_and_exposes_region(tmp_path, db_template_path):
    """Players with no games in the range drop out; every returned player carries ``region``."""
    db_path = tmp_path / "player_activity.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))

    # A retired player who last played in 2024 but still lingers on an active
    # team's snapshot (the WildTurtle case). Seed the map as a facts one-game series.
    from tests.predraft.conftest import insert_game

    insert_game(
        conn,
        game_id="m_old_1",
        day="2024-02-01",
        team1="oe:team:t1",
        team2="oe:team:g2",
        winner="oe:team:t1",
        tournament="LCK 2024",
        game_no=1,
        best_of=3,
        duration_s=1800,
        blue_team="oe:team:t1",
        patch="14.1",
        event_type="regular",
    )
    conn.execute(
        """
        INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, result,
            kills, deaths, assists, teamkills, teamdeaths)
        VALUES ('m_old_1', 'oe:player:wildturtle', 'oe:team:t1', 'Ezreal', 'bot', 1,
            3, 1, 6, 14, 8)
        """
    )
    conn.execute(
        """
        INSERT INTO player_daily_snapshot (player_id, date, team_id, role, display_name,
            champion_pool_json, games_played)
        VALUES ('oe:player:wildturtle', '2026-03-20', 'oe:team:t1', 'bot', 'WildTurtle',
            '{}', 243)
        """
    )
    conn.commit()
    conn.close()

    client = TestClient(create_app(db_path=db_path))
    resp = client.get(
        "/api/players?major_only=false&min_games=0&start_date=2026-01-01&end_date=2026-12-31"
    )
    assert resp.status_code == 200
    players = resp.json()["players"]
    names = {p["display_name"] for p in players}
    assert "WildTurtle" not in names, "retired player must be filtered out"
    assert any(p["player_id"] == "oe:player:faker" for p in players)
    assert all("region" in p for p in players)
