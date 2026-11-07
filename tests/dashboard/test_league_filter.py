"""League multiselect scoping for the team/player list endpoints (Issue 17)."""

import shutil
import sqlite3
from datetime import date

from fastapi.testclient import TestClient

from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.dashboard.app_factory import create_app
from tests.dashboard.conftest import make_terminal_client, seed_terminal_data
from tests.predraft.conftest import insert_game

LCK = "LCK"
LEC = "LEC"
RANGE = {"start_date": "2026-01-01", "end_date": "2026-12-31"}


def test_team_list_scopes_to_selected_league(tmp_path, db_template_path):
    client = make_terminal_client(tmp_path, db_template_path)
    resp = client.get("/api/teams", params={**RANGE, "major_only": "false", "leagues": LEC})
    assert resp.status_code == 200
    ids = {t["team_id"] for t in resp.json()["teams"]}
    # g2 + fnc play LEC; t1 + hle never do.
    assert {"oe:team:g2", "oe:team:fnc"} <= ids
    assert "oe:team:t1" not in ids
    assert "oe:team:hle" not in ids


def test_team_league_filter_is_not_silently_ignored(tmp_path, db_template_path):
    """LCK vs LEC yield different team sets; a dropped param returns the same list for both."""
    client = make_terminal_client(tmp_path, db_template_path)
    lck = {
        t["team_id"]
        for t in client.get(
            "/api/teams", params={**RANGE, "major_only": "false", "leagues": LCK}
        ).json()["teams"]
    }
    lec = {
        t["team_id"]
        for t in client.get(
            "/api/teams", params={**RANGE, "major_only": "false", "leagues": LEC}
        ).json()["teams"]
    }
    assert lck != lec
    assert "oe:team:hle" in lck
    assert "oe:team:hle" not in lec


def test_player_list_scopes_to_selected_league(tmp_path, db_template_path):
    client = make_terminal_client(tmp_path, db_template_path)
    resp = client.get("/api/players", params={**RANGE, "major_only": "false", "leagues": LEC})
    assert resp.status_code == 200
    ids = {p["player_id"] for p in resp.json()["players"]}
    assert "oe:player:caps" in ids  # G2 mid, played LEC
    assert "oe:player:noah" in ids  # FNC bot, played LEC
    assert "oe:player:faker" not in ids  # T1, LCK only


def test_player_league_filter_is_not_silently_ignored(tmp_path, db_template_path):
    client = make_terminal_client(tmp_path, db_template_path)
    lck = {
        p["player_id"]
        for p in client.get(
            "/api/players", params={**RANGE, "major_only": "false", "leagues": LCK}
        ).json()["players"]
    }
    lec = {
        p["player_id"]
        for p in client.get(
            "/api/players", params={**RANGE, "major_only": "false", "leagues": LEC}
        ).json()["players"]
    }
    assert lck != lec
    assert "oe:player:faker" in lck
    assert "oe:player:faker" not in lec


def test_unknown_league_scopes_to_empty(tmp_path, db_template_path):
    """An unknown league id filters everything out — never a silent full list."""
    client = make_terminal_client(tmp_path, db_template_path)
    teams = client.get(
        "/api/teams", params={**RANGE, "major_only": "false", "leagues": "NO_SUCH_LEAGUE"}
    )
    assert teams.status_code == 200
    assert teams.json()["teams"] == []
    players = client.get(
        "/api/players", params={**RANGE, "major_only": "false", "leagues": "NO_SUCH_LEAGUE"}
    )
    assert players.status_code == 200
    assert players.json()["players"] == []


def test_partial_unknown_league_csv_keeps_the_valid_one(tmp_path, db_template_path):
    client = make_terminal_client(tmp_path, db_template_path)
    resp = client.get(
        "/api/teams", params={**RANGE, "major_only": "false", "leagues": f"{LEC},NOPE"}
    )
    ids = {t["team_id"] for t in resp.json()["teams"]}
    assert {"oe:team:g2", "oe:team:fnc"} <= ids
    assert "oe:team:t1" not in ids


def test_explicit_minor_league_wins_over_major_only(tmp_path, db_template_path):
    """An explicit ``leagues`` selection silences the major lens, so major_only=true keeps it."""
    db_path = tmp_path / "minor_league.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    # A genuinely non-major league (no major-code substring) so the major lens
    # would otherwise drop its team.
    insert_game(
        conn,
        game_id="m_lfl2_1",
        day="2026-03-05",
        team1="oe:team:kc",
        team2="oe:team:sk",
        winner="oe:team:kc",
        tournament="LFL2 Spring",
        game_no=1,
        best_of=3,
        duration_s=1900,
        blue_team="oe:team:kc",
        patch="15.4",
        event_type="regular",
    )
    for tid, roles in (
        ("oe:team:kc", ("top", "jng", "mid", "bot", "sup")),
        ("oe:team:sk", ("top", "jng", "mid", "bot", "sup")),
    ):
        for role in roles:
            conn.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, "
                "result, kills, deaths, assists, teamkills, teamdeaths) VALUES "
                "('m_lfl2_1', ?, ?, 'Ahri', ?, ?, 3, 2, 5, 12, 10)",
                (f"oe:player:{tid[-2:]}_{role}", tid, role, 1 if tid == "oe:team:kc" else 0),
            )
    conn.executemany(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES (?, ?)",
        [("oe:team:kc", "Karmine Corp Blue"), ("oe:team:sk", "SK Prime")],
    )
    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))
    conn.close()
    client = TestClient(create_app(db_path=db_path))

    # Baseline: no league selection + major_only drops the minor-league team.
    major = {
        t["team_id"]
        for t in client.get("/api/teams", params={**RANGE, "major_only": "true"}).json()["teams"]
    }
    assert "oe:team:kc" not in major

    # Explicit selection of that minor league keeps it despite major_only=true.
    scoped = {
        t["team_id"]
        for t in client.get(
            "/api/teams", params={**RANGE, "major_only": "true", "leagues": "LFL2 Spring"}
        ).json()["teams"]
    }
    assert "oe:team:kc" in scoped
    assert "oe:team:sk" in scoped
    assert "oe:team:t1" not in scoped  # scoping still excludes other leagues
