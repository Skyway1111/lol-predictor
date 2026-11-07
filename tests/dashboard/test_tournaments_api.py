"""Leagues list + tournament-detail API tests."""

import sqlite3

from tests.dashboard.conftest import RANGE, _players_for, make_games_client
from tests.predraft.conftest import insert_game


def test_leagues_list(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    data = client.get(f"/api/leagues?{RANGE}").json()
    leagues = {lg["league_id"]: lg for lg in data["leagues"]}
    # Both leagues with games in range appear (dropdown shows all, not just major).
    assert set(leagues) == {"LCK", "LFL"}
    assert leagues["LCK"]["n_games"] == 2
    assert leagues["LCK"]["n_series"] == 2
    assert leagues["LCK"]["is_major"] is True
    assert leagues["LFL"]["is_major"] is False
    assert leagues["LCK"]["name"] == "LoL Champions Korea"
    # Sorted n_games desc: LCK (2) before LFL (1).
    assert [lg["league_id"] for lg in data["leagues"]] == ["LCK", "LFL"]


def test_leagues_list_major_only(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    data = client.get(f"/api/leagues?{RANGE}&major_only=true").json()
    # The tournaments rail passes the major toggle through; LFL drops out.
    assert [lg["league_id"] for lg in data["leagues"]] == ["LCK"]


def test_top_kda_excludes_unnamed_players(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    # Third LCK game so every t1/t2 player clears the >=3-games KDA floor,
    # then strip one player's alias — the raw oe:player id must not surface.
    conn = sqlite3.connect(tmp_path / "games_api.db")
    insert_game(
        conn,
        game_id="g4",
        day="2026-03-14",
        team1="oe:team:t1",
        team2="oe:team:t2",
        winner="oe:team:t1",
        tournament="LCK",
        best_of=3,
        blue_team="oe:team:t1",
        patch="15.2",
    )
    _players_for(conn, "g4", "oe:team:t1", ["Jax", "Sejuani", "Ahri", "Ezreal", "Thresh"], 1)
    _players_for(conn, "g4", "oe:team:t2", ["Ornn", "Maokai", "Azir", "Varus", "Braum"], 0)
    conn.execute("DELETE FROM player_aliases WHERE player_id = 'oe:player:oe:team:t1_mid'")
    conn.commit()
    conn.close()

    kda = client.get(f"/api/tournaments/LCK?{RANGE}").json()["overview"]["top_kda"]
    assert kda, "named players must still rank"
    assert all(not p["name"].startswith("oe:player:") for p in kda)
    assert "oe:player:oe:team:t1_mid" not in {p["player_id"] for p in kda}


def test_tournament_detail(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    d = client.get(f"/api/tournaments/LCK?{RANGE}").json()
    assert d["league"]["league_id"] == "LCK"
    assert d["league"]["is_major"] is True

    ov = d["overview"]
    assert ov["n_games"] == 2
    assert ov["n_series"] == 2
    assert ov["avg_duration_s"] == 1800.0  # games.duration_s (insert_game default)
    # Each game: 14 + 8 = 22 total kills.
    assert ov["avg_kills"] == 22.0
    assert ov["shortest_game"]["duration_s"] == 1800
    assert ov["most_kills_game"]["kills"] == 22
    # Blue won g1 but lost g2 -> blue win rate 0.5.
    assert ov["blue_red"]["win_rate"] == 0.5
    # top_kda needs >=3 games; nobody qualifies in this 2-game fixture.
    assert isinstance(ov["top_kda"], list)

    # Ranking: t1 won both games.
    ranking = {r["team_id"]: r for r in d["ranking"]}
    assert ranking["oe:team:t1"]["wins"] == 2
    assert ranking["oe:team:t1"]["losses"] == 0
    assert ranking["oe:team:t1"]["winrate"] == 1.0
    assert ranking["oe:team:t1"]["avg_gdm"] is not None

    # Picks & bans present; Ahri picked twice.
    pb = {c["champion"]: c for c in d["picks_bans"]}
    assert pb["Ahri"]["picks"] == 2
    assert pb["Zed"]["bans"] == 1

    # Matches: two one-game series, t1 wins each; game1_id links each series
    # to its first game's sheet.
    assert len(d["matches"]) == 2
    for m in d["matches"]:
        assert m["winner_team_id"] == "oe:team:t1"
        assert {m["score_a"], m["score_b"]} == {1, 0}
        assert m["best_of"] == 3
    assert {m["game1_id"] for m in d["matches"]} == {"g1", "g2"}


def test_tournament_empty_range(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    d = client.get("/api/tournaments/LCK?start_date=2019-01-01&end_date=2019-12-31").json()
    assert d["overview"]["n_games"] == 0
    assert d["overview"]["blue_red"]["win_rate"] is None
    assert d["ranking"] == []
    assert d["matches"] == []
