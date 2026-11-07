"""Games browser + game-detail API tests over the shared games fixture."""

import sqlite3

from league_predictor.analytics.team_strength_display import latent_strength_to_display
from tests.dashboard.conftest import _ROLES, RANGE, make_games_client
from tests.predraft.conftest import insert_game


def test_games_list_major_filter_and_pagination(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    # Default major_only drops the LFL game; two LCK games remain.
    data = client.get(f"/api/games?{RANGE}").json()
    assert data["total"] == 2
    ids = {g["game_id"] for g in data["games"]}
    assert ids == {"g1", "g2"}

    # Explicit league selection overrides major_only.
    lfl = client.get(f"/api/games?{RANGE}&leagues=LFL").json()
    assert lfl["total"] == 1
    assert lfl["games"][0]["game_id"] == "g3"

    # major_only=false + no leagues => all three.
    allg = client.get(f"/api/games?{RANGE}&major_only=false").json()
    assert allg["total"] == 3

    # Pagination: total stays full, page is capped.
    page = client.get(f"/api/games?{RANGE}&major_only=false&limit=1&offset=0").json()
    assert page["total"] == 3
    assert len(page["games"]) == 1


def test_games_list_ordering_and_dates(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    games = client.get(f"/api/games?{RANGE}").json()["games"]
    # scheduled_ts DESC: g2 (03-12) before g1 (03-10).
    assert [g["game_id"] for g in games] == ["g2", "g1"]
    assert games[1]["date"] == "2026-03-10"


def test_games_as_of_team_strength_has_no_future_leak(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    g1 = next(g for g in client.get(f"/api/games?{RANGE}").json()["games"] if g["game_id"] == "g1")
    # Earlier state (strength 0.5) used; the later 9.9 state must not leak.
    assert g1["rating_blue"] == latent_strength_to_display(0.5)
    assert g1["rating_red"] == latent_strength_to_display(0.0)
    assert g1["rating_blue"] != latent_strength_to_display(9.9)


def test_games_prediction_is_oos_only(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    games = {g["game_id"]: g for g in client.get(f"/api/games?{RANGE}").json()["games"]}
    # OOS p_a=0.7 surfaces (blue==team_a); the backfill 0.99 row must not.
    assert games["g1"]["pred_p_blue"] == 0.7
    # No prediction row for g2 -> null (dash).
    assert games["g2"]["pred_p_blue"] is None


def test_games_prediction_live_tail(tmp_path, db_template_path):
    """Past the corpus edge a backfill row surfaces as ``pred_source='live'``; before it, null."""
    client = make_games_client(tmp_path, db_template_path)
    conn = sqlite3.connect(str(tmp_path / "games_api.db"))
    # g0: before the edge, backfill row only -> must stay a dash.
    insert_game(
        conn,
        game_id="g0",
        day="2026-03-01",
        team1="oe:team:t1",
        team2="oe:team:t2",
        winner="oe:team:t1",
        tournament="LCK",
        blue_team="oe:team:t1",
    )
    conn.executemany(
        "INSERT INTO predictions (subject_kind, subject_id, model_id, p_a, asof_date, "
        "context, created_at) VALUES (?,?,?,?,?,?,?)",
        [
            ("game", "g0", "served:abc", 0.9, "2026-03-01", "backfill", "2026-03-20T00:00:00"),
            # g2 (2026-03-12) postdates the edge: its backfill row IS the prediction.
            ("game", "g2", "served:abc", 0.4, "2026-03-12", "backfill", "2026-03-20T00:00:00"),
        ],
    )
    conn.commit()
    conn.close()

    games = {g["game_id"]: g for g in client.get(f"/api/games?{RANGE}").json()["games"]}
    # g1 keeps the strict walk-forward row.
    assert games["g1"]["pred_p_blue"] == 0.7
    assert games["g1"]["pred_source"] == "walk"
    # g2: p_a = P(team_a=t1) = 0.4, blue is t2 -> blue prob 0.6, tagged live.
    assert games["g2"]["pred_p_blue"] == 0.6
    assert games["g2"]["pred_source"] == "live"
    # g0: in-sample only -> null, never dressed up as a prediction.
    assert games["g0"]["pred_p_blue"] is None
    assert games["g0"]["pred_source"] is None

    # Detail mirrors the list.
    assert client.get("/api/games/g2").json()["prediction"] == {
        "p_blue": 0.6,
        "asof_date": "2026-03-12",
        "source": "live",
    }
    assert client.get("/api/games/g1").json()["prediction"]["source"] == "walk"
    assert client.get("/api/games/g0").json()["prediction"] is None


def test_games_postdraft_fusion_surfaces(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    games = {g["game_id"]: g for g in client.get(f"/api/games?{RANGE}").json()["games"]}
    # Stored blue-oriented fusion surfaces for g1; g2 (no row) is null (dash).
    assert games["g1"]["post_p_blue"] == 0.66
    assert games["g2"]["post_p_blue"] is None
    # Detail carries it too.
    assert client.get("/api/games/g1").json()["post_p_blue"] == 0.66
    assert client.get("/api/games/g2").json()["post_p_blue"] is None


def test_games_grades_and_picks(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    games = {g["game_id"]: g for g in client.get(f"/api/games?{RANGE}").json()["games"]}
    # Draft grades from draft_scores (g1 present, g2 absent -> null).
    assert games["g1"]["grade_blue"] == "A"
    assert games["g1"]["grade_red"] == "C"
    assert games["g2"]["grade_blue"] is None
    # Calibrated edge rides the list payload; absent rows are null (dash-safe).
    assert games["g1"]["edge_pp_blue"] == 4.2
    assert games["g2"]["edge_pp_blue"] is None
    # g1 picks come from draft order; g2 from the game_players fallback.
    assert "Ahri" in games["g1"]["blue"]["picks"]
    assert set(games["g2"]["blue"]["picks"]) == {"Ornn", "Maokai", "Azir", "Varus", "Braum"}
    assert games["g1"]["blue"]["won"] is True
    assert games["g2"]["blue"]["won"] is False  # t1 (red) won g2


def test_games_team_matches_scoped(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    data = client.get(f"/api/teams/oe:team:t1/matches?{RANGE}").json()
    assert data["total"] == 2  # t1 played both LCK games
    assert all("oe:team:t1" in (g["blue"]["team_id"], g["red"]["team_id"]) for g in data["games"])


def test_game_detail_shape(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    d = client.get("/api/games/g1").json()
    assert d["game"]["game_id"] == "g1"
    assert d["blue"]["team_id"] == "oe:team:t1"
    assert d["blue"]["rating"] == latent_strength_to_display(0.5)
    assert len(d["blue"]["players"]) == 5
    assert [p["role"] for p in d["blue"]["players"]] == list(_ROLES)
    assert d["blue"]["kills"] == 14
    assert d["blue"]["firstblood"] is True
    assert "Zed" in d["blue"]["bans"]
    assert d["draft"]["grade_blue"] == "A"
    assert d["draft"]["edge_pp_blue"] == 4.2
    assert d["draft"]["picks"][0]["percentile"] == 0.75
    # OOS prediction surfaces in the detail too.
    assert d["prediction"]["p_blue"] == 0.7

    # A game with no draft/prediction -> those blocks are null (dash-safe).
    d2 = client.get("/api/games/g2").json()
    assert d2["draft"] is None
    assert d2["prediction"] is None


def test_game_detail_uses_the_newest_player_alias(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    player_id = "oe:player:oe:team:t1_mid"
    conn = sqlite3.connect(str(tmp_path / "games_api.db"))
    conn.execute(
        "INSERT INTO player_aliases (player_id, source, source_name) VALUES (?, ?, ?)",
        (player_id, "manual", "Current Mid"),
    )
    conn.commit()
    conn.close()

    players = client.get("/api/games/g1").json()["blue"]["players"]
    mid = next(player for player in players if player["player_id"] == player_id)
    assert mid["player_name"] == "Current Mid"


def test_game_detail_missing_404(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    assert client.get("/api/games/nope").status_code == 404
