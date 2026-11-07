"""Champion browser + detail API tests (aggregation math on the shared fixture)."""

from tests.dashboard.conftest import RANGE, make_games_client


def test_champion_table_math(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    data = client.get(f"/api/champions?{RANGE}").json()
    # Two LCK games in the major-only scope.
    assert data["n_games"] == 2
    champs = {c["champion"]: c for c in data["champions"]}

    # Ahri: picked on t1 mid in both games, won both.
    ahri = champs["Ahri"]
    assert ahri["picks"] == 2
    assert ahri["wins"] == 2
    assert ahri["losses"] == 0
    assert ahri["bans"] == 0
    assert ahri["presence"] == 1.0
    assert ahri["winrate"] == 1.0
    assert ahri["main_role"] == "mid"

    # Zed: ban-only (never picked) -> still listed, null pick stats.
    zed = champs["Zed"]
    assert zed["picks"] == 0
    assert zed["bans"] == 1
    assert zed["presence"] == 0.5
    assert zed["winrate"] is None
    assert zed["kda"] is None


def test_champion_role_and_side_filter(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    mids = client.get(f"/api/champions?{RANGE}&role=mid").json()["champions"]
    assert "Ahri" in {c["champion"] for c in mids}
    assert "Aatrox" not in {c["champion"] for c in mids}  # Aatrox is top

    # Ahri is blue-side in g1, red-side in g2. Side filter splits picks.
    blue = {
        c["champion"]: c
        for c in client.get(f"/api/champions?{RANGE}&side=blue").json()["champions"]
    }
    assert blue["Ahri"]["picks"] == 1


def test_champion_detail(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    d = client.get(f"/api/champions/Ahri?{RANGE}").json()
    s = d["stats"]
    assert s["picks"] == 2
    assert s["wins"] == 2
    assert s["picks_blue"] == 1
    assert s["picks_red"] == 1
    assert s["winrate"] == 1.0
    # Draft grade presence: Ahri's g1 side (blue) graded A; g2 has no draft score.
    assert d["grade_presence"] == {"S": 0, "A": 1, "B": 0, "C": 0, "D": 0, "ungraded": 1}
    # recent_games carry opponent + matchup label.
    assert d["recent_games"]

    # Unknown champion in range -> 404.
    assert client.get(f"/api/champions/Nonexistent?{RANGE}").status_code == 404


def test_champion_best_players(tmp_path, db_template_path):
    client = make_games_client(tmp_path, db_template_path)
    d = client.get(f"/api/champions/Ahri?{RANGE}").json()
    # The t1 mid played Ahri twice (>=2 games threshold), both wins.
    players = {p["player_id"]: p for p in d["best_players"]}
    mid = players["oe:player:oe:team:t1_mid"]
    assert mid["games"] == 2
    assert mid["wins"] == 2
    assert mid["winrate"] == 1.0
