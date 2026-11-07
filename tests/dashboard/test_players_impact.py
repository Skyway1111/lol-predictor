"""Players endpoint live-Impact extension: latest asof<=end_date, dash when absent."""

import json
import shutil
import sqlite3
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.dashboard.app_factory import create_app
from tests.dashboard.conftest import seed_terminal_data


def _impact_db(tmp_path, db_template_path):
    """Terminal-fixture DB with extra faker Impact rows straddling the window end."""
    db = tmp_path / "impact_api.db"
    shutil.copy(db_template_path, db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))
    # faker has two Impact rows; the latest on/before the window end wins.
    # A row after the window end must not be selected.
    conn.executemany(
        "INSERT INTO player_impact_scores (player_id, model_id, asof_date, role, impact) "
        "VALUES (?,?,?,?,?)",
        [
            ("oe:player:faker", "impact:z", "2026-01-05", "mid", 1.0),
            ("oe:player:faker", "impact:z", "2026-03-18", "mid", 2.5),
            ("oe:player:faker", "impact:z", "2026-09-01", "mid", 9.9),  # future -> ignored
        ],
    )
    conn.commit()
    conn.close()
    return db


def _client_with_impact(tmp_path, db_template_path):
    return TestClient(create_app(db_path=_impact_db(tmp_path, db_template_path)))


def test_players_live_impact(tmp_path, db_template_path):
    client = _client_with_impact(tmp_path, db_template_path)
    data = client.get("/api/players?major_only=false&end_date=2026-03-20").json()
    players = {p["player_id"]: p for p in data["players"]}

    faker = players["oe:player:faker"]
    assert faker["impact"] == 2.5  # latest asof <= end_date
    assert faker["impact_asof"] == "2026-03-18"

    # A player with no Impact row -> null (dash), key still present.
    caps = players["oe:player:caps"]
    assert "impact" in caps
    assert caps["impact"] is None
    assert caps["impact_asof"] is None


def test_players_sort_impact_desc_nulls_last(tmp_path, db_template_path):
    """The list is Impact-ordered; players below the qualification bar sink to the end."""
    client = _client_with_impact(tmp_path, db_template_path)
    players = client.get("/api/players?major_only=false&end_date=2026-03-20").json()["players"]

    rated = [p["impact"] for p in players if p["impact"] is not None]
    assert rated == sorted(rated, reverse=True)
    first_null = next((i for i, p in enumerate(players) if p["impact"] is None), len(players))
    assert all(p["impact"] is None for p in players[first_null:])


def test_one_impact_per_player_and_date_across_surfaces(tmp_path, db_template_path):
    """The candidate-vs-realized trap across pages: one Impact per (player, end_date)."""
    # The roster and the players page bind the same end_date, so moving the date bar
    # moves both to the same number for the same player.
    client = _client_with_impact(tmp_path, db_template_path)
    for end_date, expected in (("2026-03-20", 2.5), ("2026-12-31", 9.9)):
        page = client.get(f"/api/players?major_only=false&end_date={end_date}").json()["players"]
        roster = client.get(f"/api/teams/oe:team:t1/roster?end_date={end_date}").json()["players"]
        page_impact = {p["player_id"]: p["impact"] for p in page}
        roster_impact = {p["player_id"]: p["impact"] for p in roster}

        assert page_impact["oe:player:faker"] == expected
        assert roster_impact["oe:player:faker"] == page_impact["oe:player:faker"]


def test_overview_qualification_bar_follows_the_artifact_recipe(
    tmp_path, db_template_path, monkeypatch
):
    """The served qualification label agrees with the supported artifact recipe."""
    from league_predictor import config
    from league_predictor.predraft.ratings.player_impact import IMPACT_RECIPE

    db = _impact_db(tmp_path, db_template_path)
    stamped = IMPACT_RECIPE.to_dict()
    models_dir = tmp_path / "models"
    (models_dir / "impact").mkdir(parents=True)
    (models_dir / "impact" / "impact.json").write_text(
        json.dumps({"lambda": 10.0, "recipe": stamped}), encoding="utf-8"
    )
    monkeypatch.setattr(config, "MODELS_DIR", models_dir)

    client = TestClient(create_app(db_path=db))
    bar = client.get("/api/players/oe:player:faker/overview").json()["impact_threshold"]

    assert bar["min_games"] == stamped["min_games"]
    assert bar["window_months"] == stamped["qualification_window_months"]


@pytest.mark.slow
def test_route_impact_equals_the_prd_artifact():
    """Every number the route serves is the number the artifact published."""
    # The candidate-vs-realized trap: a displayed value must come from the model that
    # produced it, never from a parallel computation that can drift.
    from league_predictor.config import DB_PATH, MODELS_DIR

    artifact = Path(MODELS_DIR) / "impact" / "impact.json"
    if not artifact.is_file() or not Path(DB_PATH).exists():
        pytest.skip("no live artifact or DB available")
    published = json.loads(artifact.read_text(encoding="utf-8"))["players"]

    client = TestClient(create_app(db_path=Path(DB_PATH)))
    rows = client.get("/api/players?major_only=false&min_games=0").json()["players"]
    checked = 0
    for row in rows:
        expected = published.get(row["player_id"])
        if expected is None:
            assert row["impact"] is None, (
                f"{row['player_id']} serves a value the artifact never published"
            )
            continue
        assert row["impact"] == pytest.approx(expected["impact_pp"]), row["player_id"]
        assert row["stats_pp"] == pytest.approx(expected["stats_pp"]), row["player_id"]
        assert row["results_pp"] == pytest.approx(expected["results_pp"]), row["player_id"]
        assert row["percentile"] == pytest.approx(expected["percentile"]), row["player_id"]
        assert row["grade"] == expected["grade"], row["player_id"]
        checked += 1
    assert checked > 0, "expected at least one qualified player on the live route"
