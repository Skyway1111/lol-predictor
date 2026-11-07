"""Integration tests for the extended snapshot groups against the real facts DB."""

import sqlite3
from datetime import date
from pathlib import Path

import pytest

from league_predictor.config import DB_PATH
from league_predictor.predraft.features.snapshots import SnapshotStore

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def conn():
    """Real facts-DB connection from REBUILD_SMOKE_DB or the live DB; skips when neither works."""
    import os

    path = os.environ.get("REBUILD_SMOKE_DB") or str(DB_PATH)
    if not Path(path).exists():
        pytest.skip("Database not available")
    c = sqlite3.connect(path)
    c.row_factory = sqlite3.Row
    if not c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='games'").fetchone():
        c.close()
        pytest.skip("live DB lacks the facts schema; set REBUILD_SMOKE_DB")
    yield c
    c.close()


@pytest.fixture(scope="module")
def sample_match(conn):
    """Find a real match with two teams that have history."""
    row = conn.execute(
        """SELECT s.team_a AS team1_id, s.team_b AS team2_id,
                  s.scheduled_ts AS date, s.league_id AS tournament
           FROM games g JOIN series s ON s.series_id = g.series_id
           WHERE s.scheduled_ts >= '2026-04-01' AND s.scheduled_ts < '2026-05-01'
             AND g.winner IS NOT NULL
           ORDER BY date(s.scheduled_ts), s.series_id, g.game_no LIMIT 1"""
    ).fetchone()
    if not row:
        pytest.skip("No matches found in April 2026")
    return {
        "team1_id": str(row["team1_id"]),
        "team2_id": str(row["team2_id"]),
        "as_of": date.fromisoformat(str(row["date"])[:10]),
        "tournament": str(row["tournament"]),
    }


@pytest.fixture(scope="module")
def snapshot(conn, sample_match):
    """The sample team's snapshot from a two-team store."""
    store = SnapshotStore(conn, teams=[sample_match["team1_id"], sample_match["team2_id"]])
    return store.team_snapshot(sample_match["team1_id"], sample_match["as_of"])


@pytest.fixture(scope="module")
def store(conn, sample_match):
    return SnapshotStore(conn, teams=[sample_match["team1_id"], sample_match["team2_id"]])


class TestH2HFeatures:
    def test_returns_bounded_keys(self, store, sample_match):
        result = store.h2h(
            sample_match["team1_id"], sample_match["team2_id"], sample_match["as_of"]
        )
        assert set(result) == {"h2h_games", "h2h_win_rate", "h2h_recency_wr", "h2h_days_since"}
        assert 0.0 <= result["h2h_win_rate"] <= 1.0
        assert 0.0 <= result["h2h_recency_wr"] <= 1.0

    def test_nonexistent_teams_return_defaults(self, conn):
        result = SnapshotStore(conn, teams=["fake_team_1"]).h2h(
            "fake_team_1", "fake_team_2", date(2026, 4, 1)
        )
        assert result["h2h_games"] == 0
        assert result["h2h_win_rate"] == 0.5

    def test_both_orientations_share_the_pair(self, store, sample_match):
        t1, t2, as_of = sample_match["team1_id"], sample_match["team2_id"], sample_match["as_of"]
        forward = store.h2h(t1, t2, as_of)
        reverse = store.h2h(t2, t1, as_of)
        assert forward["h2h_games"] == reverse["h2h_games"]
        assert forward["h2h_days_since"] == reverse["h2h_days_since"]
        if forward["h2h_games"]:
            assert forward["h2h_win_rate"] + reverse["h2h_win_rate"] == pytest.approx(1.0)


def test_schedule_density(snapshot):
    schedule = snapshot["schedule"]
    assert 0 <= schedule["games_7d"] <= schedule["games_14d"]


def test_roster_stability(snapshot):
    stability = snapshot["roster_stability"]
    assert set(stability) == {"roster_days_stable", "roster_total_games"}
    assert stability["roster_days_stable"] >= 0


def test_upset_and_side(snapshot):
    assert 0.0 <= snapshot["upset"]["upset_rate"] <= 1.0
    assert snapshot["upset"]["upset_rate"] + snapshot["upset"]["favorite_rate"] == pytest.approx(
        1.0
    )
    assert set(snapshot["side"]) == {"blue_wr", "red_wr", "side_diff"}


def test_win_margin_champion_pool_and_game_phase(snapshot):
    assert set(snapshot["win_margin"]) == {"avg_gold_diff_end", "stomp_rate", "close_game_rate"}
    assert snapshot["champ_pool"]["unique_champs"] >= 0
    assert set(snapshot["game_phase"]) == {"early_game_index", "late_game_index"}
