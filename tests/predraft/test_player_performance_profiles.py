"""Tests for recency-weighted player performance profiles."""

import sqlite3
from contextlib import closing
from datetime import date

import pytest

from league_predictor.db.schema import CURRENT_SCHEMA_SQL
from league_predictor.db.schema import apply_current_schema as create_schema
from league_predictor.decay import half_life_weight
from league_predictor.predraft.ratings.player_performance_prefix import (
    _build_prefix,
    build_player_performance_prefixes,
)
from tests.predraft._performance_reference import (
    compute_performance_profile_from_rows,
    compute_player_performance_profile,
)

# --- Helpers ---


def _assert_profile_close(actual: dict, expected: dict) -> None:
    """Compare nested profile dictionaries with float tolerance."""
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        if isinstance(value, dict):
            _assert_profile_close(value, expected[key])
        elif isinstance(value, float):
            assert value == pytest.approx(expected[key])
        else:
            assert value == expected[key]


def _insert_match(conn, match_id, match_date):
    ts = str(match_date)
    ts = (ts.replace(" ", "T", 1) if len(ts) > 10 else ts + "T10:00:00") + "+00:00"
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, status) "
        "VALUES (?, 'TestLeague', 'teamA', 'teamB', ?, 'completed')",
        (f"ser_{match_id}", ts),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner) VALUES (?, ?, 1, 'teamA')",
        (match_id, f"ser_{match_id}"),
    )


def _insert_player_game(conn, match_id, player_id, **kwargs):
    defaults = {
        "team_id": "teamA",
        "champion_id": "Jax",
        "role": "top",
        "result": 1,
        "kills": 3,
        "deaths": 1,
        "assists": 5,
        "teamkills": 15,
        "teamdeaths": 8,
        "damageshare": 0.28,
        "killparticipation": 0.53,
        "deathshare": 0.12,
        "vspm": 1.8,
        "controlwardsbought": 4,
        "golddiffat15": 500.0,
        "csdiffat15": 12.0,
        "xpdiffat15": 350.0,
        "goldat15": 5500.0,
        "csat15": 142.0,
        "xpat15": 7200.0,
        "dpm": 650.0,
        "cspm": 8.5,
        "earnedgoldshare": 0.25,
        "firstblood": 1,
        "firsttower": 1,
        "firstdragon": 1,
        "firstbaron": 1,
    }
    defaults.update(kwargs)
    cols = ", ".join(["game_id", "player_id", *list(defaults.keys())])
    placeholders = ", ".join(["?"] * (2 + len(defaults)))
    conn.execute(
        f"INSERT INTO game_players ({cols}) VALUES ({placeholders})",
        [match_id, player_id, *list(defaults.values())],
    )


# --- recency weight tests ---


def test_recency_weight_recent():
    assert half_life_weight(0, 30) == pytest.approx(1.0)


def test_recency_weight_at_halflife():
    assert half_life_weight(30, 30) == pytest.approx(0.5, abs=1e-9)


def test_recency_weight_old():
    # 90 days = 3 half-lives → 0.5^3 = 0.125
    assert half_life_weight(90, 30) == pytest.approx(0.125, abs=1e-9)


# --- compute_player_performance_profile tests ---


def test_cached_prefix_does_not_cross_recycled_connection_ids(monkeypatch):
    from league_predictor.predraft import feature_cache

    monkeypatch.setattr(feature_cache, "id", lambda _: 1, raising=False)
    for champion in ("Ahri", "Annie"):
        with closing(sqlite3.connect(":memory:")) as conn:
            conn.row_factory = sqlite3.Row
            create_schema(conn)
            _insert_match(conn, "m1", "2025-03-15")
            _insert_player_game(conn, "m1", "player1", champion_id=champion)
            conn.commit()
            prefix = build_player_performance_prefixes(conn, ["player1"], 60.0)["player1"]
            assert prefix.profile_at("2025-03-16")["champion_pool"]["top_champions"] == [champion]


@pytest.mark.parametrize("write_mode", ["external_commit", "rollback"])
@pytest.mark.parametrize("with_generation", [True, False], ids=["tracked", "legacy"])
def test_cached_prefix_tracks_transaction_visible_facts(
    tmp_db, tmp_path, write_mode, with_generation
):
    if with_generation:
        create_schema(tmp_db)
    else:
        tmp_db.executescript(CURRENT_SCHEMA_SQL)
        tmp_db.execute("DROP TABLE facts_state")
    _insert_match(tmp_db, "m1", "2025-03-15")
    _insert_player_game(tmp_db, "m1", "player1", champion_id="Ahri")
    tmp_db.commit()

    def champions():
        prefix = build_player_performance_prefixes(tmp_db, ["player1"], 60.0)["player1"]
        return prefix.profile_at("2025-03-16")["champion_pool"]["top_champions"]

    assert champions() == ["Ahri"]
    update = "UPDATE game_players SET champion_id = 'Annie' WHERE player_id = 'player1'"
    if write_mode == "external_commit":
        with closing(sqlite3.connect(tmp_path / "test.db")) as writer:
            writer.execute(update)
            writer.commit()
        assert champions() == ["Annie"]
    else:
        tmp_db.execute(update)
        assert champions() == ["Annie"]
        tmp_db.rollback()
        assert champions() == ["Ahri"]


def test_compute_player_performance_profile_single_game(tmp_db):
    tmp_db.row_factory = sqlite3.Row
    create_schema(tmp_db)
    _insert_match(tmp_db, "m1", "2025-03-15")
    _insert_player_game(tmp_db, "m1", "player1")
    tmp_db.commit()

    # as_of must be strictly after the match date (query uses date < cutoff)
    result = compute_player_performance_profile(tmp_db, "player1", date(2025, 3, 16))
    assert result is not None
    assert "laning" in result
    assert "teamfighting" in result
    assert "vision" in result
    assert "objective" in result
    assert "champion_pool" in result


def test_compute_player_performance_profile_recency_weights_recent_more(tmp_db):
    """A recent good game should outweigh an old bad game."""
    tmp_db.row_factory = sqlite3.Row
    create_schema(tmp_db)

    # Old game: poor stats (90 days ago)
    _insert_match(tmp_db, "m_old", "2024-12-15")
    _insert_player_game(
        tmp_db,
        "m_old",
        "player1",
        golddiffat15=-1000.0,
        csdiffat15=-20.0,
        xpdiffat15=-500.0,
        dpm=200.0,
        cspm=5.0,
        kills=0,
        deaths=5,
        assists=1,
    )

    # Recent game: great stats (0 days ago)
    _insert_match(tmp_db, "m_new", "2025-03-15")
    _insert_player_game(
        tmp_db,
        "m_new",
        "player1",
        golddiffat15=2000.0,
        csdiffat15=30.0,
        xpdiffat15=800.0,
        dpm=1000.0,
        cspm=10.0,
        kills=8,
        deaths=1,
        assists=7,
    )
    tmp_db.commit()

    # as_of must be strictly after the latest match date (query uses date < cutoff)
    result = compute_player_performance_profile(tmp_db, "player1", date(2025, 3, 16))
    # Recent game dominates: gold diff should be positive
    assert result["laning"]["golddiffat15"] > 0
    # DPM should be closer to recent game's 1000 than old game's 200
    assert result["teamfighting"]["dpm"] > 500


class TestKdaDenominatorEpsilon:
    """The KDA denominator is ``avg(deaths) + 1e-3``: an epsilon guard, not a 1.0 floor."""

    def test_sub_one_death_average_does_not_halve_kda(self, tmp_db):
        """avg deaths 0.5, avg kills+assists 5: KDA is 5/(0.5+1e-3) ~= 9.98, not 5.0."""
        conn = tmp_db
        conn.row_factory = sqlite3.Row
        create_schema(conn)
        as_of = date(2026, 4, 16)

        # 10 games same day so weights are identical.
        match_date = "2026-04-15"
        for i in range(10):
            _insert_match(conn, f"m{i}", match_date)
            deaths = 1 if i < 5 else 0
            _insert_player_game(
                conn,
                f"m{i}",
                "tester",
                kills=3,
                deaths=deaths,
                assists=2,
            )
        conn.commit()

        rating = compute_player_performance_profile(conn, "tester", as_of)
        assert rating is not None
        kda = rating["teamfighting"]["kda"]
        # Floor-of-1 regression would produce kda ≈ 5.0.
        # Epsilon-in-denominator produces kda ≈ 9.98.
        assert kda > 9.0


def test_row_and_prefix_readers_share_one_profile_assembler_across_cutoffs():
    """Imputed, fallback, and full-laning prefixes produce identical public profiles."""

    def row(day, champion, *, at10=None, at15=None):
        return {
            "match_date": day,
            "series_id": f"ser_{day}",
            "game_no": 1,
            "champion_id": champion,
            "kills": 2,
            "deaths": 1,
            "assists": 3,
            "damageshare": 0.25,
            "killparticipation": 0.6,
            "vspm": 1.2,
            "controlwardsbought": 2,
            "dpm": 500,
            "cspm": 8,
            "earnedgoldshare": 0.24,
            "firstblood": 0,
            "firsttower": 1,
            "firstdragon": 0,
            "firstbaron": 1,
            "golddiffat15": at15,
            "csdiffat15": at15,
            "xpdiffat15": at15,
            "golddiffat10": at10,
            "csdiffat10": at10,
            "xpdiffat10": at10,
        }

    ascending = [
        row("2026-01-01", "A"),
        row("2026-02-01", "B", at10=30.0),
        row("2026-03-01", "A", at15=60.0),
    ]
    prefix = _build_prefix(ascending, half_life=60.0)

    for count, cutoff, quality in (
        (1, "2026-01-15", "imputed"),
        (2, "2026-02-15", "partial"),
        (3, "2026-03-15", "full"),
    ):
        row_profile = compute_performance_profile_from_rows(
            list(reversed(ascending[:count])), date.fromisoformat(cutoff), half_life=60.0
        )
        prefix_profile = prefix.profile_at(cutoff)
        assert row_profile is not None
        assert prefix_profile is not None
        _assert_profile_close(row_profile, prefix_profile)
        assert prefix_profile["data_quality"] == quality
        assert not any(prefix_profile["present"].values())

    assert prefix.profile_at("2026-02-15")["champion_pool"]["top_champions"] == ["B", "A"]
