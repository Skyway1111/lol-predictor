"""Analytics mart builder tests: snapshot contents, baselines, and generation transactions."""

import math
import shutil
import sqlite3
from datetime import date

import pytest

from league_predictor.analytics._common import tracked_builder
from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.analytics.row_types import TeamPowerDisplayRow
from league_predictor.predraft.ratings.player_scoring import compute_position_baselines
from tests.dashboard.conftest import seed_terminal_data

_AS_OF = date(2026, 3, 20)


def _snapshot_row(conn, player_id):
    """The player's ``player_daily_snapshot`` row on ``_AS_OF`` (None when absent)."""
    q = "SELECT * FROM player_daily_snapshot WHERE player_id = ? AND date = ?"
    return conn.execute(q, (player_id, _AS_OF.isoformat())).fetchone()


def test_kda_baseline_is_per_player_aggregate_not_exploded(tmp_path, db_template_path):
    """The KDA baseline is a per-player aggregate ratio, not a per-game average."""
    # A per-GAME (k+a)/(deaths+eps) mean explodes on zero-death games, killing KDA as a z-score.
    db = tmp_path / "kda.db"
    shutil.copy(db_template_path, db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    from tests.predraft.conftest import insert_game

    for i in range(1, 5):
        insert_game(
            conn,
            game_id=f"m{i}",
            day="2026-03-10",
            team1="t1",
            team2="t2",
            winner="t1",
            tournament="LCK",
            game_no=1,
            best_of=3,
        )
    # Player A: a zero-death game (would explode per-game) + a normal game.
    # agg KDA = (10+7) / (0+2 + eps) ~= 8.5
    pg = [
        ("m1", "A", "t1", "mid", 5, 0, 5),
        ("m2", "A", "t1", "mid", 3, 2, 4),
        # Player B: agg KDA = (10+7)/(1+3) ~= 4.25
        ("m3", "B", "t2", "mid", 4, 1, 6),
        ("m4", "B", "t2", "mid", 2, 3, 5),
    ]
    for mid, pid, team, role, k, d, a in pg:
        conn.execute(
            "INSERT INTO game_players (game_id, player_id, team_id, role, kills, deaths, assists) "
            "VALUES (?,?,?,?,?,?,?)",
            (mid, pid, team, role, k, d, a),
        )
    conn.commit()
    means, _ = compute_position_baselines(conn, date(2026, 3, 20), min_games=1)["mid"]
    conn.close()
    expected = ((17 / 2.001) + (17 / 4.001)) / 2  # mean of per-player aggregates
    assert means["kda"] < 50, f"KDA baseline exploded: {means['kda']}"
    assert abs(means["kda"] - expected) < 0.1, f"got {means['kda']}, expected ~{expected:.2f}"


def test_refresh_analytics_marts_builds_all_marts(tmp_path, db_template_path):
    db_path = tmp_path / "analytics.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)

    result = refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))

    assert result["team_daily_snapshot"] >= 4
    assert result["player_daily_snapshot"] >= 10
    assert result["power_rating_snapshots"] >= 4

    snapshot = conn.execute(
        "SELECT * FROM team_daily_snapshot WHERE team_id = 'oe:team:t1' AND date = '2026-03-20'"
    ).fetchone()
    assert snapshot is not None
    assert snapshot["display_name"] == "T1"
    assert snapshot["region"] == "LCK"
    # power_rating is the skill-filter-derived team power (1500-centered display scale).
    assert isinstance(snapshot["power_rating"], float)
    assert math.isfinite(snapshot["power_rating"])

    power = conn.execute(
        "SELECT * FROM power_rating_snapshots WHERE date = '2026-03-20' AND team_id = 'oe:team:t1'"
    ).fetchone()
    assert power is not None
    assert power["global_rank"] >= 1
    # The power snapshot mirrors the team_daily_snapshot power_rating.
    assert power["rating"] == snapshot["power_rating"]

    # Player snapshot schema checks.
    faker = _snapshot_row(conn, "oe:player:faker")
    assert faker is not None
    import json

    pillars = json.loads(faker["pillars_json"])
    assert {p["name"] for p in pillars} == {"laning", "combat", "vision"}
    assert all(p["rating"] is None or 1 <= p["rating"] <= 99 for p in pillars)
    pool = json.loads(faker["champion_pool_json"])
    assert "champion_diversity" in pool
    assert "champions" in pool

    conn.close()


def test_pass1_reads_the_stat_context_aggregates(tmp_path, db_template_path, monkeypatch):
    """Pass 1 is fed by stat_context: detail z, presence, and quality come from ONE home."""
    import numpy as np

    from league_predictor.analytics.player_snapshot import _compute_player_intermediates
    from league_predictor.predraft.ratings import stat_context
    from league_predictor.predraft.ratings.stat_context import PILLARS, STATS, ContextZ

    db_path = tmp_path / "pass1.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)

    day = date(2026, 3, 10).toordinal()
    stat_z = np.full((1, len(STATS)), np.nan)
    stat_z[0, STATS.index("dpm")] = 1.2
    stat_z[0, STATS.index("golddiffat15")] = 0.4
    pillar_z = np.full((1, len(PILLARS)), np.nan)
    pillar_z[0, PILLARS.index("combat")] = 1.2
    pillar_z[0, PILLARS.index("laning")] = 0.4
    fake_ctx = ContextZ(
        player_ids=("oe:player:faker",),
        appearance_players=np.array([0], dtype=np.int32),
        appearance_dates=np.array([day], dtype=np.int64),
        appearance_roles=np.array([2], dtype=np.int16),
        stat_z=stat_z,
        pillar_z=pillar_z,
        through=date(2026, 3, 20),
    )
    monkeypatch.setattr(stat_context, "build_game_context_z", lambda *_, **__: fake_ctx)

    intermediates = _compute_player_intermediates(conn, date(2026, 3, 20))
    conn.close()

    faker = next(i for i in intermediates if i.player_id == "oe:player:faker")
    assert faker.games_played == 1
    assert faker.zscores["dpm"] == pytest.approx(1.2)
    assert faker.present_mask["dpm"]
    assert not faker.present_mask["kda"]
    assert faker.pillar_scores["combat"] == pytest.approx(1.2)
    assert faker.pillar_quality["combat"] == "partial"  # dpm alone of the combat stats
    assert faker.pillar_quality["vision"] == "dropped"
    # Players absent from the context produce no intermediate at all.
    assert all(i.player_id == "oe:player:faker" for i in intermediates)


def test_sup_snapshot_emits_four_pillar_pillars_json(tmp_path, db_template_path):
    """Sup rows get four pillars_json entries; other roles get three, each with data_quality."""
    import json

    db_path = tmp_path / "sup_pillars.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))

    keria = conn.execute(
        "SELECT pillars_json FROM player_daily_snapshot "
        "WHERE player_id = 'oe:player:keria' AND date = '2026-03-20'"
    ).fetchone()
    sup_pillars = json.loads(keria["pillars_json"])
    names = [p["name"] for p in sup_pillars]
    assert names == ["lane", "playmaking", "utility", "vision"]
    for p in sup_pillars:
        assert "data_quality" in p
        assert p["data_quality"] in ("full", "partial", "dropped")

    faker = conn.execute(
        "SELECT pillars_json FROM player_daily_snapshot "
        "WHERE player_id = 'oe:player:faker' AND date = '2026-03-20'"
    ).fetchone()
    mid_pillars = json.loads(faker["pillars_json"])
    assert [p["name"] for p in mid_pillars] == ["laning", "combat", "vision"]

    conn.close()


def test_position_baselines_include_extended_stats(tmp_path, db_template_path):
    """compute_position_baselines covers the extended OE box-score stats (wpm, wcpm, ...)."""
    db_path = tmp_path / "baselines.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    baselines = compute_position_baselines(conn, date(2026, 3, 20), min_games=1)
    conn.close()

    assert "sup" in baselines
    sup_means, sup_stdevs = baselines["sup"]
    for stat in (
        "wpm",
        "wcpm",
        "assistsat15",
        "firstbloodassist",
        "damagetakenperminute",
        "damagemitigatedperminute",
    ):
        assert stat in sup_means, stat
        assert stat in sup_stdevs, stat


def test_champion_pool_counts_current_season_games_only(tmp_path, db_template_path):
    """Champion pool counts current-season games only (>= Jan 1 of the as-of year)."""
    import json

    db_path = tmp_path / "champ_pool.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    # A prior-season (2025) game on a different champion must NOT appear.
    from tests.predraft.conftest import insert_game

    insert_game(
        conn,
        game_id="m_old",
        day="2025-11-01",
        team1="oe:team:t1",
        team2="oe:team:hle",
        winner="oe:team:t1",
        tournament="LCK 2025",
        game_no=1,
        best_of=3,
        duration_s=1900,
        blue_team="oe:team:t1",
        patch="14.20",
        event_type="regular",
    )
    conn.execute(
        """
        INSERT INTO game_players (game_id, player_id, team_id, champion_id, role,
            result, kills, deaths, assists, teamkills, teamdeaths)
        VALUES ('m_old', 'oe:player:faker', 'oe:team:t1', 'OldSeasonChamp', 'mid',
            1, 5, 1, 6, 14, 8)
        """
    )
    conn.commit()
    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))
    row = conn.execute(
        "SELECT champion_pool_json FROM player_daily_snapshot "
        "WHERE player_id = 'oe:player:faker' AND date = '2026-03-20'"
    ).fetchone()
    conn.close()

    pool = json.loads(row["champion_pool_json"])
    champs = {c["champion_id"]: c["games"] for c in pool["champions"]}
    assert "OldSeasonChamp" not in champs, "prior-season game must be excluded"
    assert champs.get("Champ3") == 3, "faker played Champ3 in all three 2026 games"
    assert pool["champion_diversity"] == 1
    assert pool["top_champions"] == ["Champ3"]


def test_role_mean_stddev_skips_single_game_players():
    """_role_mean_stddev_for_consistency excludes players with fewer than two games."""
    from league_predictor.analytics.player_draft_inputs import (
        _role_mean_stddev_for_consistency,
    )

    # Two top-lane players: one with 5 games of variation, one with 1 game.
    per_game = {
        "p_multi": {"a": [1.0, 2.0, 1.5, 2.5, 1.2], "b": [0.5, 1.5, 1.0, 2.0, 0.8]},
        "p_single": {"a": [3.0], "b": [3.0]},
    }
    roles = {"p_multi": "top", "p_single": "top"}
    out = _role_mean_stddev_for_consistency(per_game, roles)
    # Only p_multi contributes; the single-game player is excluded.
    assert "top" in out
    # Sanity: result is positive (real variability) — exact value depends on pstdev.
    assert out["top"] > 0


# ── Transaction discipline: savepoints (issue 7) + slice replace (issue 19) ──


def test_tracked_builder_savepoint_rolls_back_only_its_own_dml(tmp_path, db_template_path):
    """A raising builder's DML rolls back to its SAVEPOINT; earlier generation DML survives."""
    db_path = tmp_path / "savepoint.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Explicit generation-transaction control, exactly as refresh_analytics_marts uses.
    conn.isolation_level = None
    conn.execute("BEGIN")
    conn.execute("INSERT INTO team_daily_snapshot (team_id, date) VALUES ('keep', '2026-03-20')")

    run_log: list[dict] = []
    with (  # noqa: PT012 - the DML and the raise must both sit inside the savepoint under test
        pytest.raises(RuntimeError, match="boom"),
        tracked_builder(conn, "team_daily_snapshot", "2026-03-20", run_log),
    ):
        conn.execute(
            "INSERT INTO team_daily_snapshot (team_id, date) VALUES ('gone', '2026-03-20')"
        )
        raise RuntimeError("boom")
    conn.execute("COMMIT")

    surviving = {
        r["team_id"]
        for r in conn.execute("SELECT team_id FROM team_daily_snapshot WHERE date = '2026-03-20'")
    }
    conn.close()
    assert surviving == {"keep"}, (
        "the failing builder's row rolls back to its savepoint; earlier DML survives"
    )
    assert run_log == [
        {
            "builder": "team_daily_snapshot",
            "status": "error",
            "details": {"as_of_date": "2026-03-20", "error": "boom"},
        }
    ]


def test_same_date_rerun_drops_disappeared_player(tmp_path, db_template_path):
    """Issue 19: a player vanishing between same-date refreshes leaves no stale current row."""
    db_path = tmp_path / "rerun.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)

    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))
    assert _snapshot_row(conn, "oe:player:zeus") is not None, (
        "zeus should be in the first generation"
    )

    # Zeus retires: his box-score rows leave the facts entirely.
    conn.execute("DELETE FROM game_players WHERE player_id = 'oe:player:zeus'")
    conn.commit()

    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))

    zeus = _snapshot_row(conn, "oe:player:zeus")
    faker = _snapshot_row(conn, "oe:player:faker")
    conn.close()
    assert zeus is None, "the disappeared player's stale row must be cleared by the slice replace"
    assert faker is not None, "a still-active player keeps his row"


def test_midsequence_failure_publishes_no_generation_and_records_statuses(
    tmp_path,
    db_template_path,
    monkeypatch,
):
    """Issue 7: a builder failing mid-sequence rolls back the whole generation; runs still log."""
    db_path = tmp_path / "midseq.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)

    # Fail the third builder (team-power display) after team + player have written.
    def _boom():
        raise RuntimeError("injected power-rating failure")

    monkeypatch.setattr(TeamPowerDisplayRow, "insert_sql", staticmethod(_boom))

    with pytest.raises(RuntimeError, match="power-rating failure"):
        refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))

    for table in ("team_daily_snapshot", "player_daily_snapshot", "power_rating_snapshots"):
        n = conn.execute(f"SELECT count(*) FROM {table} WHERE date = '2026-03-20'").fetchone()[0]
        assert n == 0, f"{table} must have no rows from the rolled-back generation"

    # Bookkeeping survives the rollback: the two successful builders record ok, the
    # failing one records error.
    statuses = {
        r["builder"]: r["status"]
        for r in conn.execute("SELECT builder, status FROM analytics_runs ORDER BY id")
    }
    conn.close()
    assert statuses.get("team_daily_snapshot") == "ok"
    assert statuses.get("player_daily_snapshot") == "ok"
    assert statuses.get("power_rating_snapshots") == "error"


def test_a_present_snapshot_group_missing_a_key_fails_the_mart_loudly():
    """A renamed producer field aborts the team mart build; it never writes silent zeros."""
    from league_predictor.analytics.team_snapshot import _required_group

    assert _required_group({"form": {"a": 1.0, "b": 2.0}}, "form", ("a", "b")) == {
        "a": 1.0,
        "b": 2.0,
    }
    # An absent group is a team the producer has no rows for: zeros, by design.
    assert _required_group({}, "form", ("a", "b")) == {"a": 0.0, "b": 0.0}
    assert _required_group(None, "form", ("a",)) == {"a": 0.0}
    with pytest.raises(RuntimeError, match="'form' lacks \\['b'\\]"):
        _required_group({"form": {"a": 1.0}}, "form", ("a", "b"))
