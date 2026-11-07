"""Per-game pillar scoring and the draft axes the player snapshot attaches to it."""

import shutil
import sqlite3
from datetime import date

from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.analytics.player_draft_inputs import _load_pick_rows_for_year
from tests.dashboard.conftest import seed_terminal_data


def test_compute_per_game_pillars_requires_all_pillars_present(tmp_path, db_template_path):
    """A game with one fully NULL pillar is dropped, keeping every pillar list equal-length."""
    from league_predictor.analytics.player_draft_inputs import (
        _compute_per_game_pillars_for_year,
    )

    db_path = tmp_path / "pergame.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # Two top games for one player; the first has dpm=NULL → combat pillar drops.
    from tests.predraft.conftest import insert_game

    insert_game(
        conn,
        game_id="m1",
        day="2026-03-01",
        team1="t1",
        team2="t2",
        winner=None,
        tournament="LCK",
        game_no=1,
        best_of=3,
    )
    insert_game(
        conn,
        game_id="m2",
        day="2026-03-02",
        team1="t1",
        team2="t2",
        winner=None,
        tournament="LCK",
        game_no=1,
        best_of=3,
    )
    conn.executemany(
        "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, "
        "golddiffat15, csdiffat15, xpdiffat15, cspm, dpm, killparticipation, "
        "firstblood, vspm, earnedgoldshare) "
        "VALUES (?, 'p', 't1', 'X', 'top', ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            # m1: every combat stat (dpm, killparticipation, firstblood) is NULL,
            # so the combat pillar is unscorable and the whole game must drop.
            ("m1", 100.0, 5.0, 50.0, 7.0, None, None, None, 1.5, 0.21),
            # m2: every stat present.
            ("m2", 200.0, 10.0, 100.0, 8.0, 600.0, 0.6, 1, 1.8, 0.22),
        ],
    )
    conn.commit()

    # Minimal tier baselines so the per-game z-score path works.
    bl_means = dict.fromkeys(
        (
            "golddiffat15",
            "csdiffat15",
            "xpdiffat15",
            "cspm",
            "dpm",
            "killparticipation",
            "firstblood",
            "vspm",
            "earnedgoldshare",
        ),
        0.0,
    )
    bl_stdevs = dict.fromkeys(bl_means, 1.0)
    tier_baselines = {1: {"top": (bl_means, bl_stdevs)}}

    out = _compute_per_game_pillars_for_year(
        conn,
        year_start=date(2026, 1, 1),
        as_of=date(2026, 3, 20),
        tier_baselines=tier_baselines,
        roles_by_player={"p": "top"},
        player_tiers={"p": 1},
    )
    # m2 contributes; m1's combat pillar can't score so the whole game is dropped.
    assert "p" in out
    lengths = {len(v) for v in out["p"].values()}
    assert lengths == {1}, f"all pillars must align; got {lengths}"
    conn.close()


def test_compute_per_game_pillars_skips_unknown_players(tmp_path, db_template_path):
    """Players missing from roles_by_player are skipped — they have no snapshot row."""
    from league_predictor.analytics.player_draft_inputs import (
        _compute_per_game_pillars_for_year,
    )

    db_path = tmp_path / "skip.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from tests.predraft.conftest import insert_game

    insert_game(
        conn,
        game_id="m1",
        day="2026-03-01",
        team1="t1",
        team2="t2",
        winner=None,
        tournament="LCK",
        game_no=1,
        best_of=3,
    )
    conn.execute(
        "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, "
        "golddiffat15, csdiffat15, xpdiffat15, cspm, dpm, killparticipation, "
        "firstblood, vspm, earnedgoldshare) "
        "VALUES ('m1', 'unknown_player', 't1', 'X', 'top', "
        "100, 5, 50, 7, 600, 0.6, 0, 1.5, 0.21)"
    )
    conn.commit()
    out = _compute_per_game_pillars_for_year(
        conn,
        year_start=date(2026, 1, 1),
        as_of=date(2026, 3, 20),
        tier_baselines={1: {"top": ({}, {})}},
        roles_by_player={},  # unknown_player not here → skipped
        player_tiers={},
    )
    assert out == {}
    conn.close()


def test_player_snapshot_attaches_draft_axes_when_pick_scores_present(tmp_path, db_template_path):
    """Draft-axis columns populate when ``pick_scores`` has rows for the player; else NULL."""
    from league_predictor.analytics.player_draft import MIN_GAMES
    from tests.predraft.conftest import insert_game, register_draft_model

    db_path = tmp_path / "axes.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    draft_model = register_draft_model(conn)

    # Two "mid" players are needed for percentile rank to reach 0/100 instead of the
    # lone-player 50.0. No draft_actions is seeded, so every slot classifies blind.
    pick_rows = []
    for i in range(MIN_GAMES):
        gf, gc = f"g_axes_faker_{i}", f"g_axes_caps_{i}"
        insert_game(
            conn,
            game_id=gf,
            day="2026-03-15",
            team1="A",
            team2="B",
            winner="A",
            series_id=f"ser_{gf}",
        )
        insert_game(
            conn,
            game_id=gc,
            day="2026-03-15",
            team1="A",
            team2="B",
            winner="A",
            series_id=f"ser_{gc}",
        )
        # faker scores high (percentile 0.9); caps scores low (0.2).
        pick_rows.append((gf, "blue", "mid", draft_model, "oe:player:faker", 0.9, 5, 1.0, 0.0, 0.9))
        pick_rows.append((gc, "blue", "mid", draft_model, "oe:player:caps", 0.2, 5, 1.0, 0.0, 0.2))
    conn.executemany(
        """INSERT INTO pick_scores
           (game_id, side, role, model_id, player_id,
            actual_score, alt_count, best_score, worst_score, percentile)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        pick_rows,
    )
    conn.commit()

    refresh_analytics_marts(conn, as_of_date=date(2026, 3, 20))

    faker = conn.execute(
        "SELECT blind_quality, blind_quality_n, counter_quality, counter_quality_n, "
        "       overall_quality, consistency, consistency_n "
        "FROM player_daily_snapshot "
        "WHERE player_id = 'oe:player:faker' AND date = '2026-03-20'"
    ).fetchone()
    caps = conn.execute(
        "SELECT blind_quality, blind_quality_n FROM player_daily_snapshot "
        "WHERE player_id = 'oe:player:caps' AND date = '2026-03-20'"
    ).fetchone()
    # Player without any pick_score rows.
    zeus = conn.execute(
        "SELECT blind_quality, blind_quality_n, counter_quality "
        "FROM player_daily_snapshot "
        "WHERE player_id = 'oe:player:zeus' AND date = '2026-03-20'"
    ).fetchone()
    conn.close()

    # Faker (high blind percentile) ranks 100; Caps (low) ranks 0.
    assert faker["blind_quality"] == 100.0
    assert faker["blind_quality_n"] == MIN_GAMES
    assert caps["blind_quality"] == 0.0
    assert caps["blind_quality_n"] == MIN_GAMES
    # Counter quality has 0 picks → NULL.
    assert faker["counter_quality"] is None
    assert faker["counter_quality_n"] == 0
    # Overall Quality is the blend over the non-NULL weighted axes; must be defined.
    assert faker["overall_quality"] is not None
    # Consistency: only 3 per-game samples in this fixture (< MIN_GAMES) → NULL.
    assert faker["consistency"] is None
    # A player with no pick_score rows gets NULL across the board.
    assert zeus["blind_quality"] is None
    assert zeus["counter_quality"] is None


def test_load_pick_rows_excludes_picks_after_as_of(tmp_path, db_template_path):
    """_load_pick_rows_for_year is walk-forward: picks after ``as_of`` never leak in."""
    from tests.predraft.conftest import insert_game, register_draft_model

    db_path = tmp_path / "pickrows.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    draft_model = register_draft_model(conn)

    # Two games for the same player: one before as_of, one after.
    insert_game(
        conn,
        game_id="g_before",
        day="2026-03-10",
        team1="A",
        team2="B",
        winner="A",
        series_id="ser_g_before",
    )
    insert_game(
        conn,
        game_id="g_after",
        day="2026-04-01",
        team1="A",
        team2="B",
        winner="A",
        series_id="ser_g_after",
    )
    conn.executemany(
        """INSERT INTO pick_scores
           (game_id, side, role, model_id, player_id,
            actual_score, alt_count, best_score, worst_score, percentile)
           VALUES (?, 'blue', 'mid', ?, 'oe:player:p', 0.9, 5, 1.0, 0.0, ?)""",
        [("g_before", draft_model, 0.9), ("g_after", draft_model, 0.1)],
    )
    conn.commit()

    out = _load_pick_rows_for_year(
        conn,
        year_start=date(2026, 1, 1),
        as_of=date(2026, 3, 20),
    )
    conn.close()

    # Only the pre-as_of pick survives; the 2026-04-01 pick is filtered out.
    assert len(out["oe:player:p"]) == 1
    assert out["oe:player:p"][0]["percentile"] == 0.9


def test_load_pick_rows_aggregates_mixed_vintage_rows(tmp_path, db_template_path):
    """The reader aggregates BOTH pick vintages; filtering to the newest model id drops half."""
    from tests.predraft.conftest import insert_game, register_draft_model

    db_path = tmp_path / "vintages.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    old_model = register_draft_model(conn, model_id="draft:old_vintage")
    new_model = register_draft_model(conn, model_id="draft:new_vintage")

    insert_game(
        conn,
        game_id="g_jan",
        day="2026-01-10",
        team1="A",
        team2="B",
        winner="A",
        series_id="ser_g_jan",
    )
    insert_game(
        conn,
        game_id="g_mar",
        day="2026-03-10",
        team1="A",
        team2="B",
        winner="A",
        series_id="ser_g_mar",
    )
    conn.executemany(
        """INSERT INTO pick_scores
           (game_id, side, role, model_id, player_id,
            actual_score, alt_count, best_score, worst_score, percentile)
           VALUES (?, 'blue', 'mid', ?, 'oe:player:p', 0.9, 5, 1.0, 0.0, ?)""",
        [("g_jan", old_model, 0.3), ("g_mar", new_model, 0.7)],
    )
    conn.commit()

    out = _load_pick_rows_for_year(
        conn,
        year_start=date(2026, 1, 1),
        as_of=date(2026, 3, 20),
    )
    conn.close()

    # Both vintages count — one row per slot, whatever id stamped it.
    assert sorted(r["percentile"] for r in out["oe:player:p"]) == [0.3, 0.7]
