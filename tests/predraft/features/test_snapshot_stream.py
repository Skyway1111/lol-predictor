"""The snapshot store: every group's numbers on fixtures, the visibility rule, bulk == lazy."""

# file-length-ok: one module's contract mirror, split adds cross-file hops

from __future__ import annotations

import random
import sqlite3
from datetime import date, datetime

import pytest

from league_predictor import config
from league_predictor.asof import AsOf
from league_predictor.db.schema import apply_current_schema as create_schema
from league_predictor.predraft.features.snapshots import SnapshotStore
from tests.predraft.conftest import insert_game, insert_roster
from tests.predraft.features._feature_helpers import (
    _insert_match,
    _insert_player_game,
    _insert_team_game_stats,
    _setup_team,
)

_ROLES = ("top", "jng", "mid", "bot", "sup")


def _snap(conn, team, as_of):
    return SnapshotStore(conn, teams=[team]).team_snapshot(team, as_of)


# --- recent form -----------------------------------------------------------------------


def test_recent_form_all_wins(tmp_db):
    create_schema(tmp_db)
    for i in range(10):
        _insert_match(tmp_db, f"m{i}", f"2025-03-{i + 1:02d}", winner="teamA")
    tmp_db.commit()

    form = _snap(tmp_db, "teamA", date(2025, 3, 15))["form"]
    assert form == {"win_rate_10": 1.0, "win_rate_20": 1.0, "win_streak": 10, "loss_streak": 0}


def test_recent_form_no_games(tmp_db):
    create_schema(tmp_db)
    form = _snap(tmp_db, "teamA", date(2025, 3, 15))["form"]
    assert form == {"win_rate_10": 0.0, "win_rate_20": 0.0, "win_streak": 0, "loss_streak": 0}


def test_recent_form_mixed(tmp_db):
    """3 wins then 2 losses -> win_rate = 0.6, win_streak=0, loss_streak=2."""
    create_schema(tmp_db)
    wins = [(f"mw{i}", f"2025-03-{i + 1:02d}", "teamA") for i in range(3)]
    losses = [(f"ml{i}", f"2025-03-{i + 4:02d}", "teamB") for i in range(2)]
    for mid, mdate, winner in wins + losses:
        _insert_match(tmp_db, mid, mdate, team1="teamA", team2="teamB", winner=winner)
    tmp_db.commit()

    form = _snap(tmp_db, "teamA", date(2025, 3, 15))["form"]
    assert form["win_rate_10"] == pytest.approx(0.6)
    assert form["win_streak"] == 0
    assert form["loss_streak"] == 2


def test_recent_form_sees_earlier_series_and_own_earlier_maps(tmp_db):
    """The one visibility rule through the store: earlier series, own earlier maps."""
    create_schema(tmp_db)
    _insert_match(tmp_db, "yesterday", "2025-03-14 20:00:00", winner="teamA")
    _insert_match(tmp_db, "earlier_same_day", "2025-03-15 09:00:00", winner="teamB")
    _insert_match(tmp_db, "later_same_day", "2025-03-15 14:00:00", winner="teamB")
    # A Bo3 at 11:00: teamB took map one, teamA map two; map three is the query.
    tmp_db.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, best_of, status)"
        " VALUES ('ser_bo3', 'TestLeague', 'teamA', 'teamB', '2025-03-15T11:00:00+00:00', 3,"
        " 'completed')"
    )
    for game_no, winner in ((1, "teamB"), (2, "teamA"), (3, "teamB")):
        tmp_db.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner) VALUES (?, 'ser_bo3', ?, ?)",
            (f"bo3_{game_no}", game_no, winner),
        )
    tmp_db.commit()
    store = SnapshotStore(tmp_db, teams=["teamA"])

    day = store.team_snapshot("teamA", date(2025, 3, 15))["form"]
    assert day["win_rate_10"] == pytest.approx(1.0)  # yesterday only

    instant = store.team_snapshot("teamA", datetime(2025, 3, 15, 10, 0, 0))["form"]
    assert instant["win_rate_10"] == pytest.approx(0.5)  # yesterday + the 09:00 series

    map_three = AsOf.game("2025-03-15T11:00:00+00:00", "ser_bo3", 3)
    form = store.team_snapshot("teamA", map_three)["form"]
    # yesterday (W), 09:00 (L), own map one (L), own map two (W); never map three or 14:00
    assert form["win_rate_10"] == pytest.approx(0.5)
    assert form["win_streak"] == 1
    # The pair's head-to-head sees the same four maps.
    assert store.h2h("teamA", "teamB", map_three)["h2h_games"] == 4
    assert store.h2h("teamB", "teamA", map_three)["h2h_win_rate"] == pytest.approx(0.5)


# --- momentum, objectives, pace --------------------------------------------------------


def test_momentum_empty_db(tmp_db):
    create_schema(tmp_db)
    momentum = _snap(tmp_db, "teamA", date(2025, 3, 15))["momentum"]
    assert momentum == {"win_rate_5": 0.0, "performance_slope": 0.0, "days_since_last": 999.0}


def test_momentum_with_data(tmp_db):
    create_schema(tmp_db)
    for i in range(5):
        _insert_match(tmp_db, f"mw{i}", f"2025-03-{i + 6:02d}", winner="teamA")
    for i in range(3):
        _insert_match(tmp_db, f"ml{i}", f"2025-03-{i + 1:02d}", winner="teamB")
    tmp_db.commit()

    momentum = _snap(tmp_db, "teamA", date(2025, 3, 15))["momentum"]
    assert momentum["win_rate_5"] == pytest.approx(1.0)  # the newest five are all wins
    assert momentum["days_since_last"] == 5.0


def test_objective_control_empty_db(tmp_db):
    create_schema(tmp_db)
    objective = _snap(tmp_db, "teamA", date(2025, 3, 15))["objective"]
    assert set(objective) == {
        "dragon_rate",
        "baron_rate",
        "tower_rate",
        "first_dragon_rate",
        "first_baron_rate",
        "first_tower_rate",
        "first_herald_rate",
        "grub_rate",
    }
    assert set(objective.values()) == {0.0}


def test_objective_control_with_data_and_null_grubs(tmp_db):
    """Objective rates come from the team's stats rows; NULL void_grubs read as 0."""
    create_schema(tmp_db)
    _insert_match(tmp_db, "m1", "2025-03-01", winner="teamA")
    _insert_team_game_stats(
        tmp_db, "m1", "teamA", dragons=4, barons=2, towers=9, firstbaron=1, void_grubs=5
    )
    _insert_match(tmp_db, "m2", "2025-03-02", winner="teamA")
    _insert_team_game_stats(tmp_db, "m2", "teamA", dragons=2, barons=0, towers=5, void_grubs=None)
    tmp_db.commit()

    objective = _snap(tmp_db, "teamA", date(2025, 3, 15))["objective"]
    assert objective["dragon_rate"] == pytest.approx(3.0)
    assert objective["baron_rate"] == pytest.approx(1.0)
    assert objective["tower_rate"] == pytest.approx(7.0)
    assert objective["first_baron_rate"] == pytest.approx(0.5)
    assert objective["first_herald_rate"] == pytest.approx(1.0)
    assert objective["grub_rate"] == pytest.approx(2.5)


def test_team_stats_rows_off_the_series_pair_never_surface(tmp_db):
    """A stats row keyed to a team outside its series' pair belongs to no team's window."""
    create_schema(tmp_db)
    for i, day in enumerate(["2026-01-05", "2026-01-06", "2026-01-07"]):
        insert_game(tmp_db, game_id=f"g{i}", day=day, team1="A", team2="B", winner="A")
        tmp_db.execute(
            "INSERT INTO game_team_stats (game_id, team_id, dragons, barons, towers, "
            "firstdragon, firstbaron, firsttower, firstherald, void_grubs, teamkills, "
            "gamelength) VALUES (?, 'A', 2, 1, 6, 1, 0, 1, 0, 3, 12, 1900)",
            (f"g{i}",),
        )
    tmp_db.execute(
        "INSERT INTO game_team_stats (game_id, team_id, dragons, gamelength)"
        " VALUES ('g0', 'Z', 9, 1)"
    )
    tmp_db.commit()
    store = SnapshotStore(tmp_db, teams=["A", "B", "Z"])

    objective = store.team_snapshot("A", date(2026, 1, 8))["objective"]
    assert objective["dragon_rate"] == 2.0
    assert objective["first_dragon_rate"] == 1.0
    assert objective["grub_rate"] == 3.0
    assert set(store.team_snapshot("B", date(2026, 1, 8))["objective"].values()) == {0.0}
    assert set(store.team_snapshot("Z", date(2026, 1, 8))["objective"].values()) == {0.0}


def test_pace_empty_db(tmp_db):
    create_schema(tmp_db)
    pace = _snap(tmp_db, "teamA", date(2025, 3, 15))["pace"]
    assert pace == {"avg_game_duration": 0.0, "kills_per_min": 0.0, "first_blood_rate": 0.0}


def test_first_blood_rate_uses_the_pace_window(tmp_db):
    """first_blood_rate counts the newest fifteen fielded games, not the all-time rate."""
    create_schema(tmp_db)
    team_id = "oe:team:fb_test"
    for i in range(25):
        match_id = f"fb_match_{i}"
        _insert_match(
            tmp_db,
            match_id,
            f"2025-01-{i + 1:02d}",
            team1=team_id,
            team2="oe:team:other",
            winner=team_id,
            fmt="Bo3",
            blue_team=team_id,
        )
        tmp_db.execute(
            "INSERT INTO game_team_stats (game_id, team_id, gamelength, teamkills)"
            " VALUES (?,?,?,?)",
            (match_id, team_id, 1800, 10),
        )
        tmp_db.execute(
            "INSERT INTO game_players (game_id, player_id, team_id, role, result,"
            " kills, deaths, assists, firstblood) VALUES (?,?,?,?,?,?,?,?,?)",
            (match_id, "player1", team_id, "mid", 1, 5, 2, 3, 1 if i >= 20 else 0),
        )
    tmp_db.commit()

    pace = _snap(tmp_db, team_id, date(2025, 1, 26))["pace"]
    assert pace["first_blood_rate"] == pytest.approx(5 / 15)  # never 5 / 25


def test_pace_with_data(tmp_db):
    create_schema(tmp_db)
    _insert_match(tmp_db, "m1", "2025-03-01", winner="teamA")
    _insert_team_game_stats(tmp_db, "m1", "teamA", gamelength=1800, teamkills=20)
    _insert_player_game(tmp_db, "m1", "p1", team_id="teamA", firstblood=1)
    tmp_db.commit()

    pace = _snap(tmp_db, "teamA", date(2025, 3, 15))["pace"]
    assert pace["avg_game_duration"] == pytest.approx(1800.0)
    assert pace["kills_per_min"] == pytest.approx(20.0 / (1800.0 / 60.0))
    assert pace["first_blood_rate"] == pytest.approx(1.0)


# --- region -----------------------------------------------------------------------------


@pytest.mark.parametrize("international", ["MSI", "Worlds"])
def test_region_skips_international_events(tmp_db, international):
    create_schema(tmp_db)
    _insert_match(tmp_db, "m1", "2025-01-01", team1="teamA", team2="teamB")
    tmp_db.execute("UPDATE series SET league_id = 'LCK' WHERE series_id = 'ser_m1'")
    _insert_match(tmp_db, "m2", "2025-06-01", team1="teamA", team2="teamC")
    tmp_db.execute("UPDATE series SET league_id = ? WHERE series_id = 'ser_m2'", (international,))
    tmp_db.commit()

    assert _snap(tmp_db, "teamA", date(2025, 7, 1))["region"] == "LCK"


@pytest.mark.parametrize("league_id", ["LTA N", "LCKC", "NACL", "LJLCS"])
def test_region_resolves_domestic_leagues_exactly(tmp_db, league_id):
    create_schema(tmp_db)
    _insert_match(tmp_db, "m1", "2025-01-01", team1="teamA", team2="teamB")
    tmp_db.execute("UPDATE series SET league_id = ? WHERE series_id = 'ser_m1'", (league_id,))
    tmp_db.commit()

    assert _snap(tmp_db, "teamA", date(2025, 2, 1))["region"] == league_id


def test_region_of_a_team_with_no_games_is_none(tmp_db):
    create_schema(tmp_db)
    assert _snap(tmp_db, "teamA", date(2025, 2, 1))["region"] is None


# --- roster, composite, synergy ---------------------------------------------------------


def test_roster_falls_back_to_the_rosters_table(tmp_db):
    create_schema(tmp_db)
    for role in _ROLES:
        insert_roster(tmp_db, "teamA", f"player_{role}", role, "2025-01-01")
    tmp_db.commit()

    roster = SnapshotStore(tmp_db, teams=["teamA"]).roster("teamA", date(2025, 3, 15))
    assert roster == {role: f"player_{role}" for role in _ROLES}


def test_roster_table_fallback_honours_end_dates(tmp_db):
    create_schema(tmp_db)
    insert_roster(tmp_db, "teamA", "old_top", "top", "2024-01-01", end_date="2025-02-01")
    insert_roster(tmp_db, "teamA", "new_top", "top", "2025-02-01")
    tmp_db.commit()

    roster = SnapshotStore(tmp_db, teams=["teamA"]).roster("teamA", date(2025, 3, 15))
    assert roster == {"top": "new_top"}


def test_roster_prefers_the_latest_observed_lineup(tmp_db):
    """A fielded five-role lineup overrides the rosters table's (duplicate) active rows."""
    create_schema(tmp_db)
    observed = {role: f"obs_{role}" for role in _ROLES}
    for role, player_id in observed.items():
        insert_roster(tmp_db, "teamA", player_id, role, "2025-01-01")
    for role in _ROLES:
        insert_roster(tmp_db, "teamA", f"stale_{role}", role, "2025-03-10")
    _insert_match(tmp_db, "m1", "2025-03-14 12:00:00")
    for role, player_id in observed.items():
        _insert_player_game(tmp_db, "m1", player_id, team_id="teamA", role=role)
    tmp_db.commit()

    store = SnapshotStore(tmp_db, teams=["teamA"])
    assert store.roster("teamA", date(2025, 3, 15)) == observed
    # Before the fielded game the table is all there is, and its latest start wins.
    assert store.roster("teamA", date(2025, 3, 14)) == {r: f"stale_{r}" for r in _ROLES}


def test_composite_is_role_weighted_over_the_roster_profiles(tmp_db, monkeypatch):
    create_schema(tmp_db)
    insert_roster(tmp_db, "teamA", "p_mid", "mid", "2025-01-01")
    insert_roster(tmp_db, "teamA", "p_sup", "sup", "2025-01-01")
    _insert_match(tmp_db, "m1", "2025-03-14")
    _insert_player_game(
        tmp_db, "m1", "p_mid", team_id="teamA", role="mid", golddiffat15=1000.0, dpm=800.0
    )
    _insert_player_game(
        tmp_db, "m1", "p_sup", team_id="teamA", role="sup", golddiffat15=0.0, dpm=200.0
    )
    tmp_db.commit()

    monkeypatch.setattr(config, "ROLE_WEIGHTS", None)
    unweighted = _snap(tmp_db, "teamA", date(2025, 3, 15))["composite"]
    assert set(unweighted) == {"avg_laning", "avg_teamfighting", "avg_vision"}
    assert unweighted["avg_laning"]["golddiffat15"] == pytest.approx(500.0)
    assert unweighted["avg_teamfighting"]["dpm"] == pytest.approx(500.0)

    monkeypatch.setattr(
        config, "ROLE_WEIGHTS", {"mid": 1.3, "jng": 1.3, "bot": 1.1, "top": 1.0, "sup": 0.7}
    )
    weighted = _snap(tmp_db, "teamA", date(2025, 3, 15))["composite"]
    assert weighted["avg_laning"]["golddiffat15"] == pytest.approx(650.0)  # 1300 / 2.0
    assert weighted["avg_teamfighting"]["dpm"] == pytest.approx(590.0)  # 1180 / 2.0


def test_composite_reads_only_series_started_before_the_query(tmp_db):
    """A same-day later series is invisible to a day query; an earlier one, to an instant."""
    create_schema(tmp_db)
    insert_roster(tmp_db, "teamA", "p_top", "top", "2025-01-01")
    _insert_match(tmp_db, "prior", "2025-03-14 12:00:00")
    _insert_player_game(tmp_db, "prior", "p_top", team_id="teamA", golddiffat15=100.0)
    _insert_match(tmp_db, "same_day", "2025-03-15 09:00:00")
    _insert_player_game(tmp_db, "same_day", "p_top", team_id="teamA", golddiffat15=900.0)
    _insert_match(tmp_db, "later", "2025-03-15 11:00:00")
    _insert_player_game(tmp_db, "later", "p_top", team_id="teamA", golddiffat15=1300.0)
    tmp_db.commit()
    store = SnapshotStore(tmp_db, teams=["teamA"])

    day = store.team_snapshot("teamA", date(2025, 3, 15))["composite"]
    assert day["avg_laning"]["golddiffat15"] == pytest.approx(100.0)
    instant = store.team_snapshot("teamA", datetime(2025, 3, 15, 10, 0, 0))["composite"]
    assert 100.0 < instant["avg_laning"]["golddiffat15"] < 900.0  # a recency-weighted blend
    assert store.team_snapshot("teamA", date(2025, 3, 14))["composite"] == {}


def test_composite_orders_a_career_by_series_start_not_by_series_id(tmp_db):
    """Two same-day series whose ids sort against their start times still bisect by start."""
    create_schema(tmp_db)
    insert_roster(tmp_db, "teamA", "p_top", "top", "2025-01-01")
    # 'ser_zzz' starts first but sorts last by id; 'ser_aaa' starts second.
    for sid, hour, gd in (("ser_zzz", "09", 100.0), ("ser_aaa", "18", 900.0)):
        tmp_db.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, status)"
            " VALUES (?, 'T', 'teamA', 'teamB', ?, 'completed')",
            (sid, f"2025-03-15T{hour}:00:00+00:00"),
        )
        tmp_db.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner) VALUES (?, ?, 1, 'teamA')",
            (f"g_{sid}", sid),
        )
        _insert_player_game(tmp_db, f"g_{sid}", "p_top", team_id="teamA", golddiffat15=gd)
    tmp_db.commit()

    at_second = AsOf.game("2025-03-15T18:00:00+00:00", "ser_aaa", 1)
    composite = _snap(tmp_db, "teamA", at_second)["composite"]
    assert composite["avg_laning"]["golddiffat15"] == pytest.approx(100.0)  # never its own game


def test_synergy_counts_games_the_listed_players_all_played(tmp_db):
    create_schema(tmp_db)
    for i in range(10):
        _insert_match(tmp_db, f"m{i}", f"2025-03-{i + 1:02d}", winner="teamA")
        _insert_player_game(tmp_db, f"m{i}", "p1", team_id="teamA", result=1, role="top")
        _insert_player_game(tmp_db, f"m{i}", "p2", team_id="teamA", result=1, role="jng")
    for role in ("top", "jng"):  # the roster fallback names p1 / p2
        insert_roster(tmp_db, "teamA", "p1" if role == "top" else "p2", role, "2025-01-01")
    tmp_db.commit()

    synergy = _snap(tmp_db, "teamA", date(2025, 3, 15))["synergy"]
    assert synergy["games_together"] == 10
    assert synergy["lineup_win_rate"] == pytest.approx(1.0)
    assert synergy["weighted_win_rate"] == pytest.approx(1.0)


def test_a_new_store_sees_rows_written_since_the_last(tmp_db):
    create_schema(tmp_db)
    insert_roster(tmp_db, "teamA", "p_top", "top", "2025-01-01")
    _insert_match(tmp_db, "m1", "2025-03-14")
    _insert_player_game(tmp_db, "m1", "p_top", team_id="teamA", golddiffat15=100.0)
    tmp_db.commit()
    first = _snap(tmp_db, "teamA", date(2025, 3, 15))["composite"]

    _insert_match(tmp_db, "m2", "2025-03-14 12:00:00")
    _insert_player_game(tmp_db, "m2", "p_top", team_id="teamA", golddiffat15=500.0)
    tmp_db.commit()
    second = _snap(tmp_db, "teamA", date(2025, 3, 15))["composite"]

    assert second["avg_laning"]["golddiffat15"] > first["avg_laning"]["golddiffat15"]


# --- draft tendency ---------------------------------------------------------------------


def test_draft_tendency_resolves_the_draft_scores_id_not_the_newest_model(rebuild_db):
    conn = rebuild_db
    conn.execute("INSERT INTO teams (team_id) VALUES ('A'), ('B')")
    conn.execute("INSERT INTO leagues (league_id, name) VALUES ('LCK', 'LCK')")
    insert_game(
        conn, game_id="g1", day="2026-01-10", team1="A", team2="B", winner="A", blue_team="A"
    )
    conn.commit()
    defaults = _snap(conn, "A", date(2026, 1, 20))["draft"]
    assert defaults["draft_score_avg"] == 0.5  # no draft_scores rows at all

    conn.execute("INSERT INTO models (model_id, kind) VALUES ('draft:rolledback_a', 'draft')")
    conn.execute("INSERT INTO models (model_id, kind) VALUES ('draft:superseded_b', 'draft')")
    conn.execute(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red) "
        "VALUES ('g1', 'draft:rolledback_a', 0.7, 'A', 'C')"
    )
    conn.commit()

    out = _snap(conn, "A", date(2026, 1, 20))["draft"]
    assert out["draft_score_avg"] == pytest.approx(0.7)  # A's scores, blue side == A
    assert out["draft_grade_avg"] == pytest.approx(4.0)
    assert out["draft_wr_5"] == 1.0


def test_draft_tendency_mixed_score_ids_raise_loudly(rebuild_db):
    """Two model_ids in ``draft_scores`` (a torn backfill) raise instead of picking one."""
    conn = rebuild_db
    conn.execute("INSERT INTO teams (team_id) VALUES ('A'), ('B')")
    conn.execute("INSERT INTO leagues (league_id, name) VALUES ('LCK', 'LCK')")
    insert_game(
        conn, game_id="g1", day="2026-01-10", team1="A", team2="B", winner="A", blue_team="A"
    )
    insert_game(
        conn, game_id="g2", day="2026-01-12", team1="A", team2="B", winner="B", blue_team="B"
    )
    for mid, gid in (("draft:test_v1", "g1"), ("draft:test_v2", "g2")):
        conn.execute("INSERT INTO models (model_id, kind) VALUES (?, 'draft')", (mid,))
        conn.execute(
            "INSERT INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red) "
            "VALUES (?, ?, 0.6, 'B', 'B')",
            (gid, mid),
        )
    conn.commit()

    with pytest.raises(RuntimeError, match="draft_scores"):
        SnapshotStore(conn, teams=["A"])


# --- the store's two loading modes agree -------------------------------------------------


def _mixed_db(path) -> sqlite3.Connection:
    """Three teams, stats, sides, a sub, a Bo3 and an international event."""
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    create_schema(conn)
    _setup_team(conn, "teamA")
    _setup_team(conn, "teamB")
    rng = random.Random(3)
    for i in range(12):
        mid = f"x{i}"
        a, b = ("teamA", "teamC") if i % 2 else ("teamB", "teamC")
        winner = a if rng.random() < 0.6 else b
        _insert_match(
            conn,
            mid,
            f"2025-03-{i + 6:02d}",
            team1=a,
            team2=b,
            winner=winner,
            blue_team=a if i % 3 else b,
        )
        for tid in (a, b):
            _insert_team_game_stats(
                conn, mid, tid, gamelength=1400 + 100 * i, totalgold=50000 + 1000 * i
            )
            for role in _ROLES:
                sub = tid == "teamC" and i > 8 and role == "sup"
                _insert_player_game(
                    conn,
                    mid,
                    "sub" if sub else f"{tid}_{role}",
                    team_id=tid,
                    result=int(winner == tid),
                    role=role,
                    champion_id=rng.choice(["Jax", "Ahri", "Lee", None]),
                    firstblood=int(rng.random() < 0.5),
                    golddiffat15=rng.choice([None, 100.0, -50.0]),
                )
    conn.execute("UPDATE series SET league_id = 'LCK'")
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, best_of, status)"
        " VALUES ('ser_bo3', 'MSI', 'teamA', 'teamB', '2025-03-20T11:00:00+00:00', 3, 'completed')"
    )
    for game_no, winner in ((1, "teamB"), (2, "teamA"), (3, "teamB")):
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner, blue_team)"
            " VALUES (?, 'ser_bo3', ?, ?, ?)",
            (f"bo3_{game_no}", game_no, winner, "teamA" if game_no != 2 else "teamB"),
        )
        for tid in ("teamA", "teamB"):
            for role in _ROLES:
                _insert_player_game(conn, f"bo3_{game_no}", f"{tid}_{role}", team_id=tid, role=role)
    conn.commit()
    return conn


def test_bulk_and_lazy_stores_answer_every_group_identically(tmp_path):
    conn = _mixed_db(tmp_path / "facts.db")
    bulk = SnapshotStore(conn)
    lazy = SnapshotStore(conn, teams=["teamA"])  # teamB / teamC load on first use
    points = [
        ("teamA", AsOf.of(date(2025, 3, 15))),
        ("teamB", AsOf.of(date(2025, 3, 15))),
        ("teamC", AsOf.of(date(2025, 3, 30))),
        ("teamA", AsOf.game("2025-03-20T11:00:00+00:00", "ser_bo3", 3)),
        ("teamB", AsOf.game("2025-03-20T11:00:00+00:00", "ser_bo3", 2)),
        ("teamA", AsOf.of(date(2025, 4, 30))),
        ("nobody", AsOf.of(date(2025, 4, 30))),
    ]
    for team, as_of in points:
        assert bulk.team_snapshot(team, as_of) == lazy.team_snapshot(team, as_of)
    for t1, t2 in (("teamA", "teamB"), ("teamB", "teamA"), ("teamC", "teamA")):
        as_of = AsOf.game("2025-03-20T11:00:00+00:00", "ser_bo3", 3)
        assert bulk.h2h(t1, t2, as_of) == lazy.h2h(t1, t2, as_of)
    # Non-vacuous: the fixture exercises the sub-carrying lineups and the international event.
    late = bulk.team_snapshot("teamC", AsOf.of(date(2025, 3, 30)))
    assert late["roster"]["sup"] == "sub"
    assert late["roster_stability"]["roster_total_games"] == 3
    assert late["champ_pool"]["unique_champs"] > 0
    assert bulk.team_snapshot("teamA", AsOf.of(date(2025, 4, 30)))["region"] == "LCK"  # not MSI
