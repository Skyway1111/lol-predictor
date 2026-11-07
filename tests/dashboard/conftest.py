"""Shared client + seed builders for the dashboard API tests: the games and terminal fixtures."""

import shutil
import sqlite3
from datetime import date

from fastapi.testclient import TestClient

from league_predictor.analytics.builders import refresh_analytics_marts
from league_predictor.dashboard.app_factory import create_app
from league_predictor.predraft.pipeline.model_identity import current_rating_model_id
from tests.predraft.conftest import insert_game

# Fixed window that always contains the 2026 fixtures regardless of the clock.
RANGE = "start_date=2026-01-01&end_date=2026-12-31"

_ROLES = ("top", "jng", "mid", "bot", "sup")

# Marts anchor: every terminal-fixture snapshot is computed as of this date.
_TERMINAL_AS_OF = date(2026, 3, 20)


# Games fixture: hand-built facts DB for the games/champions/tournaments/predict endpoints.
def _players_for(conn, game_id, team_id, champs, result):
    """Insert one team's five box-score lines for a game (role-ordered champs)."""
    for role, champ in zip(_ROLES, champs, strict=True):
        firstblood = 1 if (role == "top" and result == 1) else 0
        conn.execute(
            "INSERT INTO game_players (game_id, player_id, team_id, role, champion_id, "
            "result, kills, deaths, assists, dpm, cspm, golddiffat15, csdiffat15, "
            "xpdiffat15, killparticipation, damageshare, firstblood) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                game_id,
                f"oe:player:{team_id}_{role}",
                team_id,
                role,
                champ,
                result,
                4,
                2,
                6,
                450.0,
                8.5,
                120.0,
                3.0,
                90.0,
                0.6,
                0.24,
                firstblood,
            ),
        )


def _team_stats(conn, game_id, team_id, won):
    conn.execute(
        "INSERT INTO game_team_stats (game_id, team_id, dragons, barons, towers, heralds, "
        "firstdragon, firstbaron, firsttower, firstherald, totalgold, totalcs, teamkills, "
        "teamdeaths, gamelength, void_grubs) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            game_id,
            team_id,
            3 if won else 1,
            1 if won else 0,
            8 if won else 3,
            1,
            1 if won else 0,
            1 if won else 0,
            1 if won else 0,
            1,
            58000 if won else 52000,
            900,
            14 if won else 8,
            8 if won else 14,
            1980,
            4,
        ),
    )


def seed_games_fixture(conn: sqlite3.Connection) -> None:
    """Seed the shared games fixture into a fresh canonical-schema DB."""
    t1, t2, t3, t4 = "oe:team:t1", "oe:team:t2", "oe:team:t3", "oe:team:t4"

    # g1: LCK, blue=t1 wins. team_a==t1 (string order), so blue==team_a.
    insert_game(
        conn,
        game_id="g1",
        day="2026-03-10",
        team1=t1,
        team2=t2,
        winner=t1,
        tournament="LCK",
        best_of=3,
        blue_team=t1,
        patch="15.1",
    )
    # g2: LCK, blue=t2, but t1 wins. No draft_actions -> tests the game_players
    # pick fallback path.
    insert_game(
        conn,
        game_id="g2",
        day="2026-03-12",
        team1=t2,
        team2=t1,
        winner=t1,
        tournament="LCK",
        best_of=3,
        blue_team=t2,
        patch="15.2",
    )
    # g3: LFL (non-major) -> dropped by major_only default.
    insert_game(
        conn,
        game_id="g3",
        day="2026-03-11",
        team1=t3,
        team2=t4,
        winner=t3,
        tournament="LFL",
        best_of=3,
        blue_team=t3,
        patch="15.1",
    )

    # Champion picks. Ahri: t1 mid both LCK games (2 picks, 2 wins).
    _players_for(conn, "g1", t1, ["Aatrox", "LeeSin", "Ahri", "Kaisa", "Nautilus"], 1)
    _players_for(conn, "g1", t2, ["Gnar", "Vi", "Orianna", "Jinx", "Rakan"], 0)
    _players_for(conn, "g2", t1, ["Jax", "Sejuani", "Ahri", "Ezreal", "Thresh"], 1)
    _players_for(conn, "g2", t2, ["Ornn", "Maokai", "Azir", "Varus", "Braum"], 0)
    _players_for(conn, "g3", t3, ["Sion", "Nocturne", "Sylas", "Zeri", "Lulu"], 1)
    _players_for(conn, "g3", t4, ["Renekton", "Elise", "Viktor", "Aphelios", "Karma"], 0)

    for gid, blue, red, blue_won in (
        ("g1", t1, t2, True),
        ("g2", t2, t1, False),
        ("g3", t3, t4, True),
    ):
        _team_stats(conn, gid, blue, blue_won)
        _team_stats(conn, gid, red, not blue_won)

    # g1 draft: full pick order + a Zed ban (never picked -> ban-only champion).
    da = [
        ("g1", 1, "blue", "ban", "Zed", None),
        ("g1", 2, "red", "ban", "Yasuo", None),
        ("g1", 7, "blue", "pick", "Ahri", "mid"),
        ("g1", 8, "red", "pick", "Orianna", "mid"),
        ("g1", 9, "blue", "pick", "Aatrox", "top"),
        ("g1", 10, "red", "pick", "Gnar", "top"),
    ]
    rating_id = current_rating_model_id()
    conn.executemany(
        "INSERT INTO draft_actions (game_id, seq, side, kind, champion_id, role) "
        "VALUES (?,?,?,?,?,?)",
        da,
    )

    conn.execute(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red, edge_pp_blue) "
        "VALUES ('g1', 'draft:test', 0.6, 'A', 'C', 4.2)",
    )
    # Post-draft fusion for g1 (both parents present); g2 has neither -> dash.
    conn.execute(
        "INSERT INTO postdraft_scores (game_id, p_blue, served_model_id, draft_model_id) "
        "VALUES ('g1', 0.66, 'served:abc', 'draft:test')",
    )
    conn.executemany(
        "INSERT INTO pick_scores (game_id, side, role, model_id, player_id, "
        "actual_score, alt_count, best_score, worst_score, percentile) VALUES "
        "(?,?,?,?,?,?,?,?,?,?)",
        [("g1", "blue", "mid", "draft:test", "oe:player:oe:team:t1_mid", 0.8, 10, 0.9, 0.1, 0.75)],
    )

    # As-of ratings for g1 (2026-03-10). The earlier t1 state must win; the later
    # one (after the game) must never leak.
    conn.executemany(
        "INSERT INTO team_state (team_id, model_id, asof_date, strength) VALUES (?,?,?,?)",
        [
            (t1, rating_id, "2026-03-01", 0.5),  # used
            (t1, rating_id, "2026-03-15", 9.9),  # leaks if the guard is wrong
            (t2, rating_id, "2026-03-05", 0.0),
            (t1, "rating:obsolete", "2026-03-09", 99.0),
        ],
    )

    # Predictions: an OOS row (must surface) and a backfill row (must NOT) for g1.
    conn.executemany(
        "INSERT INTO predictions (subject_kind, subject_id, model_id, p_a, asof_date, "
        "context, created_at) VALUES (?,?,?,?,?,?,?)",
        [
            ("game", "g1", "served:abc", 0.7, "2026-03-09", "oos", "2026-03-09T00:00:00"),
            ("game", "g1", "served:abc", 0.99, "2026-03-09", "backfill", "2026-03-09T01:00:00"),
        ],
    )

    conn.executemany(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES (?,?)",
        [(t1, "T1"), (t2, "Gen.G"), (t3, "Karmine Corp"), (t4, "Vitality")],
    )
    conn.executemany(
        "INSERT INTO player_aliases (player_id, source, source_name) VALUES (?,?,?)",
        [
            (f"oe:player:{t}_{r}", "oe", f"{t[-2:]}-{r}")
            for t in ("oe:team:t1", "oe:team:t2")
            for r in _ROLES
        ],
    )
    conn.executemany(
        "INSERT INTO leagues (league_id, name) VALUES (?,?)",
        [("LCK", "LoL Champions Korea"), ("LFL", "LFL")],
    )

    # Impact: one player has scores, another does not (dash).
    conn.executemany(
        "INSERT INTO player_impact_scores (player_id, model_id, asof_date, role, impact) "
        "VALUES (?,?,?,?,?)",
        [
            ("oe:player:oe:team:t1_mid", "impact:z", "2026-01-01", "mid", 1.0),
            ("oe:player:oe:team:t1_mid", "impact:z", "2026-06-01", "mid", 2.5),
        ],
    )
    conn.commit()


def make_games_client(tmp_path, db_template_path) -> TestClient:
    """TestClient over a fresh DB carrying the games fixture."""
    db = tmp_path / "games_api.db"
    shutil.copy(db_template_path, db)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    seed_games_fixture(conn)
    conn.close()
    return TestClient(create_app(db_path=db))


# Terminal fixture: completed maps refreshed through the marts, for players/teams endpoints.
def seed_terminal_data(conn: sqlite3.Connection) -> None:
    """Seed facts + team-name vocabulary for four completed LCK/LEC maps, one series each."""
    # match_id, date, tournament, stage, patch, format, team1, team2, winner,
    # blue_side_team, game_number, gamelength
    matches = [
        (
            "m_lck_1",
            "2026-03-01",
            "LCK",
            "regular",
            "15.4",
            "Bo3",
            "oe:team:t1",
            "oe:team:hle",
            "oe:team:t1",
            "oe:team:t1",
            1,
            2100,
        ),
        (
            "m_lec_1",
            "2026-03-02",
            "LEC",
            "regular",
            "15.4",
            "Bo3",
            "oe:team:g2",
            "oe:team:fnc",
            "oe:team:g2",
            "oe:team:g2",
            1,
            1980,
        ),
        (
            "m_lck_2",
            "2026-03-10",
            "LCK",
            "regular",
            "15.5",
            "Bo3",
            "oe:team:t1",
            "oe:team:g2",
            "oe:team:t1",
            "oe:team:t1",
            1,
            2040,
        ),
        (
            "m_lck_3",
            "2026-03-15",
            "LCK",
            "regular",
            "15.5",
            "Bo3",
            "oe:team:g2",
            "oe:team:t1",
            "oe:team:t1",
            "oe:team:g2",
            1,
            2010,
        ),
    ]
    for (
        match_id,
        day,
        tournament,
        stage,
        patch,
        fmt,
        team1,
        team2,
        winner,
        blue,
        game_number,
        gamelength,
    ) in matches:
        insert_game(
            conn,
            game_id=match_id,
            day=day,
            team1=team1,
            team2=team2,
            winner=winner,
            tournament=tournament,
            game_no=game_number,
            best_of=int(fmt[2:]),
            duration_s=gamelength,
            blue_team=blue,
            patch=patch,
            event_type=stage,
        )
    conn.executemany(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES (?, ?)",
        [
            ("oe:team:t1", "T1"),
            ("oe:team:g2", "G2 Esports"),
            ("oe:team:hle", "Hanwha Life Esports"),
            ("oe:team:fnc", "Fnatic"),
        ],
    )
    players = {
        "oe:team:t1": [
            ("top", "zeus"),
            ("jng", "oner"),
            ("mid", "faker"),
            ("bot", "gumayusi"),
            ("sup", "keria"),
        ],
        "oe:team:g2": [
            ("top", "bb"),
            ("jng", "yike"),
            ("mid", "caps"),
            ("bot", "hans"),
            ("sup", "mikyx"),
        ],
        "oe:team:hle": [
            ("top", "doran"),
            ("jng", "peanut"),
            ("mid", "zeka"),
            ("bot", "viper"),
            ("sup", "delight"),
        ],
        "oe:team:fnc": [
            ("top", "oscar"),
            ("jng", "razork"),
            ("mid", "humanoid"),
            ("bot", "noah"),
            ("sup", "jun"),
        ],
    }
    player_rows = []
    tgs_rows = []
    for match_id, _, _, _, _, _, team1, team2, winner, _, _, gamelength in matches:
        for team_id, kill_base, gold_base in ((team1, 14, 5300), (team2, 10, 4900)):
            tgs_rows.append(
                (
                    match_id,
                    team_id,
                    2 if team_id == winner else 1,
                    1 if team_id == winner else 0,
                    8 if team_id == winner else 5,
                    1,
                    1 if team_id == winner else 0,
                    1 if team_id == winner else 0,
                    1 if team_id == winner else 0,
                    1,
                    55000 if team_id == winner else 50000,
                    1500,
                    kill_base,
                    8 if team_id == winner else 14,
                    gamelength,
                )
            )
            for idx, (role, handle) in enumerate(players[team_id], start=1):
                player_id = f"oe:player:{handle}"
                result = 1 if team_id == winner else 0
                player_rows.append(
                    (
                        match_id,
                        player_id,
                        team_id,
                        f"Champ{idx}",
                        role,
                        result,
                        3 + result,
                        1 + (0 if result else 1),
                        6,
                        kill_base,
                        8 if team_id == winner else 14,
                        0.20 + idx * 0.01,
                        0.55,
                        1.5 + idx * 0.05,
                        2,
                        # Laning diffs vary by win/loss so each role's baseline has real
                        # stdev; a constant diff would z-score to +/-inf on this micro-DB.
                        gold_base + idx * 50 + (300 if result else -300),
                        10 + idx + (4 if result else -4),
                        250 + idx * 10 + (40 if result else -40),
                        610 + idx * 20,
                        7.5 + idx * 0.1,
                        0.22 + idx * 0.01,
                        1 if idx == 1 and result else 0,
                        1 if idx == 1 and result else 0,
                        1 if idx == 1 and result else 0,
                        1 if idx == 1 and result else 0,
                    )
                )
                conn.execute(
                    "INSERT OR IGNORE INTO player_aliases (player_id, source, source_name)"
                    " VALUES (?, 'oe', ?)",
                    (player_id, handle),
                )
    conn.executemany(
        """
        INSERT INTO game_team_stats (
            game_id, team_id, dragons, barons, towers, heralds,
            firstdragon, firstbaron, firsttower, firstherald,
            totalgold, totalcs, teamkills, teamdeaths, gamelength
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        tgs_rows,
    )
    conn.executemany(
        """
        INSERT INTO game_players (
            game_id, player_id, team_id, champion_id, role, result,
            kills, deaths, assists, teamkills, teamdeaths, damageshare,
            killparticipation, vspm, controlwardsbought, golddiffat15,
            csdiffat15, xpdiffat15, dpm, cspm, earnedgoldshare,
            firstblood, firsttower, firstdragon, firstbaron
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        player_rows,
    )
    conn.commit()


def make_terminal_client(tmp_path, db_template_path) -> TestClient:
    """TestClient over a fresh DB seeded with the terminal fixture + refreshed marts."""
    db_path = tmp_path / "terminal_api.db"
    shutil.copy(db_template_path, db_path)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    seed_terminal_data(conn)
    refresh_analytics_marts(conn, as_of_date=_TERMINAL_AS_OF)
    conn.close()
    return TestClient(create_app(db_path=db_path))
