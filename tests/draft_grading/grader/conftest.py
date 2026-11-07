"""Shared schema and seeded-DB factory for the draft grader test suite."""

from __future__ import annotations

import sqlite3

# Constraint-free (no FK / CHECK) mirror of the rebuilt facts tables the model
# reads; a game's date is ``date(series.scheduled_ts)``.
GRADER_TEST_SCHEMA = """
    CREATE TABLE series (
        series_id TEXT PRIMARY KEY, league_id TEXT, team_a TEXT, team_b TEXT,
        scheduled_ts TEXT, best_of INTEGER, winner TEXT, status TEXT
    );
    CREATE TABLE games (
        game_id TEXT PRIMARY KEY, series_id TEXT, game_no INTEGER DEFAULT 1,
        blue_team TEXT, red_team TEXT, winner TEXT, patch TEXT, duration_s INTEGER
    );
    CREATE TABLE game_players (
        game_id TEXT, player_id TEXT, team_id TEXT, role TEXT, champion_id TEXT,
        result INTEGER, golddiffat15 REAL, firstdragon INTEGER, firsttower INTEGER,
        firstbaron INTEGER, firstblood INTEGER, dpm REAL, damageshare REAL,
        PRIMARY KEY (game_id, player_id)
    );
    CREATE TABLE draft_actions (
        game_id TEXT, seq INTEGER, side TEXT, kind TEXT, champion_id TEXT, role TEXT,
        PRIMARY KEY (game_id, seq)
    );
    CREATE TABLE impact_history (
        player_id TEXT NOT NULL, asof_date TEXT NOT NULL, impact_logodds REAL NOT NULL,
        PRIMARY KEY (player_id, asof_date)
    );
"""

_ROLES = ("top", "jng", "mid", "bot", "sup")
_BLUE_CHAMPS = ("Gnar", "LeeSin", "Ahri", "Jinx", "Thresh")
_RED_CHAMPS = ("Ksante", "Viego", "Syndra", "Caitlyn", "Lulu")


def make_grader_test_db(game_count: int = 0) -> sqlite3.Connection:
    """In-memory DB on the shared grader schema, seeded with ``game_count`` games.

    Args:
        game_count: Games of the blue-vs-red pattern (5 roles per side,
            deterministic winners) to seed over June 2025; 0 leaves schema only.
    """
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row  # bulk_load reads rows by column name
    db.executescript(GRADER_TEST_SCHEMA)

    for g in range(game_count):
        game_id = f"game_{g:03d}"
        date = f"2025-06-{(g % 28) + 1:02d}"
        scheduled_ts = f"{date}T00:00:00+00:00"
        winner = "blue_team" if g % 3 != 0 else "red_team"
        series_id = f"ser_{game_id}"
        db.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, "
            "scheduled_ts, winner, status) VALUES (?, 'LCK', 'blue_team', "
            "'red_team', ?, ?, 'completed')",
            (series_id, scheduled_ts, winner),
        )
        db.execute(
            "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, "
            "winner) VALUES (?, ?, 1, 'blue_team', 'red_team', ?)",
            (game_id, series_id, winner),
        )
        for i, (role, bc, rc) in enumerate(zip(_ROLES, _BLUE_CHAMPS, _RED_CHAMPS, strict=True)):
            result_b = 1 if winner == "blue_team" else 0
            db.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, "
                "champion_id, role, result) VALUES (?, ?, 'blue_team', ?, ?, ?)",
                (game_id, f"bp{i}", bc, role, result_b),
            )
            db.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, "
                "champion_id, role, result) VALUES (?, ?, 'red_team', ?, ?, ?)",
                (game_id, f"rp{i}", rc, role, 1 - result_b),
            )
    db.commit()
    return db
