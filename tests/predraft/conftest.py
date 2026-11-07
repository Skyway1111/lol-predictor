"""Shared synthetic facts-DB factories for the predraft feature/model tests."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

import pytest

from league_predictor.db.schema import apply_current_schema

_ROLES = ("top", "jng", "mid", "bot", "sup")
_DEFAULT_SKILL = {"AAA": 3.0, "BBB": 1.0, "CCC": -1.0, "DDD": -3.0}


def ensure_event_type_column(conn: sqlite3.Connection) -> None:
    """Guarantee ``games.event_type`` exists for hand-rolled fixtures."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(games)")}
    if "event_type" not in cols:
        conn.execute("ALTER TABLE games ADD COLUMN event_type TEXT")


def insert_game(
    conn,
    *,
    game_id: str,
    day: str | date,
    team1: str,
    team2: str,
    winner: str | None,
    tournament: str = "LCK",
    game_no: int = 1,
    best_of: int = 3,
    duration_s: int | None = 1800,
    blue_team: str | None = None,
    patch: str | None = None,
    series_id: str | None = None,
    time: str = "12:00:00",
    event_type: str | None = None,
) -> str:
    """Insert one canonically oriented map into ``series_id`` or its own series; returns it."""
    team_a, team_b = sorted((team1, team2))
    sid = series_id or f"ser_{game_id}"
    ts = f"{day}T{time}+00:00"
    conn.execute(
        "INSERT OR IGNORE INTO series (series_id, league_id, team_a, team_b, "
        "scheduled_ts, best_of, winner, status) VALUES (?, ?, ?, ?, ?, ?, NULL, 'completed')",
        (sid, tournament, team_a, team_b, ts, best_of),
    )
    red = None
    if blue_team is not None:
        red = team2 if blue_team == team1 else team1
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, "
        "winner, patch, duration_s) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (game_id, sid, game_no, blue_team, red, winner, patch, duration_s),
    )
    if event_type is not None:
        ensure_event_type_column(conn)
        conn.execute("UPDATE games SET event_type = ? WHERE game_id = ?", (event_type, game_id))
    return sid


def _generate(conn, *, start: date, end: date, seed: int, skill: dict, per_day: int) -> None:
    rng = random.Random(seed)
    teams = list(skill)
    n = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
    d = start
    while d < end:
        for _ in range(per_day):
            a, b = rng.sample(teams, 2)
            p_a = 1.0 / (1.0 + 10 ** (-(skill[a] - skill[b]) / 4.0))
            winner = a if rng.random() < p_a else b
            mid = f"m{seed}_{n}"
            n += 1
            insert_game(
                conn,
                game_id=mid,
                day=d.isoformat(),
                team1=a,
                team2=b,
                winner=winner,
                series_id=f"s{seed}_{n:06d}",
            )
            for tid in (a, b):
                for role in _ROLES:
                    conn.execute(
                        "INSERT INTO game_players (game_id, team_id, player_id, role, "
                        "kills, deaths, assists) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (mid, tid, f"{tid}_{role}", role, 5, 5, 10),
                    )
        d += timedelta(days=1)
    conn.commit()


# Compact toy-history tuple format: 11-col match, 22-col player, in OE box-score
# column order.
_MATCH_TUPLE_COLS = (
    "match_id",
    "date",
    "team1_id",
    "team2_id",
    "winner",
    "tournament",
    "gamelength",
    "game_number",
    "patch",
    "blue_side_team",
    "stage",
)
_PG_TUPLE_COLS = (
    "match_id",
    "team_id",
    "player_id",
    "role",
    "kills",
    "deaths",
    "assists",
    "golddiffat10",
    "csdiffat10",
    "xpdiffat10",
    "golddiffat15",
    "csdiffat15",
    "xpdiffat15",
    "damageshare",
    "vspm",
    "earnedgoldshare",
    "teamkills",
    "teamdeaths",
    "firsttower",
    "firstdragon",
    "firstbaron",
    "side",
)


def facts_db_from_legacy(matches, player_games) -> sqlite3.Connection:
    """In-memory facts DB seeded from compact 11-col match / 22-col player tuples.

    Returns:
        Connection where each map is its own series ``ser_<match_id>``; canonical facts
        order (day, series_id, game_no) matches the tuples' ``(date, match_id)`` order.
        ``stage`` and ``side`` have no facts column and are dropped.
    """
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    for m in matches:
        row = dict(zip(_MATCH_TUPLE_COLS, m, strict=True))
        day, time = str(row["date"])[:10], str(row["date"])[11:] or "10:00:00"
        team_a, team_b = sorted((row["team1_id"], row["team2_id"]))
        blue = row["blue_side_team"]
        red = None
        if blue is not None:
            red = row["team2_id"] if blue == row["team1_id"] else row["team1_id"]
        conn.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
            "best_of, status) VALUES (?, ?, ?, ?, ?, NULL, 'completed')",
            (f"ser_{row['match_id']}", row["tournament"], team_a, team_b, f"{day}T{time}+00:00"),
        )
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, "
            "winner, patch, duration_s) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row["match_id"],
                f"ser_{row['match_id']}",
                1 if row["game_number"] is None else int(row["game_number"]),
                blue,
                red,
                row["winner"],
                row["patch"],
                row["gamelength"],
            ),
        )
    pg_cols = [c for c in _PG_TUPLE_COLS if c != "side"]
    for p in player_games:
        row = dict(zip(_PG_TUPLE_COLS, p, strict=True))
        conn.execute(
            f"INSERT INTO game_players (game_id, team_id, player_id, "
            f"{', '.join(pg_cols[3:])}) VALUES ({', '.join('?' * (len(pg_cols)))})",
            (row["match_id"], row["team_id"], row["player_id"], *(row[c] for c in pg_cols[3:])),
        )
    conn.commit()
    return conn


def facts_seed(conn: sqlite3.Connection, matches, player_games) -> None:
    """Facts seed from (mid, date, t1, t2, winner, league, dur) and (mid, tid, k, d, a) tuples."""
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    for mid, dt, t1, t2, winner, league, dur in matches:
        ta, tb = sorted((t1, t2))
        ts = (
            str(dt).replace(" ", "T", 1) if len(str(dt)) > 10 else str(dt) + "T10:00:00"
        ) + "+00:00"
        conn.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
            "status) VALUES (?, ?, ?, ?, ?, 'completed')",
            (f"ser_{mid}", league, ta, tb, ts),
        )
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
            "VALUES (?, ?, 1, ?, ?)",
            (mid, f"ser_{mid}", winner, dur),
        )
    for i, (mid, tid, k, d, a) in enumerate(player_games):
        conn.execute(
            "INSERT INTO game_players (game_id, team_id, player_id, kills, deaths, "
            "assists) VALUES (?, ?, ?, ?, ?, ?)",
            (mid, tid, f"p{i}", k, d, a),
        )
    conn.commit()


def insert_roster(
    conn: sqlite3.Connection, team_id, player_id, role, start_date, end_date=None
) -> None:
    """Insert one ``rosters`` row (the roster fallback table, not a facts table)."""
    conn.execute(
        "CREATE TABLE IF NOT EXISTS rosters (team_id TEXT, player_id TEXT, "
        "role TEXT, start_date TEXT, end_date TEXT)"
    )
    conn.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date, end_date) "
        "VALUES (?, ?, ?, ?, ?)",
        (team_id, player_id, role, start_date, end_date),
    )


def register_draft_model(
    conn: sqlite3.Connection, model_id: str = "draft:test_grader", artifact_sha: str = "0" * 64
) -> str:
    """Register a synthetic ``draft:`` models row so score-write FKs hold; returns model_id."""
    conn.execute(
        "INSERT OR IGNORE INTO models (model_id, kind, artifact_sha) VALUES (?, 'draft', ?)",
        (model_id, artifact_sha),
    )
    conn.commit()
    return model_id


@pytest.fixture
def skill_db():
    """Factory for in-memory skill-driven facts; ``poison=(s,e)`` adds a later leakage window."""

    def make(
        *,
        start: date,
        end: date,
        seed: int = 0,
        skill: dict | None = None,
        per_day: int = 2,
        poison: tuple[date, date] | None = None,
    ) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_current_schema(conn)
        sk = skill or dict(_DEFAULT_SKILL)
        _generate(conn, start=start, end=end, seed=seed, skill=sk, per_day=per_day)
        if poison is not None:
            _generate(conn, start=poison[0], end=poison[1], seed=seed + 999, skill=sk, per_day=10)
        return conn

    return make
