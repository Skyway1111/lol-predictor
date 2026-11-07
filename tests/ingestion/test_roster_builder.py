"""Tests for ingestion-owned roster derivation from played-game facts."""

import sqlite3

from league_predictor.ingestion.roster_builder import (
    build_rosters_from_player_games,
)


def _add_game(
    conn: sqlite3.Connection, game_id: str, date: str, lineups: dict[str, dict[str, str]]
) -> None:
    """Insert a series + game + game_players; ``lineups`` is {team_id: {role: player_id}}."""
    teams = list(lineups)
    team1 = teams[0]
    team2 = teams[1] if len(teams) > 1 else "OPP"
    team_a, team_b = sorted((team1, team2))
    sid = f"s_{game_id}"
    conn.execute(
        "INSERT OR IGNORE INTO series (series_id, team_a, team_b, scheduled_ts, "
        "status, winner) VALUES (?, ?, ?, ?, 'completed', ?)",
        (sid, team_a, team_b, date, team1),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner) "
        "VALUES (?, ?, 1, ?, ?, ?)",
        (game_id, sid, team1, team2, team1),
    )
    for team_id, lineup in lineups.items():
        for role, player_id in lineup.items():
            conn.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, role, result) "
                "VALUES (?, ?, ?, ?, 1)",
                (game_id, player_id, team_id, role),
            )
    conn.commit()


def _roster_rows(conn: sqlite3.Connection) -> list[tuple]:
    return [
        tuple(row)
        for row in conn.execute(
            "SELECT team_id, player_id, role, start_date, end_date FROM rosters "
            "ORDER BY player_id, start_date"
        ).fetchall()
    ]


def _active(conn: sqlite3.Connection, team: str, player: str, role: str, as_of: str) -> bool:
    """Roster-active: ``start_date <= as_of AND (end_date IS NULL OR end_date > as_of)``."""
    return (
        conn.execute(
            "SELECT 1 FROM rosters WHERE team_id=? AND player_id=? AND role=? "
            "AND start_date <= ? AND (end_date IS NULL OR end_date > ?)",
            (team, player, role, as_of, as_of),
        ).fetchone()
        is not None
    )


def test_build_rosters_from_player_games(schema_db):
    """build_rosters_from_player_games derives rosters from the game facts."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"top": "Zeus", "mid": "Faker"}})

    count = build_rosters_from_player_games(schema_db)
    assert count == 2

    rows = schema_db.execute(
        "SELECT team_id, player_id, role, start_date FROM rosters ORDER BY player_id"
    ).fetchall()

    assert len(rows) == 2
    assert rows[0]["player_id"] == "Faker"
    assert rows[0]["role"] == "mid"
    assert rows[0]["start_date"] == "2025-03-01"
    assert rows[1]["player_id"] == "Zeus"
    assert rows[1]["role"] == "top"


def test_build_rosters_from_player_games_no_data(schema_db):
    """build_rosters_from_player_games returns 0 when no game_players exist."""
    count = build_rosters_from_player_games(schema_db)
    assert count == 0


def test_resync_is_a_no_op_for_unchanged_lineups(schema_db):
    """Re-running after new games with the same lineup leaves the open row untouched."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"top": "Zeus", "mid": "Faker"}})
    assert build_rosters_from_player_games(schema_db) == 2

    _add_game(schema_db, "m2", "2025-03-08", {"T1": {"top": "Zeus", "mid": "Faker"}})
    assert build_rosters_from_player_games(schema_db) == 0

    assert _roster_rows(schema_db) == [
        ("T1", "Faker", "mid", "2025-03-01", None),
        ("T1", "Zeus", "top", "2025-03-01", None),
    ]


def test_departure_is_closed_half_open_at_successor_boundary(schema_db):
    """A superseded seat closes at the successor's first game date (half-open)."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    build_rosters_from_player_games(schema_db)

    _add_game(schema_db, "m2", "2025-04-01", {"T1": {"mid": "NewGuy"}})
    assert build_rosters_from_player_games(schema_db) == 1

    assert _roster_rows(schema_db) == [
        ("T1", "Faker", "mid", "2025-03-01", "2025-04-01"),
        ("T1", "NewGuy", "mid", "2025-04-01", None),
    ]
    # Faker is active on the date of his own last game; not on the successor's date.
    assert _active(schema_db, "T1", "Faker", "mid", "2025-03-01")
    assert not _active(schema_db, "T1", "Faker", "mid", "2025-04-01")
    assert _active(schema_db, "T1", "NewGuy", "mid", "2025-04-01")


def test_fresh_build_closes_historical_assignments(schema_db):
    """A single build over accumulated history closes vacated seats at the successor boundary."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    _add_game(schema_db, "m2", "2025-04-01", {"T1": {"mid": "NewGuy"}})

    build_rosters_from_player_games(schema_db)

    assert _roster_rows(schema_db) == [
        ("T1", "Faker", "mid", "2025-03-01", "2025-04-01"),
        ("T1", "NewGuy", "mid", "2025-04-01", None),
    ]


def test_rejoin_opens_a_second_stint(schema_db):
    """A player returning to a seat gets a new row starting at their first game after the gap."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    build_rosters_from_player_games(schema_db)

    _add_game(schema_db, "m2", "2025-04-01", {"T1": {"mid": "Sub"}})
    build_rosters_from_player_games(schema_db)

    _add_game(schema_db, "m3", "2025-05-01", {"T1": {"mid": "Faker"}})
    assert build_rosters_from_player_games(schema_db) == 1

    assert _roster_rows(schema_db) == [
        ("T1", "Faker", "mid", "2025-03-01", "2025-04-01"),
        ("T1", "Faker", "mid", "2025-05-01", None),
        ("T1", "Sub", "mid", "2025-04-01", "2025-05-01"),
    ]


def test_player_active_on_every_date_they_played_the_seat(schema_db):
    """A player is roster-active on every date they played the seat, and not during an absence."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    _add_game(schema_db, "m2", "2025-03-15", {"T1": {"mid": "Faker"}})
    _add_game(schema_db, "m3", "2025-04-01", {"T1": {"mid": "Sub"}})
    _add_game(schema_db, "m4", "2025-04-15", {"T1": {"mid": "Sub"}})
    _add_game(schema_db, "m5", "2025-05-01", {"T1": {"mid": "Faker"}})

    build_rosters_from_player_games(schema_db)

    # Faker active on each of his played dates, including his last-before-gap game.
    for d in ("2025-03-01", "2025-03-15", "2025-05-01"):
        assert _active(schema_db, "T1", "Faker", "mid", d), d
    # Faker NOT credited during Sub's tenure (the mid-tenure absence).
    for d in ("2025-04-01", "2025-04-15"):
        assert not _active(schema_db, "T1", "Faker", "mid", d), d
        assert _active(schema_db, "T1", "Sub", "mid", d), d
    # Sub not active once Faker returns.
    assert not _active(schema_db, "T1", "Sub", "mid", "2025-05-01")


def test_idempotent_across_a_leave_and_return(schema_db):
    """Re-running on unchanged data after a leave-and-return inserts nothing and mutates no row."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    _add_game(schema_db, "m2", "2025-04-01", {"T1": {"mid": "Sub"}})
    _add_game(schema_db, "m3", "2025-05-01", {"T1": {"mid": "Faker"}})
    build_rosters_from_player_games(schema_db)
    before = _roster_rows(schema_db)

    assert build_rosters_from_player_games(schema_db) == 0
    assert _roster_rows(schema_db) == before


def test_rebuild_heals_old_collapsed_window(schema_db):
    """A collapsed window spanning a gap is reconciled toward the derived half-open stints."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    _add_game(schema_db, "m2", "2025-04-01", {"T1": {"mid": "Sub"}})
    _add_game(schema_db, "m3", "2025-05-01", {"T1": {"mid": "Faker"}})
    # Seed the damaged state: Faker's stints collapsed into one open interval spanning
    # Sub's tenure, and Sub closed inclusively at his last game.
    schema_db.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date, end_date) "
        "VALUES ('T1', 'Faker', 'mid', '2025-03-01', NULL)"
    )
    schema_db.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date, end_date) "
        "VALUES ('T1', 'Sub', 'mid', '2025-04-01', '2025-04-01')"
    )
    schema_db.commit()

    build_rosters_from_player_games(schema_db)

    assert _roster_rows(schema_db) == [
        ("T1", "Faker", "mid", "2025-03-01", "2025-04-01"),
        ("T1", "Faker", "mid", "2025-05-01", None),
        ("T1", "Sub", "mid", "2025-04-01", "2025-05-01"),
    ]
    assert not _active(schema_db, "T1", "Faker", "mid", "2025-04-01")
    assert _active(schema_db, "T1", "Sub", "mid", "2025-04-01")
    # A second build is a no-op once healed.
    assert build_rosters_from_player_games(schema_db) == 0


def test_rebuild_heals_mis_closed_departure(schema_db):
    """A departure closed inclusively at the last game is re-closed at the successor boundary."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    _add_game(schema_db, "m2", "2025-04-01", {"T1": {"mid": "NewGuy"}})
    schema_db.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date, end_date) "
        "VALUES ('T1', 'Faker', 'mid', '2025-03-01', '2025-03-01')"
    )
    schema_db.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date, end_date) "
        "VALUES ('T1', 'NewGuy', 'mid', '2025-04-01', NULL)"
    )
    schema_db.commit()

    build_rosters_from_player_games(schema_db)

    assert _roster_rows(schema_db) == [
        ("T1", "Faker", "mid", "2025-03-01", "2025-04-01"),
        ("T1", "NewGuy", "mid", "2025-04-01", None),
    ]
    assert _active(schema_db, "T1", "Faker", "mid", "2025-03-01")


def test_rebuild_prunes_evidence_backed_orphan_row(schema_db):
    """An evidence-backed row whose start_date is not a derived stint boundary is pruned."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    _add_game(schema_db, "m2", "2025-03-08", {"T1": {"mid": "Faker"}})
    # A spurious extra row for a player who DOES have game evidence, at a
    # start_date no single-run stint supports.
    schema_db.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date, end_date) "
        "VALUES ('T1', 'Faker', 'mid', '2025-03-05', NULL)"
    )
    schema_db.commit()

    build_rosters_from_player_games(schema_db)

    assert _roster_rows(schema_db) == [
        ("T1", "Faker", "mid", "2025-03-01", None),
    ]


def test_rows_without_game_evidence_stay_open(schema_db):
    """Rows with no game evidence stay open: absence of games is not evidence of departure."""
    schema_db.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date) "
        "VALUES ('T1', 'Ghost', 'mid', '2025-01-01')"
    )
    schema_db.commit()
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"top": "Zeus"}})

    build_rosters_from_player_games(schema_db)

    assert _roster_rows(schema_db) == [
        ("T1", "Ghost", "mid", "2025-01-01", None),
        ("T1", "Zeus", "top", "2025-03-01", None),
    ]


def test_teams_are_independent(schema_db):
    """One team's newer games never close another team's assignments."""
    _add_game(schema_db, "m1", "2025-03-01", {"T1": {"mid": "Faker"}})
    build_rosters_from_player_games(schema_db)

    _add_game(schema_db, "m2", "2025-04-01", {"GenG": {"mid": "Chovy"}})
    build_rosters_from_player_games(schema_db)

    assert _roster_rows(schema_db) == [
        ("GenG", "Chovy", "mid", "2025-04-01", None),
        ("T1", "Faker", "mid", "2025-03-01", None),
    ]
