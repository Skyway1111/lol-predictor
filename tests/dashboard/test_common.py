"""Shared dashboard filter SQL."""

import sqlite3

from league_predictor.dashboard.common import (
    active_player_ids_sql,
    active_team_ids_sql,
    parse_league_csv,
)


def test_parse_league_csv_trims_and_drops_blanks():
    """The one home for the leagues CSV: trim, drop empties, preserve order."""
    assert parse_league_csv("LCK, LEC ,,") == ["LCK", "LEC"]
    assert parse_league_csv(None) == []
    assert parse_league_csv("") == []
    assert parse_league_csv("   ") == []


def _seed_series(conn):
    conn.execute(
        "CREATE TABLE series (series_id TEXT, league_id TEXT, team_a TEXT, "
        "team_b TEXT, scheduled_ts TEXT)"
    )
    conn.execute("CREATE TABLE games (game_id TEXT, series_id TEXT)")
    conn.execute("CREATE TABLE game_players (game_id TEXT, player_id TEXT)")
    conn.executemany(
        "INSERT INTO series VALUES (?, ?, ?, ?, ?)",
        [
            ("s1", "LCK", "a", "b", "2026-03-01T00:00:00"),
            ("s2", "LEC", "c", "d", "2026-03-02T00:00:00"),
        ],
    )
    conn.executemany("INSERT INTO games VALUES (?, ?)", [("g1", "s1"), ("g2", "s2")])
    conn.executemany(
        "INSERT INTO game_players VALUES (?, ?)",
        [("g1", "pa"), ("g1", "pb"), ("g2", "pc"), ("g2", "pd")],
    )
    conn.commit()


def test_active_team_ids_sql_scopes_by_league():
    """Passing league_ids restricts activity to those leagues' series."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_series(conn)
    sql, params = active_team_ids_sql("2026-01-01", "2026-12-31", league_ids=["LEC"])
    ids = {r[0] for r in conn.execute(f"SELECT team_id FROM {sql}", params)}
    conn.close()
    assert ids == {"c", "d"}


def test_active_player_ids_sql_scopes_by_league():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_series(conn)
    sql, params = active_player_ids_sql("2026-01-01", "2026-12-31", league_ids=["LCK"])
    ids = {r[0] for r in conn.execute(f"SELECT player_id FROM {sql}", params)}
    conn.close()
    assert ids == {"pa", "pb"}


def test_active_ids_sql_no_league_filter_is_unchanged():
    """No league_ids → all teams/players in the window (back-compat)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_series(conn)
    tsql, tparams = active_team_ids_sql("2026-01-01", "2026-12-31")
    teams = {r[0] for r in conn.execute(f"SELECT team_id FROM {tsql}", tparams)}
    psql, pparams = active_player_ids_sql("2026-01-01", "2026-12-31")
    players = {r[0] for r in conn.execute(f"SELECT player_id FROM {psql}", pparams)}
    conn.close()
    assert teams == {"a", "b", "c", "d"}
    assert players == {"pa", "pb", "pc", "pd"}
