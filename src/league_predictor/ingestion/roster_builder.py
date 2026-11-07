"""Roster derivation from the played-game facts (game_players)."""

import sqlite3
from itertools import groupby


def _derive_stints(conn: sqlite3.Connection) -> list[tuple]:
    """Derive the half-open ``[start, end)`` seat-holder stints from game facts.

    Returns:
        ``(team_id, player_id, role, start_date, end_date)`` per maximal single-player run
        on a ``(team, role)`` seat, games ordered by appearance date
        (``series.scheduled_ts``, day-truncated) then ``game_id``. Each non-final stint
        closes at its successor's first game date; the final stint stays open (NULL end).
    """
    rows = conn.execute(
        """
        SELECT pg.team_id, pg.role,
               substr(s.scheduled_ts, 1, 10) AS day,
               pg.game_id, pg.player_id
        FROM game_players pg
        JOIN games g ON g.game_id = pg.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE pg.role IS NOT NULL
        ORDER BY pg.team_id, pg.role, day, pg.game_id
        """
    ).fetchall()

    stints: list[tuple] = []
    for (team_id, role), seat_games in groupby(rows, key=lambda r: (r[0], r[1])):
        runs: list[tuple] = []  # (player_id, first_day) per maximal single-player run
        for _, _, day, _game_id, player_id in seat_games:
            if not runs or runs[-1][0] != player_id:
                runs.append((player_id, day))
        for i, (player_id, start_date) in enumerate(runs):
            end_date = runs[i + 1][1] if i + 1 < len(runs) else None
            stints.append((team_id, player_id, role, start_date, end_date))
    return stints


def build_rosters_from_player_games(conn: sqlite3.Connection) -> int:
    """Reconcile ``rosters`` toward the ``_derive_stints`` windows from played games.

    Returns:
        Number of roster rows inserted. Sole maintainer of the evidence-backed rows:
        inserts new stints, heals drifted ``end_date`` boundaries, prunes evidence-backed
        rows at a ``start_date`` no derived stint supports. Rows with no ``game_players``
        evidence are left alone -- absence of games is not evidence of departure. A re-run
        on unchanged data writes nothing (idempotent).
    """
    stints = _derive_stints(conn)

    conn.execute("DROP TABLE IF EXISTS _desired_stints")
    try:
        conn.execute(
            "CREATE TEMP TABLE _desired_stints ("
            "team_id TEXT, player_id TEXT, role TEXT, start_date TEXT, end_date TEXT)"
        )
        conn.executemany("INSERT INTO _desired_stints VALUES (?, ?, ?, ?, ?)", stints)
        conn.execute(
            "CREATE INDEX _desired_stints_key "
            "ON _desired_stints(team_id, player_id, role, start_date)"
        )

        # New stints: no roster row at this (team, player, role, start_date).
        inserted = conn.execute(
            """
            INSERT OR IGNORE INTO rosters (team_id, player_id, role, start_date, end_date)
            SELECT d.team_id, d.player_id, d.role, d.start_date, d.end_date
            FROM _desired_stints d
            WHERE NOT EXISTS (
                SELECT 1 FROM rosters r
                WHERE r.team_id = d.team_id AND r.player_id = d.player_id
                  AND r.role = d.role AND r.start_date = d.start_date
            )
            """
        ).rowcount

        # Heal drifted boundaries: matching (team, player, role, start_date) rows
        # whose end_date differs from the derived stint (NULL-safe compare).
        conn.execute(
            """
            UPDATE rosters SET end_date = (
                SELECT d.end_date FROM _desired_stints d
                WHERE d.team_id = rosters.team_id AND d.player_id = rosters.player_id
                  AND d.role = rosters.role AND d.start_date = rosters.start_date
            )
            WHERE EXISTS (
                SELECT 1 FROM _desired_stints d
                WHERE d.team_id = rosters.team_id AND d.player_id = rosters.player_id
                  AND d.role = rosters.role AND d.start_date = rosters.start_date
                  AND d.end_date IS NOT rosters.end_date
            )
            """
        )

        # Prune orphans: evidence-backed (team, player, role) rows whose start_date is no
        # derived stint boundary. Evidence-less seeded rows never match, so they are kept.
        conn.execute(
            """
            DELETE FROM rosters
            WHERE EXISTS (
                SELECT 1 FROM _desired_stints d
                WHERE d.team_id = rosters.team_id AND d.player_id = rosters.player_id
                  AND d.role = rosters.role
            )
            AND NOT EXISTS (
                SELECT 1 FROM _desired_stints d
                WHERE d.team_id = rosters.team_id AND d.player_id = rosters.player_id
                  AND d.role = rosters.role AND d.start_date = rosters.start_date
            )
            """
        )

        conn.commit()
        return inserted
    finally:
        conn.execute("DROP TABLE IF EXISTS _desired_stints")
