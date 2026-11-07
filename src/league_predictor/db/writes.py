"""Consolidated writers for tables with multiple production callers; none of them commit."""

# Dates are 'YYYY-MM-DD' and timestamps UTC ISO-8601 '+00:00': lexicographic ordering is
# load-bearing, and a derived row's asof_date is INCLUSIVE of games through end of day D.

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING

from league_predictor.db.rows import (
    DraftActionRow,
    GamePlayerRow,
    GameRow,
    GameTeamStatRow,
    PredictionRow,
    SeriesRow,
    TeamStateRow,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable


def append_predictions(conn: sqlite3.Connection, rows: Iterable[PredictionRow]) -> None:
    """Batch INSERT into the append-only ``predictions`` log.

    Raises:
        sqlite3.IntegrityError: Repeated natural key
            ``(subject_kind, subject_id, model_id, context, created_at)``; never updates.
    """
    conn.executemany(
        "INSERT INTO predictions (subject_kind, subject_id, model_id, p_a, "
        "asof_date, context, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            (
                r.subject_kind,
                r.subject_id,
                r.model_id,
                r.p_a,
                r.asof_date,
                r.context,
                r.created_at,
            )
            for r in rows
        ],
    )


def replace_team_state(
    conn: sqlite3.Connection, model_id: str, rows: Iterable[TeamStateRow]
) -> None:
    """Replace this model's ``team_state`` coverage wholesale; other models' rows untouched."""
    conn.execute("DELETE FROM team_state WHERE model_id = ?", (model_id,))
    conn.executemany(
        "INSERT INTO team_state (team_id, model_id, asof_date, strength) VALUES (?, ?, ?, ?)",
        [(r.team_id, r.model_id, r.asof_date, r.strength) for r in rows],
    )


def drop_foreign_team_state(conn: sqlite3.Connection, keep_model_id: str) -> int:
    """Enforce ``team_state``'s single-owner contract: delete rows under any other id."""
    return conn.execute("DELETE FROM team_state WHERE model_id != ?", (keep_model_id,)).rowcount


_SERIES_COLUMNS = tuple(f.name for f in fields(SeriesRow))

_GAME_COLUMNS = tuple(f.name for f in fields(GameRow))

_GAME_PLAYER_COLUMNS = tuple(f.name for f in fields(GamePlayerRow))

_DRAFT_ACTION_COLUMNS = tuple(f.name for f in fields(DraftActionRow))


# Series statuses that end a series' lifecycle: a non-terminal write must never walk one
# back, enforced by _upsert_on_conflict's terminal_status clause.
TERMINAL_SERIES_STATUSES = ("completed", "canceled")


def _upsert_on_conflict(
    conn: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    conflict_col: str,
    *,
    row: object,
    preserve: tuple[str, ...] = (),
    terminal_status: tuple[str, ...] = (),
) -> None:
    """INSERT ... ON CONFLICT(conflict_col) DO UPDATE for a single-column key.

    Args:
        row: The complete current fact; every column is overwritten with the incoming value.
        preserve: Columns that COALESCE instead, so an incoming None never wipes a known
            value (None means "don't know", never "erase").
        terminal_status: Makes ``status`` MONOTONIC — once the stored status is one of
            these, an incoming None or non-terminal value keeps it; terminal to terminal
            still applies.
    """
    # DO UPDATE, not INSERT OR REPLACE: replace would delete-then-reinsert the parent
    # row and take its child rows with it.
    cols = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)

    def _clause(c: str) -> str:
        if c == "status" and terminal_status:
            terms = ", ".join(f"'{t}'" for t in terminal_status)
            return (
                f"status = CASE WHEN {table}.status IN ({terms}) "
                f"AND (excluded.status IS NULL OR excluded.status NOT IN ({terms})) "
                f"THEN {table}.status "
                f"ELSE COALESCE(excluded.status, {table}.status) END"
            )
        if c in preserve:
            return f"{c} = COALESCE(excluded.{c}, {table}.{c})"
        return f"{c} = excluded.{c}"

    updates = ", ".join(_clause(c) for c in columns if c != conflict_col)
    conn.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders}) "
        f"ON CONFLICT({conflict_col}) DO UPDATE SET {updates}",
        tuple(getattr(row, c) for c in columns),
    )


def upsert_series(conn: sqlite3.Connection, row: SeriesRow) -> None:
    """Idempotent upsert of a ``series`` fact by ``series_id``.

    Args:
        row: ``winner``/``scheduled_ts`` preserve — None never erases a known value, so a
            partial re-read cannot wipe an outcome or start time; ``status`` is monotonic
            over :data:`TERMINAL_SERIES_STATUSES`.
    """
    _upsert_on_conflict(
        conn,
        "series",
        _SERIES_COLUMNS,
        "series_id",
        row=row,
        preserve=("winner", "scheduled_ts"),
        terminal_status=TERMINAL_SERIES_STATUSES,
    )


def upsert_game(conn: sqlite3.Connection, row: GameRow) -> None:
    """Idempotent upsert of a ``games`` fact by ``game_id``."""
    _upsert_on_conflict(conn, "games", _GAME_COLUMNS, "game_id", row=row)


def _replace_leaf_rows(
    conn: sqlite3.Connection, table: str, columns: tuple[str, ...], rows: Iterable[object]
) -> None:
    """Batch INSERT OR REPLACE keyed on the table's natural key.

    Args:
        table: A LEAF table only — replace deletes then reinserts, taking any child rows
            with it. Parent facts go through :func:`_upsert_on_conflict` instead.
    """
    cols = ", ".join(columns)
    placeholders = ", ".join("?" for _ in columns)
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})",
        [tuple(getattr(r, c) for c in columns) for r in rows],
    )


def upsert_game_players(conn: sqlite3.Connection, rows: list[GamePlayerRow]) -> None:
    """Idempotent upsert of ``game_players`` box scores by (game_id, player_id)."""
    _replace_leaf_rows(conn, "game_players", _GAME_PLAYER_COLUMNS, rows)


def upsert_draft_actions(conn: sqlite3.Connection, rows: list[DraftActionRow]) -> None:
    """Idempotent upsert of ``draft_actions`` by (game_id, seq)."""
    _replace_leaf_rows(conn, "draft_actions", _DRAFT_ACTION_COLUMNS, rows)


_GAME_TEAM_STAT_COLUMNS = tuple(f.name for f in fields(GameTeamStatRow))


def upsert_game_team_stats(conn: sqlite3.Connection, rows: list[GameTeamStatRow]) -> None:
    """Idempotent upsert of ``game_team_stats`` facts by (game_id, team_id)."""
    _replace_leaf_rows(conn, "game_team_stats", _GAME_TEAM_STAT_COLUMNS, rows)
