"""The one SQL home for predictions-log reads: ``oos`` wins, ``backfill`` fills history."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from league_predictor.db.schema import PREDICTION_CONTEXTS

if TYPE_CHECKING:
    import sqlite3


def _require_context(context: str) -> None:
    if context not in PREDICTION_CONTEXTS:
        raise ValueError(f"context {context!r} is not one of {PREDICTION_CONTEXTS}")


def _require_id_prefix(id_prefix: str) -> None:
    """An id-kind prefix is a fixed ``<token>:`` code literal, never runtime input."""
    if not re.fullmatch(r"[a-z_]+:", id_prefix):
        raise ValueError(f"id_prefix {id_prefix!r} is not a '<model>:' id-kind literal")


def context_row_count(conn: sqlite3.Connection, model_id: str, context: str) -> int:
    """The number of ``predictions`` rows one ``(model_id, context)`` pair holds."""
    _require_context(context)
    row = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE model_id = ? AND context = ?",
        (model_id, context),
    ).fetchone()
    return int(row[0])


def latest_p_a_select(context: str, game_expr: str, *, model_param: str) -> str:
    """Scalar SQL: the latest ``p_a`` game row of ``context`` under a bound model id.

    Args:
        game_expr: SQL naming the game id to correlate on; always a fixed code literal.
        model_param: the model-id binding as a named parameter (e.g. ``:served_id``).
    """
    _require_context(context)
    return (
        "(SELECT p.p_a FROM predictions p WHERE p.subject_kind = 'game' "
        f"AND p.subject_id = {game_expr} AND p.context = '{context}' "
        f"AND p.model_id = {model_param} ORDER BY p.created_at DESC LIMIT 1)"
    )


def realized_p_a_select(game_expr: str, *, model_param: str) -> str:
    """Scalar SQL: a model's realized ``p_a`` for a played game — ``oos``, else ``backfill``."""
    oos = latest_p_a_select("oos", game_expr, model_param=model_param)
    backfill = latest_p_a_select("backfill", game_expr, model_param=model_param)
    return f"COALESCE({oos}, {backfill})"


def latest_p_a_prefix_select(context: str, game_expr: str, *, id_prefix: str) -> str:
    """Scalar SQL: the latest ``p_a`` of ``context`` for ``game_expr`` under an id-kind prefix.

    Args:
        id_prefix: a fixed ``"<token>:"`` code literal. Prefix reads serve display surfaces
            that follow whichever id is current; writers always pin exact ids.
    """
    _require_context(context)
    _require_id_prefix(id_prefix)
    return (
        "(SELECT p.p_a FROM predictions p WHERE p.subject_kind = 'game' "
        f"AND p.subject_id = {game_expr} AND p.context = '{context}' "
        f"AND p.model_id LIKE '{id_prefix}%' ORDER BY p.created_at DESC LIMIT 1)"
    )


def latest_rows_by_context(
    conn: sqlite3.Connection, game_id: str, *, id_prefix: str
) -> dict[str, sqlite3.Row]:
    """The latest ``(p_a, asof_date)`` row per context for one game under ``id_prefix``."""
    _require_id_prefix(id_prefix)
    placeholders = ", ".join("?" for _ in PREDICTION_CONTEXTS)
    rows = conn.execute(
        f"""
        SELECT context, p_a, asof_date FROM (
            SELECT context, p_a, asof_date,
                   ROW_NUMBER() OVER (PARTITION BY context ORDER BY created_at DESC) AS rn
            FROM predictions
            WHERE subject_kind = 'game' AND subject_id = ? AND model_id LIKE ?
              AND context IN ({placeholders})
        ) WHERE rn = 1
        """,
        (game_id, f"{id_prefix}%", *PREDICTION_CONTEXTS),
    ).fetchall()
    return {row["context"]: row for row in rows}


def oos_corpus_edge(conn: sqlite3.Connection, *, id_prefix: str) -> str | None:
    """Last date the walk-forward ``oos`` corpus covers under ``id_prefix`` (MAX ``asof_date``).

    Returns:
        The edge date, or None; games dated past it postdate the model's training data.
    """
    _require_id_prefix(id_prefix)
    row = conn.execute(
        "SELECT MAX(asof_date) FROM predictions "
        "WHERE subject_kind = 'game' AND context = 'oos' AND model_id LIKE ?",
        (f"{id_prefix}%",),
    ).fetchone()
    return row[0]


def realized_prediction(
    oos_p_a: float | None,
    backfill_p_a: float | None,
    *,
    game_date: str,
    oos_edge: str | None,
) -> tuple[float | None, str | None]:
    """The walk/live rule: ``oos`` wins; only a past-edge ``backfill`` is ``'live'``."""
    if oos_p_a is not None:
        return oos_p_a, "walk"
    if oos_edge is not None and game_date > oos_edge and backfill_p_a is not None:
        return backfill_p_a, "live"
    return None, None


def oos_blue_probs(conn: sqlite3.Connection, model_id: str) -> tuple[dict[str, float], int]:
    """Blue-oriented walk-forward ``oos`` priors for draft-scorable games, pinned model id.

    Returns:
        ``({game_id: p_blue}, omitted)`` over games with a known blue side and a winner;
        a scorable game with no ``oos`` row is counted in ``omitted``, never substituted.
    """
    # The CASE is the SQL twin of probability.orient_blue (SQL cannot call the Python).
    rows = conn.execute(
        "SELECT g.game_id AS game_id, "
        "  CASE WHEN g.blue_team = s.team_a THEN p.p_a "
        "       WHEN g.blue_team = s.team_b THEN 1.0 - p.p_a END AS p_blue "
        "FROM games g "
        "JOIN series s ON s.series_id = g.series_id "
        "LEFT JOIN predictions p "
        "  ON p.subject_id = g.game_id AND p.subject_kind = 'game' "
        "  AND p.context = 'oos' AND p.model_id = ? "
        "WHERE g.blue_team IS NOT NULL AND g.winner IS NOT NULL",
        (model_id,),
    ).fetchall()
    probs: dict[str, float] = {}
    omitted = 0
    for row in rows:
        if row["p_blue"] is None:
            omitted += 1
        else:
            probs[row["game_id"]] = float(row["p_blue"])
    return probs, omitted


def corpus_model_ids(conn: sqlite3.Connection, *, id_prefix: str) -> list[str]:
    """DISTINCT model ids holding corpus rows (any context) under ``id_prefix``."""
    placeholders = ", ".join("?" for _ in PREDICTION_CONTEXTS)
    return [
        row[0]
        for row in conn.execute(
            f"SELECT DISTINCT model_id FROM predictions WHERE model_id LIKE ? "
            f"AND context IN ({placeholders})",
            (f"{id_prefix}%", *PREDICTION_CONTEXTS),
        )
    ]
