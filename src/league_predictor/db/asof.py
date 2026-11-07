"""AS-OF reads: the two temporal rules — team_state strictly-before, impact on-or-before."""

# team_state stamped D prices only D+1 or later (strictly before); an impact score stamped D
# is readable ON D (<=, the writer stamps as_of - 1 day). The two comparisons differ on purpose.

from __future__ import annotations

import sqlite3
from datetime import date

from league_predictor.db.rows import TeamStateRow


def team_state_strength_asof_select(team_expr: str, event_date_expr: str) -> str:
    """Return strict-before strength SQL with one model-id placeholder."""
    return (
        "(SELECT ts.strength FROM team_state ts "
        f"WHERE ts.team_id = {team_expr} AND ts.model_id = ? "
        f"AND ts.asof_date < {event_date_expr} "
        "ORDER BY ts.asof_date DESC LIMIT 1)"
    )


def team_state_asof(
    conn: sqlite3.Connection,
    team_ids: list[str],
    model_id: str,
    event_date: date,
) -> dict[str, TeamStateRow]:
    """Latest ``team_state`` row per team with ``asof_date`` STRICTLY before ``event_date``.

    Returns:
        A dict keyed by ``team_id``; teams with no qualifying row are absent. One windowed
        query covers the whole batch, never a per-team loop.
    """
    if not team_ids:
        return {}

    placeholders = ", ".join("?" for _ in team_ids)
    query = f"""
        SELECT team_id, model_id, asof_date, strength
        FROM (
            SELECT team_id, model_id, asof_date, strength,
                   ROW_NUMBER() OVER (
                       PARTITION BY team_id ORDER BY asof_date DESC
                   ) AS rn
            FROM team_state
            WHERE model_id = ?
              AND asof_date < ?
              AND team_id IN ({placeholders})
        )
        WHERE rn = 1
    """
    params = [model_id, event_date.isoformat(), *team_ids]
    rows = conn.execute(query, params).fetchall()
    return {
        row["team_id"]: TeamStateRow(
            team_id=row["team_id"],
            model_id=row["model_id"],
            asof_date=row["asof_date"],
            strength=row["strength"],
        )
        for row in rows
    }


def impact_score_asof_scalar(column: str, player_expr: str) -> str:
    """Latest ``player_impact_scores`` row on or before the bound ``?`` date, for ``column``.

    Args:
        player_expr: SQL naming the player id to correlate on; always a fixed code literal.
    """
    return (
        f"(SELECT w.{column} FROM player_impact_scores w "
        f"WHERE w.player_id = {player_expr} AND w.model_id LIKE 'impact:%' "
        "AND w.asof_date <= ? ORDER BY w.asof_date DESC LIMIT 1)"
    )


def team_state_coverage(conn: sqlite3.Connection, model_id: str) -> tuple[int, str | None]:
    """(row count, max asof_date) for a model_id; max is None when the model has no rows."""
    row = conn.execute(
        "SELECT COUNT(*), MAX(asof_date) FROM team_state WHERE model_id = ?",
        (model_id,),
    ).fetchone()
    return int(row[0]), row[1]
