"""Fit the post-draft stack and rebuild ``postdraft_scores`` from it, in the caller's txn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from league_predictor.db.draft_scores_read import preferred_context_predicate
from league_predictor.db.predictions_read import (
    corpus_model_ids,
    latest_p_a_select,
    realized_p_a_select,
)
from league_predictor.postdraft.stack import GLOBAL_TEAM_ID, Stack, StackRow, fit_stack
from league_predictor.probability import logit, orient_blue

if TYPE_CHECKING:
    import sqlite3


#: Both rebuilt tables, deleted and rewritten together so a stack never outlives its scores.
_DERIVED_TABLES = ("postdraft_stack", "postdraft_scores")

#: Games both parents cover out-of-sample — the ONLY rows the stack fit ever sees.
_FIT_SQL = f"""
SELECT date(s.scheduled_ts) AS game_date, g.blue_team AS blue_team, g.red_team AS red_team,
       g.winner AS winner, s.team_a AS team_a, s.team_b AS team_b,
       ds.p_blue AS draft_p_blue,
       {latest_p_a_select("oos", "ds.game_id", model_param=":served_id")} AS served_p_a
FROM draft_scores ds
JOIN games g ON g.game_id = ds.game_id
JOIN series s ON s.series_id = g.series_id
WHERE ds.model_id = :draft_id AND ds.context = 'oos' AND ds.p_blue IS NOT NULL
  AND g.blue_team IS NOT NULL AND g.red_team IS NOT NULL AND g.winner IS NOT NULL
"""

#: Every game the stack can score: one preferred draft row and any realized served row.
_SCORE_SQL = f"""
SELECT ds.game_id AS game_id, g.blue_team AS blue_team, g.red_team AS red_team,
       s.team_a AS team_a, s.team_b AS team_b, ds.p_blue AS draft_p_blue,
       {realized_p_a_select("ds.game_id", model_param=":served_id")} AS served_p_a
FROM draft_scores ds
JOIN games g ON g.game_id = ds.game_id
JOIN series s ON s.series_id = g.series_id
WHERE ds.model_id = :draft_id AND ds.p_blue IS NOT NULL AND g.blue_team IS NOT NULL
  AND {preferred_context_predicate()}
"""


def _fit_rows(conn: sqlite3.Connection, served_id: str, draft_id: str) -> list[StackRow]:
    """The oos fit corpus: both parents out-of-sample, blue-oriented, with blue's outcome."""
    rows = conn.execute(_FIT_SQL, {"served_id": served_id, "draft_id": draft_id}).fetchall()
    fit_rows = []
    for row in rows:
        if row["served_p_a"] is None:
            continue
        served_blue = orient_blue(float(row["served_p_a"]), row["blue_team"], *_sides(row))
        if served_blue is None:
            continue  # blue side not one of the series teams — unstackable, omit
        fit_rows.append(
            StackRow(
                date=str(row["game_date"]),
                l_served=logit(served_blue),
                l_draft=logit(float(row["draft_p_blue"])),
                blue_team=row["blue_team"],
                red_team=row["red_team"],
                blue_won=row["winner"] == row["blue_team"],
            )
        )
    return fit_rows


def _sides(row: sqlite3.Row) -> tuple[str, str]:
    """The series' canonical (team_a, team_b) pair, the orientation reference."""
    return row["team_a"], row["team_b"]


def _write_stack(conn: sqlite3.Connection, stack: Stack, served_id: str, draft_id: str) -> None:
    """Replace ``postdraft_stack``: the global coefficients row, then one row per team."""
    conn.execute(
        "INSERT INTO postdraft_stack (team_id, a, beta, u, n_oos, penalty, holdout_gain, "
        "served_model_id, draft_model_id) VALUES (?, ?, ?, 0.0, ?, ?, ?, ?, ?)",
        (
            GLOBAL_TEAM_ID,
            stack.a,
            stack.beta,
            stack.n_rows,
            stack.penalty,
            stack.holdout_gain,
            served_id,
            draft_id,
        ),
    )
    conn.executemany(
        "INSERT INTO postdraft_stack (team_id, u, n_oos, served_model_id, draft_model_id) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            (team, stack.u.get(team, 0.0), n, served_id, draft_id)
            for team, n in sorted(stack.n_oos.items())
        ],
    )


def rebuild(
    conn: sqlite3.Connection,
    *,
    served_model_id: str | None = None,
    draft_model_id: str | None = None,
) -> int:
    """Fit the stack on the oos corpus, write it, then rescore every game; never commits.

    Returns:
        The number of ``postdraft_scores`` rows written. The served parent is
        ``COALESCE(oos, backfill)`` for the served id, the draft parent its preferred
        context row; a game missing either gets no row. Each id defaults to the id
        currently owning its parent prod table.

    Raises:
        RuntimeError: both parents exist but their oos corpora do not overlap, so the
            stack cannot be fit; nothing is written. An ABSENT parent is not that case —
            no game could carry a row anyway, so both tables are simply emptied.
    """
    served_id = served_model_id or corpus_served_model_id(conn)
    draft_id = draft_model_id or current_draft_model_id(conn)
    if served_id is None or draft_id is None:
        for table in _DERIVED_TABLES:
            conn.execute(f"DELETE FROM {table}")
        return 0
    stack = fit_stack(_fit_rows(conn, served_id, draft_id))

    rows = conn.execute(_SCORE_SQL, {"served_id": served_id, "draft_id": draft_id}).fetchall()
    inserts: list[tuple] = []
    for row in rows:
        if row["served_p_a"] is None:
            continue
        served_blue = orient_blue(float(row["served_p_a"]), row["blue_team"], *_sides(row))
        if served_blue is None:
            continue  # blue side not one of the series teams — unstackable, omit
        p_blue = stack.p_blue(
            logit(served_blue),
            logit(float(row["draft_p_blue"])),
            row["blue_team"],
            row["red_team"],
        )
        inserts.append((row["game_id"], p_blue, served_id, draft_id))

    for table in _DERIVED_TABLES:
        conn.execute(f"DELETE FROM {table}")
    _write_stack(conn, stack, served_id, draft_id)
    conn.executemany(
        "INSERT INTO postdraft_scores (game_id, p_blue, served_model_id, draft_model_id) "
        "VALUES (?, ?, ?, ?)",
        inserts,
    )
    return len(inserts)


def rebuild_and_verify(
    conn: sqlite3.Connection,
    *,
    served_model_id: str | None = None,
    draft_model_id: str | None = None,
) -> int:
    """``rebuild`` then ``assert_fresh`` under one resolved id pair; ``rebuild``'s row count."""
    served_id = served_model_id or corpus_served_model_id(conn)
    draft_id = draft_model_id or current_draft_model_id(conn)
    n = rebuild(conn, served_model_id=served_id, draft_model_id=draft_id)
    assert_fresh(conn, served_model_id=served_id, draft_model_id=draft_id)
    return n


def assert_fresh(
    conn: sqlite3.Connection,
    *,
    served_model_id: str | None = None,
    draft_model_id: str | None = None,
) -> None:
    """Assert both derived tables carry only the current served+draft id pair.

    Raises:
        RuntimeError: a row's stamps differ. Each id defaults to the id owning its
            parent prod table; an empty table passes vacuously.
    """
    served_id = served_model_id or corpus_served_model_id(conn) or ""
    draft_id = draft_model_id or current_draft_model_id(conn) or ""
    for table in _DERIVED_TABLES:
        stale = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE served_model_id != ? OR draft_model_id != ?",
            (served_id, draft_id),
        ).fetchone()[0]
        if stale:
            raise RuntimeError(
                f"{stale} {table} rows do not carry the current model pair "
                f"(served={served_id}, draft={draft_id}); re-run the owning served/draft "
                "retrain or backfill."
            )


# The public parent-id resolvers: current ids read from the prod tables the stack
# composes (postdraft imports neither lifecycle nor predraft).


def corpus_served_model_id(conn: sqlite3.Connection) -> str | None:
    """The corpus-owning served id: None when empty, RuntimeError on several."""
    return _one_id("served", corpus_model_ids(conn, id_prefix="served:"))


def current_draft_model_id(conn: sqlite3.Connection) -> str | None:
    """The draft id owning ``draft_scores``: None when empty, RuntimeError on several."""
    ids = [r[0] for r in conn.execute("SELECT DISTINCT model_id FROM draft_scores")]
    return _one_id("draft", ids)


def _one_id(kind: str, ids: list[str]) -> str | None:
    """Exactly-one selector: None for empty, the id for one, a loud raise for several."""
    if len(ids) > 1:
        raise RuntimeError(
            f"the {kind} parent table carries multiple ids {sorted(ids)} — cannot resolve a "
            "single current model for the post-draft rebuild; pass the id explicitly."
        )
    return ids[0] if ids else None
