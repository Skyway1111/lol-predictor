"""Version-aware whole-draft grading: the ``draft_scores`` writer (PK ``game_id, model_id``)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from league_predictor.db.rebuild_guard import require_rebuild_schema
from league_predictor.draft_grading.state import DraftState

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping
    from datetime import date

    from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
    from league_predictor.draft_grading.grader.predictor import DraftScore
    from league_predictor.draft_grading.live import DraftGrader

logger = logging.getLogger(__name__)

BATCH_SIZE = 5000

_REQUIRED_TABLES = ("models", "series", "games", "game_players", "draft_actions", "draft_scores")

#: Ceiling on the fraction of ATTEMPTED predictions one scoring pass may lose to predictor
#: failures before it is a systematic fault; shared by the whole-draft and per-pick scorers.
SCORING_ERROR_BUDGET = 0.01


def enforce_error_budget(counts: Mapping[str, int], *, scope: str) -> None:
    """Raise :class:`RuntimeError` when predictor failures exceed the budget.

    Args:
        counts: Rate is over ATTEMPTED predictions (``scored + skipped_predict_error``);
            ``skipped_no_draft``/``skipped_degenerate`` sit outside both numerator and
            denominator, and a pass attempting nothing is never a breach.

    Raises:
        RuntimeError: raised before the caller commits, so the pass rolls back.
    """
    errors = counts.get("skipped_predict_error", 0)
    attempted = counts.get("scored", 0) + errors
    if attempted and errors > SCORING_ERROR_BUDGET * attempted:
        raise RuntimeError(
            f"{scope}: {errors}/{attempted} attempted predictions failed "
            f"({100.0 * errors / attempted:.1f}%), over the {100.0 * SCORING_ERROR_BUDGET:.1f}% "
            "budget — a systematic predictor/feature fault, not isolated bad rows. The pass's "
            "rows are NOT committed. Investigate the grader before re-running (a corrupt "
            "artifact, a schema/feature drift, or a stale matchup store are the usual causes)."
        )


# Scorable: ten resolved pick actions, both sides known, and a winner (draft_scores
# feed the predraft tendency features off completed games only).
_SCORABLE_GAMES = """
    FROM games g
    JOIN series s ON s.series_id = g.series_id
    WHERE s.scheduled_ts >= ?
      AND g.winner IS NOT NULL
      AND g.blue_team IS NOT NULL AND g.red_team IS NOT NULL
      AND (SELECT COUNT(*) FROM draft_actions da
           WHERE da.game_id = g.game_id AND da.kind = 'pick'
             AND da.role IS NOT NULL AND da.champion_id IS NOT NULL) = 10
"""

_ORDER_BY = " ORDER BY s.scheduled_ts, g.game_id"


@dataclass(frozen=True)
class DraftScope:
    """The window a (re)score pass covers: date floor plus an optional game cap."""

    start: date
    max_games: int | None = None


def score_new_drafts(
    conn: sqlite3.Connection,
    predictor: DraftGrader,
    model_id: str,
    scope: DraftScope,
    *,
    stats_cache: BulkStatsCache | None = None,
) -> dict[str, int]:
    """Score every scorable game with NO ``draft_scores`` row under ``model_id``.

    Returns:
        Per-outcome counters.
    """
    require_rebuild_schema(conn, *_REQUIRED_TABLES, purpose="draft scoring")
    game_ids = _missing_draft_games(conn, model_id, scope)
    if scope.max_games is not None:
        game_ids = game_ids[: scope.max_games]
    return _process(conn, predictor, model_id, game_ids, stats_cache, delete_stale=False)


def rescore_stale_drafts(
    conn: sqlite3.Connection,
    predictor: DraftGrader,
    model_id: str,
    scope: DraftScope,
    *,
    stats_cache: BulkStatsCache | None = None,
    force: bool = False,
) -> int:
    """Rescore games whose ``draft_scores`` sit under another model; ``force`` rescores any.

    Returns:
        Rows written.
    """
    require_rebuild_schema(conn, *_REQUIRED_TABLES, purpose="draft rescore")
    game_ids = _stale_draft_games(conn, model_id, scope, force=force)
    if scope.max_games is not None:
        game_ids = game_ids[: scope.max_games]
    counts = _process(conn, predictor, model_id, game_ids, stats_cache, delete_stale=True)
    return counts["scored"]


def _missing_draft_games(conn: sqlite3.Connection, model_id: str, scope: DraftScope) -> list[str]:
    """Scorable games with no ``draft_scores`` row under ``model_id``."""
    rows = conn.execute(
        "SELECT g.game_id" + _SCORABLE_GAMES + " AND NOT EXISTS (SELECT 1 FROM draft_scores ds "
        "                   WHERE ds.game_id = g.game_id AND ds.model_id = ?"
        "                     AND ds.context = 'backfill')" + _ORDER_BY,
        (scope.start.isoformat(), model_id),
    ).fetchall()
    return [r["game_id"] for r in rows]


def _stale_draft_games(
    conn: sqlite3.Connection, model_id: str, scope: DraftScope, *, force: bool
) -> list[str]:
    """Scorable games needing replacement: rows under another model, or (``force``) any."""
    if force:
        predicate = (
            " AND EXISTS (SELECT 1 FROM draft_scores ds "
            "             WHERE ds.game_id = g.game_id AND ds.context = 'backfill')"
        )
        params: tuple = (scope.start.isoformat(),)
    else:
        predicate = (
            " AND EXISTS (SELECT 1 FROM draft_scores ds "
            "             WHERE ds.game_id = g.game_id AND ds.model_id != ?"
            "               AND ds.context = 'backfill')"
        )
        params = (scope.start.isoformat(), model_id)
    rows = conn.execute(
        "SELECT g.game_id" + _SCORABLE_GAMES + predicate + _ORDER_BY, params
    ).fetchall()
    return [r["game_id"] for r in rows]


def _score_present(
    predictor: DraftGrader,
    present: list[tuple[str, DraftState]],
    stats_cache,
    counts: dict[str, int],
) -> list[tuple[str, DraftScore]]:
    """Score one batch's loaded drafts in one forward, ``(game_id, score)`` per success.

    Returns:
        A batch-level failure falls back to per-game scoring, so one corrupt draft is
        counted in ``counts['skipped_predict_error']`` and dropped, never fatal.
    """
    scored: list[tuple[str, DraftScore]] = []
    try:
        scores = predictor.predict_batch([d for _gid, d in present], stats_cache=stats_cache)
    except Exception as exc:  # noqa: BLE001 - any batch failure must fall back per-game, never abort the pass
        logger.warning("draft batch score error, falling back per-game: %s", exc)
        for gid, draft in present:
            try:
                score = predictor.predict(draft, stats_cache=stats_cache)
            except Exception as inner:  # noqa: BLE001 - one corrupt draft counts against the error budget, never aborts
                logger.warning("draft score error on %s: %s", gid, inner)
                counts["skipped_predict_error"] += 1
                continue
            scored.append((gid, score))
        return scored
    if len(scores) != len(present):
        raise RuntimeError(f"predict_batch returned {len(scores)} scores for {len(present)} drafts")
    return [(gid, score) for (gid, _draft), score in zip(present, scores, strict=True)]


def _process(
    conn: sqlite3.Connection,
    predictor: DraftGrader,
    model_id: str,
    game_ids: list[str],
    stats_cache,
    *,
    delete_stale: bool,
) -> dict[str, int]:
    """Score ``game_ids`` and write ``draft_scores`` in batches into the CALLER's transaction.

    Args:
        model_id: Caller-resolved ``'draft:<stem>'``; draft_grading never composes it.
        delete_stale: Delete each game's non-target rows in the write's transaction.

    Returns:
        Per-outcome counters.

    Raises:
        RuntimeError: :data:`SCORING_ERROR_BUDGET` breached at the pass end.
    """
    counts = {"scored": 0, "skipped_no_draft": 0, "skipped_predict_error": 0}
    if not game_ids:
        return counts

    for start in range(0, len(game_ids), BATCH_SIZE):
        batch = game_ids[start : start + BATCH_SIZE]
        drafts = DraftState.bulk_load(conn, batch)
        present = [(gid, drafts[gid]) for gid in batch if drafts.get(gid) is not None]
        counts["skipped_no_draft"] += len(batch) - len(present)

        scored = _score_present(predictor, present, stats_cache, counts)
        # edge_pp_blue: blue's calibrated expected edge (red's is its exact
        # negation — one stored orientation); NULL under a pre-edge grader.
        rows: list[tuple] = [
            (
                gid,
                model_id,
                float(score.prob_blue_wins),
                str(score.grade_blue),
                str(score.grade_red),
                None if score.edge_pp_blue is None else float(score.edge_pp_blue),
            )
            for gid, score in scored
        ]
        stale_keys: list[tuple] = (
            [(gid, model_id) for gid, _score in scored] if delete_stale else []
        )

        if delete_stale and stale_keys:
            conn.executemany(
                "DELETE FROM draft_scores WHERE game_id = ? AND model_id != ? "
                "AND context = 'backfill'",
                stale_keys,
            )
        if rows:
            conn.executemany(
                "INSERT OR REPLACE INTO draft_scores "
                "(game_id, model_id, p_blue, grade_blue, grade_red, edge_pp_blue, context) "
                "VALUES (?, ?, ?, ?, ?, ?, 'backfill')",
                rows,
            )
            counts["scored"] += len(rows)

    enforce_error_budget(
        counts, scope="whole-draft rescore" if delete_stale else "whole-draft scoring"
    )
    return counts
