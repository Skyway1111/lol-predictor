"""Draft-score and pick-score coverage under a caller-owned transaction."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from typing import TYPE_CHECKING

from league_predictor.draft_grading import oos_walk, player_pick_scoring
from league_predictor.draft_grading.draft_scoring import (
    DraftScope,
    rescore_stale_drafts,
    score_new_drafts,
)
from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.grader.features import DEFAULT_MIN_ROLE_GAMES, build_role_pools
from league_predictor.draft_grading.live import load_grader
from league_predictor.draft_grading.player_pick_scoring import (
    PickScope,
    begin_pick_reconcile,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from league_predictor.draft_grading.live import DraftGrader


_BACKFILL_START = date(2017, 1, 1)

# The compact pick-radar counterfactual set: the most-played legal champions per role,
# plus the actual pick when it falls outside the set.
PICK_RADAR_CANDIDATES_PER_ROLE = 10

# The D6 reproducibility spot check: stored pick rows re-scored per run on the
# singular path, compared bitwise.
SPOT_CHECK_ROWS = 20

# Above this many player rows the run pays for a background stats-cache build.
_LARGE_CORPUS_ROWS = 10_000


def rescore_all(
    conn: sqlite3.Connection,
    prd_id: str,
    *,
    model_path: Path,
) -> dict[str, int]:
    """Recompute BOTH score tables in full (``force=True``); minutes over the ~250k-slot corpus.

    Returns:
        Per-outcome counters plus ``spot_checked``: pick rows a composed blend re-scored.

    Raises:
        RuntimeError: a re-scored stored pick row drifts bitwise.
    """
    n_rows = conn.execute("SELECT COUNT(*) FROM game_players").fetchone()[0]
    large_corpus = n_rows >= _LARGE_CORPUS_ROWS
    cache_future = None
    cache_pool = None
    if large_corpus:
        predictor, draft_scope, pick_scope = _build_backfill_shell(conn, model_path=model_path)
        db_path = next(
            (row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), ""
        )
        cache_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="draft-parent-cache")
        cache_future = cache_pool.submit(
            _build_stats_cache_readonly, db_path, predictor.pro_excess_baseline
        )
        stats_cache = None
    else:
        predictor, stats_cache, draft_scope, pick_scope = _build_backfill_context(
            conn, model_path=model_path
        )
    counts = {
        "scored": 0,
        "skipped_no_draft": 0,
        "skipped_predict_error": 0,
        "skipped_degenerate": 0,
    }

    def _accumulate(pass_counts: Mapping[str, int]) -> None:
        for key in counts:
            counts[key] += pass_counts.get(key, 0)

    try:
        # Pick pass first: a large corpus dispatches CPU workers and returns immediately,
        # so the draft (GPU) passes below overlap; finish() joins and flushes here.
        pick_pass = begin_pick_reconcile(
            conn,
            predictor,
            prd_id,
            pick_scope,
            stats_cache=stats_cache,
            stats_cache_future=cache_future,
            parallel_model_path=model_path,
            force=True,
        )
        if cache_future is not None:
            stats_cache = cache_future.result()
        counts["scored"] += rescore_stale_drafts(
            conn, predictor, prd_id, draft_scope, stats_cache=stats_cache, force=True
        )
        _accumulate(score_new_drafts(conn, predictor, prd_id, draft_scope, stats_cache=stats_cache))
        _accumulate(pick_pass.finish(conn))
        counts["pruned_superseded"] = _prune_superseded_scores(conn, prd_id)
        counts["spot_checked"] = (
            player_pick_scoring.spot_check_pick_rows(
                conn,
                predictor,
                prd_id,
                pick_scope,
                stats_cache=stats_cache,
                sample_size=SPOT_CHECK_ROWS,
            )
            if True
            else 0
        )
    finally:
        if cache_pool is not None:
            cache_pool.shutdown(wait=True)
    return counts


def write_oos_walk(
    conn: sqlite3.Connection,
    model_id: str,
    *,
    through: date,
    served_id: str,
    code_fingerprint: str,
) -> int:
    """Replace the id's ``oos`` rows from ``oos_walk`` (COST: twelve refits); rows written."""
    scores = oos_walk.walk_draft_oos(
        conn, through=through, served_id=served_id, code_fingerprint=code_fingerprint
    )
    conn.execute("DELETE FROM draft_scores WHERE model_id = ? AND context = 'oos'", (model_id,))
    conn.executemany(
        "INSERT INTO draft_scores "
        "(game_id, model_id, p_blue, grade_blue, grade_red, edge_pp_blue, context) "
        "VALUES (?, ?, ?, ?, ?, ?, 'oos')",
        [
            (s.game_id, model_id, s.p_blue, s.grade_blue, s.grade_red, s.edge_pp_blue)
            for s in scores
        ],
    )
    return len(scores)


def _prune_superseded_scores(conn: sqlite3.Connection, model_id: str) -> int:
    """Remove old-model rows that an incomplete draft cannot be rescored to replace."""
    deleted = 0
    for table in ("draft_scores", "pick_scores"):
        cursor = conn.execute(f"DELETE FROM {table} WHERE model_id != ?", (model_id,))
        deleted += cursor.rowcount
    return deleted


def _build_backfill_context(
    conn: sqlite3.Connection,
    *,
    model_path: Path,
) -> tuple[DraftGrader, BulkStatsCache, DraftScope, PickScope]:
    """Build the grader, shared stats cache, and coverage scopes."""
    predictor, draft_scope, pick_scope = _build_backfill_shell(conn, model_path=model_path)
    stats_cache = BulkStatsCache(conn, baseline=predictor.pro_excess_baseline)
    return predictor, stats_cache, draft_scope, pick_scope


def _build_backfill_shell(
    conn: sqlite3.Connection,
    *,
    model_path: Path,
) -> tuple[DraftGrader, DraftScope, PickScope]:
    """Load the artifact through the dispatching loader plus the immutable scopes."""
    predictor = load_grader(conn, artifact=model_path)
    draft_scope = DraftScope(start=_BACKFILL_START)
    role_pools = build_role_pools(
        conn,
        min_role_games=DEFAULT_MIN_ROLE_GAMES,
        max_champions=PICK_RADAR_CANDIDATES_PER_ROLE,
    )
    pick_scope = PickScope(year_start=_BACKFILL_START, role_pools=role_pools)
    return predictor, draft_scope, pick_scope


def _build_stats_cache_readonly(db_path: str, baseline: str) -> BulkStatsCache:
    """Build the parent cache on an isolated read-only connection for overlap."""
    cache_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return BulkStatsCache(cache_conn, baseline=baseline)
    finally:
        cache_conn.close()
