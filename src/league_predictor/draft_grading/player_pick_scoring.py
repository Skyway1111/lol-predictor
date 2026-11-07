"""Score each drafted player pick for the dashboard radar."""  # sightline-ok: 27 - one pipeline

# file-length-ok: one reconcile pipeline — one PickScope, model id and error budget travel
# from slot selection through serial and parallel scoring, row emission and the spot check.

from __future__ import annotations

import logging
import random
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from league_predictor import resources
from league_predictor.db.rebuild_guard import require_rebuild_schema
from league_predictor.draft_grading.draft_scoring import enforce_error_budget
from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.grader.enemy_response import cached_enemy_response_distribution
from league_predictor.draft_grading.grader.predictor import SwapRequest
from league_predictor.draft_grading.grader.swaps import legal_alternatives
from league_predictor.draft_grading.live import load_grader
from league_predictor.draft_grading.state import ROLE_ORDER, DraftState

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from concurrent.futures import Future
    from datetime import date, datetime

    from league_predictor.draft_grading.live import DraftGrader


logger = logging.getLogger(__name__)

# Compact blind-pick expectation: the most likely walk-forward response. With the bounded
# role pool this is the fast radar recipe — every slot scored, no counterfactual fan-out.
ENEMY_K = 1

# Flush cadence: bounds the in-memory row buffer (the writes stream into the caller's
# open transaction; the verb boundary owns the single commit).
MIN_PICKS_PER_BATCH = 500

# Cross-game inference width: each game expands into its role-alternative x enemy-response
# rows, so this bounds the materialized tensor while amortizing dispatch across the corpus.
PICK_GAMES_PER_FORWARD = 256

# Below this many slots a pass scores in-process; at or above it, read-only
# process workers score partitions and the parent owns the atomic write.
_PARALLEL_MIN_SLOTS = 10_000

# One worker loads the draft family (both blend sides), encoder(s), and its bulk stats cache:
# measured peak RSS ~1.5 GB over 250k slots, so 2.5 GB keeps a ~1 GB per-child margin.
PICK_SCORING_WORKER_GB = 2.5

# Tables a pick-scoring run reads or writes; the guard names any that are absent.
_REQUIRED_TABLES = ("models", "series", "games", "game_players", "draft_actions", "pick_scores")


@dataclass(frozen=True)
class PickScope:
    """The window a (re)score pass covers: date floor and per-role alternative pools.

    Attributes:
        year_start: bounds slot detection (``_SCORABLE_SLOTS`` carries the predicate);
            production fixes it at the corpus floor, so stale detection and new coverage
            see the same complete historical population.
    """

    year_start: date
    role_pools: dict[str, set[str]]


class _SeqStep(NamedTuple):
    """One pick action, duck-typed for ``_classify_blind_or_counter``.

    Attributes:
        step: the ``draft_actions`` ``seq`` — monotonic in pick order, so relative
            comparisons match the true snake sequence.
    """

    step: int
    side: str
    role: str


# A game is scorable when its stored draft is complete: ten pick actions with a resolved role
# + champion, both sides known. Ban holes leave seq gaps and are irrelevant — only picks gate.
_SLOT_SIDE = "CASE WHEN gp.team_id = g.blue_team THEN 'blue' ELSE 'red' END"

_SCORABLE_SLOTS = """
    FROM game_players gp
    JOIN games g ON g.game_id = gp.game_id
    JOIN series s ON s.series_id = g.series_id
    WHERE s.scheduled_ts >= ?
      AND g.blue_team IS NOT NULL AND g.red_team IS NOT NULL
      AND gp.role IS NOT NULL AND gp.champion_id IS NOT NULL
      AND gp.team_id IN (g.blue_team, g.red_team)
      AND (SELECT COUNT(*) FROM draft_actions da
           WHERE da.game_id = gp.game_id AND da.kind = 'pick'
             AND da.role IS NOT NULL AND da.champion_id IS NOT NULL) = 10
"""

_SLOT_SELECT = f"""
    SELECT gp.game_id AS game_id, {_SLOT_SIDE} AS side, gp.role AS role,
           gp.player_id AS player_id, gp.champion_id AS picked_champ
"""

_ORDER_BY = " ORDER BY s.scheduled_ts, gp.game_id"


@dataclass
class PickReconcile:
    """A started missing-or-stale pick pass (``begin_pick_reconcile``); ``finish`` completes it."""

    _counts: dict[str, int] | None = None
    _job: _ParallelPickJob | None = None

    def finish(self, conn: sqlite3.Connection) -> dict[str, int]:
        if self._counts is not None:
            return self._counts
        assert self._job is not None  # constructed with exactly one of the two
        return self._job.finish(conn)


def begin_pick_reconcile(
    conn: sqlite3.Connection,
    predictor: DraftGrader,
    model_id: str,
    scope: PickScope,
    *,
    stats_cache: BulkStatsCache | None = None,
    stats_cache_future: Future[BulkStatsCache] | None = None,
    parallel_model_path: Path | None = None,
    force: bool = False,
) -> PickReconcile:
    """Start the missing-or-stale pick pass; the caller overlaps work, then ``finish``es.

    Args:
        force: score EVERY scorable slot regardless of coverage — the version-safe full
            re-score (a same-id row is not proof of freshness once a data refresh has
            revised the facts under it).
        parallel_model_path: large corpora dispatch the read-only pool and RETURN
            IMMEDIATELY; ``finish(conn)`` joins and flushes into the caller's transaction
            (workers never read the tables the parent writes). Small corpora score
            in-process at begin.
    """
    require_rebuild_schema(conn, *_REQUIRED_TABLES, purpose="player pick reconciliation")
    slots = _reconcile_slots(conn, model_id, scope, force=force)
    if parallel_model_path is not None and len(slots) >= _PARALLEL_MIN_SLOTS:
        return PickReconcile(
            _job=_dispatch_parallel(
                conn, model_id, slots, scope, parallel_model_path, delete_stale=True
            )
        )
    if stats_cache is None and stats_cache_future is not None:
        stats_cache = stats_cache_future.result()
    return PickReconcile(
        _counts=_process(conn, predictor, model_id, slots, scope, stats_cache, delete_stale=True)
    )


# Slot detection (the games ⋈ draft_actions ⋈ game_players set-difference)


def _reconcile_slots(
    conn: sqlite3.Connection, model_id: str, scope: PickScope, *, force: bool
) -> list[dict]:
    """The staleness predicate: every slot (force), else missing-or-stale vs ``model_id``."""
    if force:
        predicate = ""
        params = (scope.year_start.isoformat(),)
    else:
        predicate = f"""
          AND (NOT EXISTS (SELECT 1 FROM pick_scores target
                           WHERE target.game_id = gp.game_id AND target.side = {_SLOT_SIDE}
                             AND target.role = gp.role AND target.model_id = ?)
               OR EXISTS (SELECT 1 FROM pick_scores stale
                          WHERE stale.game_id = gp.game_id AND stale.side = {_SLOT_SIDE}
                            AND stale.role = gp.role AND stale.model_id != ?))
        """
        params = (scope.year_start.isoformat(), model_id, model_id)
    rows = conn.execute(_SLOT_SELECT + _SCORABLE_SLOTS + predicate + _ORDER_BY, params).fetchall()
    return [dict(row) for row in rows]


def _pick_sequences(conn: sqlite3.Connection, game_ids: Sequence[str]) -> dict[str, list[_SeqStep]]:
    """``{game_id: [_SeqStep, ...]}`` per game's pick actions, ordered by the persisted ``seq``."""
    seqs: dict[str, list[_SeqStep]] = {}
    for start in range(0, len(game_ids), 500):
        chunk = game_ids[start : start + 500]
        placeholders = ",".join("?" for _ in chunk)
        for r in conn.execute(
            f"SELECT game_id, seq, side, role FROM draft_actions "
            f"WHERE kind = 'pick' AND game_id IN ({placeholders}) "
            f"ORDER BY game_id, seq",
            chunk,
        ):
            seqs.setdefault(r["game_id"], []).append(_SeqStep(r["seq"], r["side"], r["role"]))
    return seqs


# Scoring core


def _prepare_game_requests(
    conn: sqlite3.Connection,
    draft: DraftState,
    game_slots: Iterable[dict],
    sequence: list[_SeqStep],
    *,
    index,
    scope: PickScope,
    counts: dict[str, int],
) -> tuple[list[tuple[dict, list[str]]], list[SwapRequest]]:
    """One marginalized-swap request per scorable slot of one game.

    Returns:
        ``(prepared, requests)``, pairwise aligned, ``prepared`` carrying each request's
        ``(slot, alternatives)``; unclassifiable, degenerate, and unresolvable slots are
        counted in ``counts`` and dropped before the forward.
    """
    as_of = draft.date
    prepared: list[tuple[dict, list[str]]] = []  # (slot, alternatives)
    requests: list[SwapRequest] = []
    for slot in game_slots:
        side, role, picked = slot["side"], slot["role"], slot["picked_champ"]
        try:
            is_blind = _classify_blind_or_counter(sequence, side, role)
        except ValueError:
            counts["skipped_predict_error"] += 1
            continue

        alts_full = legal_alternatives(draft, index)
        pool = scope.role_pools.get(role, set())
        alternatives = [a for a in alts_full if a in pool]
        # Prepend the actual pick if the role pool excludes it, so its
        # percentile is always defined.
        if picked not in alternatives:
            alternatives = [picked, *alternatives]
        if len(alternatives) < 2:  # noqa: PLR2004 - a percentile needs a rival
            counts["skipped_degenerate"] += 1  # degenerate set — expected-small, non-fatal
            continue

        try:
            responses = _resolve_enemy_responses(
                conn, draft, side, role, is_blind=is_blind, as_of=as_of
            )
        except Exception as exc:  # noqa: BLE001 - one unresolvable slot counts against the error budget
            logger.warning(
                "enemy-response resolve error on %s/%s/%s: %s", draft.match_id, side, role, exc
            )
            counts["skipped_predict_error"] += 1
            continue

        prepared.append((slot, alternatives))
        requests.append(SwapRequest(side, role, alternatives, role, responses))
    return prepared, requests


@dataclass(frozen=True)
class _PassState:
    """One scoring pass's accumulators: outcome counters, stale keys, and the delete policy."""

    counts: dict[str, int]
    stale_keys: list[tuple]
    delete_stale: bool

    def mark_stale(self, slot: Mapping, model_id: str) -> None:
        """Queue this slot's non-target rows for deletion, when the pass deletes at all."""
        if self.delete_stale:
            self.stale_keys.append((slot["game_id"], slot["side"], slot["role"], model_id))


def _emit_rows(
    conn: sqlite3.Connection,
    rows: list[tuple],
    state: _PassState,
    *,
    sink: list[tuple] | None,
) -> None:
    """Land one row buffer: into the caller's transaction, or into ``sink`` when it collects."""
    if sink is None:
        _flush(conn, rows, state.stale_keys, delete_stale=state.delete_stale)
    else:
        sink.extend(rows)
    state.counts["scored"] += len(rows)


def _process(
    conn: sqlite3.Connection,
    predictor: DraftGrader,
    model_id: str,
    slots: list[dict],
    scope: PickScope,
    stats_cache,
    *,
    delete_stale: bool,
    _collect_rows: list[tuple] | None = None,
) -> dict[str, int]:
    """Score ``slots`` under ``model_id`` and write ``pick_scores`` in game-grouped batches.

    Args:
        delete_stale: each flushed batch also deletes the slots' non-target rows, in the
            same transaction as their replacements.

    Returns:
        Per-outcome counters. Writes land in the CALLER's transaction (no per-batch commit,
        so a budget breach rolls the whole pass back); :data:`SCORING_ERROR_BUDGET` is
        enforced at the end, while ``skipped_no_draft`` / ``skipped_degenerate`` stay
        non-fatal.
    """
    counts = {
        "scored": 0,
        "skipped_no_draft": 0,
        "skipped_predict_error": 0,
        "skipped_degenerate": 0,
    }
    if not slots:
        return counts

    slots_by_game: dict[str, list[dict]] = {}
    for slot in slots:
        slots_by_game.setdefault(slot["game_id"], []).append(slot)
    game_ids = list(slots_by_game)

    drafts = DraftState.bulk_load(conn, game_ids)
    sequences = _pick_sequences(conn, game_ids)

    rows_buffer: list[tuple] = []
    state = _PassState(counts, stale_keys=[], delete_stale=delete_stale)
    inference_batch: list[tuple[DraftState, list[tuple[dict, list[str]]], list[SwapRequest]]] = []
    t0 = time.time()
    n_games = 0

    for gid, game_slots in slots_by_game.items():
        draft = drafts.get(gid)
        if draft is None:
            counts["skipped_no_draft"] += len(game_slots)
            continue
        prepared, requests = _prepare_game_requests(
            conn,
            draft,
            game_slots,
            sequences.get(gid, []),
            index=predictor.index,
            scope=scope,
            counts=counts,
        )

        inference_batch.append((draft, prepared, requests))
        if len(inference_batch) >= PICK_GAMES_PER_FORWARD:
            rows_buffer.extend(
                _score_many_game_requests(
                    predictor,
                    inference_batch,
                    model_id=model_id,
                    stats_cache=stats_cache,
                    state=state,
                )
            )
            inference_batch.clear()

        n_games += 1
        if len(rows_buffer) >= MIN_PICKS_PER_BATCH:
            _emit_rows(conn, rows_buffer, state, sink=_collect_rows)
            rows_buffer.clear()
            state.stale_keys.clear()
            logger.info(
                "score picks: %d games, %d scored (%.0fs)",
                n_games,
                counts["scored"],
                time.time() - t0,
            )

    if inference_batch:
        rows_buffer.extend(
            _score_many_game_requests(
                predictor, inference_batch, model_id=model_id, stats_cache=stats_cache, state=state
            )
        )

    if rows_buffer:
        _emit_rows(conn, rows_buffer, state, sink=_collect_rows)

    # A large predict-error rate is a systematic bug that must abort the pass before its rows
    # commit; degenerate alt sets are counted separately, outside the budget.
    enforce_error_budget(counts, scope="per-pick rescore" if delete_stale else "per-pick scoring")
    return counts


def _parallel_pick_worker(
    db_path: str,
    model_path: str,
    *,
    model_id: str,
    slots: list[dict],
    scope: PickScope,
) -> tuple[list[tuple], dict[str, int]]:
    """Read-only worker: score one slot partition and return rows; never writes the facts DB."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        predictor = load_grader(conn, artifact=Path(model_path))
        stats_cache = BulkStatsCache(conn, baseline=predictor.pro_excess_baseline)
        rows: list[tuple] = []
        counts = _process(
            conn,
            predictor,
            model_id,
            slots,
            scope,
            stats_cache,
            delete_stale=False,
            _collect_rows=rows,
        )
        return rows, counts
    finally:
        conn.close()


@dataclass
class _ParallelPickJob:
    """Dispatched pick-partition workers; ``finish`` joins, budget-checks, and flushes."""

    pool: ProcessPoolExecutor | None
    pending: list
    outputs: list[tuple[list[tuple], dict[str, int]]]
    model_id: str
    delete_stale: bool

    def finish(self, conn: sqlite3.Connection) -> dict[str, int]:
        if self.pool is not None:
            try:
                self.outputs = [future.result() for future in self.pending]
            finally:
                self.pool.shutdown(wait=True)
        counts = {
            key: sum(output_counts.get(key, 0) for _rows, output_counts in self.outputs)
            for key in (
                "scored",
                "skipped_no_draft",
                "skipped_predict_error",
                "skipped_degenerate",
            )
        }
        rows = [row for output_rows, _counts in self.outputs for row in output_rows]
        enforce_error_budget(counts, scope="parallel per-pick rescore")
        stale_keys = (
            [(row[0], row[1], row[2], self.model_id) for row in rows] if self.delete_stale else []
        )
        _flush(conn, rows, stale_keys, delete_stale=self.delete_stale)
        return counts


def _dispatch_parallel(
    conn: sqlite3.Connection,
    model_id: str,
    slots: list[dict],
    scope: PickScope,
    model_path: Path,
    *,
    delete_stale: bool,
) -> _ParallelPickJob:
    """Partition ``slots`` and dispatch read-only workers; the caller joins via ``finish``.

    Returns:
        The job. A width-one pool runs inline — a memory-starved box must not pay
        process-pool overhead for serial work, so its "begin" completes synchronously.
    """
    db_path = next((row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main"), "")
    if not db_path:
        raise RuntimeError("parallel pick scoring requires a file-backed SQLite database")
    workers = resources.bounded_workers(PICK_SCORING_WORKER_GB, cap=6)
    stride = (len(slots) + workers - 1) // workers
    partitions = [slots[start : start + stride] for start in range(0, len(slots), stride)]
    logger.info(
        "parallel pick scoring: %d slots across %d worker(s) (%.1f GB available)",
        len(slots),
        len(partitions),
        resources.available_memory_gb(),
    )
    if len(partitions) == 1:
        outputs = [
            _parallel_pick_worker(
                db_path, str(model_path), model_id=model_id, slots=partitions[0], scope=scope
            )
        ]
        return _ParallelPickJob(
            pool=None, pending=[], outputs=outputs, model_id=model_id, delete_stale=delete_stale
        )
    pool = ProcessPoolExecutor(max_workers=len(partitions))
    futures = [
        pool.submit(
            _parallel_pick_worker,
            db_path,
            str(model_path),
            model_id=model_id,
            slots=partition,
            scope=scope,
        )
        for partition in partitions
    ]
    return _ParallelPickJob(
        pool=pool, pending=futures, outputs=[], model_id=model_id, delete_stale=delete_stale
    )


def _score_many_game_requests(
    predictor: DraftGrader,
    batch: list[tuple[DraftState, list[tuple[dict, list[str]]], list[SwapRequest]]],
    *,
    model_id: str,
    stats_cache,
    state: _PassState,
) -> list[tuple]:
    """Score several games in one forward, preserving the per-game fallback contract."""
    try:
        output_batches = predictor.predict_swaps_marginalized_many(
            [(draft, requests) for draft, _prepared, requests in batch],
            stats_cache=stats_cache,
        )
        rows: list[tuple] = []
        for (_draft, prepared, _requests), outputs in zip(batch, output_batches, strict=True):
            for (slot, alternatives), scores in zip(prepared, outputs, strict=True):
                state.mark_stale(slot, model_id)
                rows.append(_pick_row(slot, alternatives, scores, model_id))
    except Exception as exc:  # noqa: BLE001 - any batch failure must fall back per-game, never abort the pass
        logger.warning("multi-game pick score error, falling back per-game: %s", exc)
    else:
        return rows

    rows = []
    for draft, prepared, requests in batch:
        rows.extend(
            _score_game_requests(
                predictor,
                draft,
                prepared,
                requests,
                model_id=model_id,
                stats_cache=stats_cache,
                state=state,
            )
        )
    return rows


def _pick_row(slot: Mapping, alternatives: list[str], scores: np.ndarray, model_id: str) -> tuple:
    """Assemble one ``pick_scores`` row from a finalized per-alternative score array."""
    actual, best, worst, pct = _finalize_pick_scores(
        scores, slot["side"], alternatives, slot["picked_champ"]
    )
    return (
        slot["game_id"],
        slot["side"],
        slot["role"],
        model_id,
        slot["player_id"],
        actual,
        len(alternatives),
        best,
        worst,
        pct,
    )


def _score_game_requests(
    predictor: DraftGrader,
    draft: DraftState,
    prepared: list[tuple[dict, list[str]]],
    requests: list[SwapRequest],
    *,
    model_id: str,
    stats_cache,
    state: _PassState,
) -> list[tuple]:
    """Batch-score a game's prepared swap requests into ``pick_scores`` rows.

    Returns:
        The row tuples. Any batch-level failure falls back to per-pick scoring, so a single
        unscorable slot is counted (``skipped_predict_error``) instead of dropping the game.
    """
    if not requests:
        return []

    def _emit(slot: dict, alternatives: list[str], scores: np.ndarray) -> tuple:
        state.mark_stale(slot, model_id)
        return _pick_row(slot, alternatives, scores, model_id)

    try:
        score_arrays = predictor.predict_swaps_marginalized_batch(
            draft, requests, stats_cache=stats_cache
        )
        # strict: one score array per prepared request — a wrong-length return is a contract
        # breach that must surface (and fall back per-pick), never a silent truncation.
        return [
            _emit(slot, alts, scores)
            for (slot, alts), scores in zip(prepared, score_arrays, strict=True)
        ]
    except Exception as exc:  # noqa: BLE001 - any batch failure must fall back per-pick, never drop the game
        logger.warning(
            "pick batch score error on %s, falling back per-pick: %s", draft.match_id, exc
        )

    rows: list[tuple] = []
    # prepared and requests are appended in lockstep above, so strict just pins that invariant.
    for (slot, alternatives), req in zip(prepared, requests, strict=True):
        try:
            scores = predictor.predict_swaps_marginalized(
                draft,
                req.side,
                req.role,
                req.alternatives,
                enemy_role=req.enemy_role,
                enemy_responses=req.enemy_responses,
                stats_cache=stats_cache,
            )
        except Exception as exc:  # noqa: BLE001 - one unscorable pick counts against the error budget
            logger.warning(
                "score error on %s/%s/%s: %s",
                slot["game_id"],
                slot["side"],
                slot["role"],
                exc,
            )
            state.counts["skipped_predict_error"] += 1
            continue
        rows.append(_emit(slot, alternatives, scores))
    return rows


def _flush(
    conn: sqlite3.Connection,
    rows: list[tuple],
    stale_keys: list[tuple],
    *,
    delete_stale: bool,
) -> None:
    """Write one batch into the caller's transaction; deletes stale rows first, never commits."""
    if delete_stale and stale_keys:
        conn.executemany(
            "DELETE FROM pick_scores WHERE game_id = ? AND side = ? AND role = ? AND model_id != ?",
            stale_keys,
        )
    conn.executemany(
        "INSERT OR REPLACE INTO pick_scores "
        "(game_id, side, role, model_id, player_id, actual_score, alt_count, "
        " best_score, worst_score, percentile) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )


def spot_check_pick_rows(
    conn: sqlite3.Connection,
    predictor: DraftGrader,
    model_id: str,
    scope: PickScope,
    *,
    stats_cache: BulkStatsCache | None = None,
    sample_size: int = 20,
    seed: int = 0,
) -> int:
    """Re-score sampled stored rows on the singular path; raise on any bitwise drift.

    Args:
        sample_size: rows sampled under ``model_id``; COST is that many single forwards.

    Returns:
        The number of rows checked.

    Raises:
        RuntimeError: a stored row differs from its re-score through
            ``predict_swaps_marginalized`` — batch/parallel composition must never change
            a row.
    """
    rows = conn.execute(
        "SELECT ps.game_id, ps.side, ps.role, ps.player_id, ps.actual_score, ps.alt_count, "
        "       ps.best_score, ps.worst_score, ps.percentile, gp.champion_id AS picked_champ "
        "FROM pick_scores ps JOIN game_players gp ON gp.game_id = ps.game_id "
        "     AND gp.player_id = ps.player_id AND gp.role = ps.role "
        "WHERE ps.model_id = ? ORDER BY ps.game_id, ps.side, ps.role",
        (model_id,),
    ).fetchall()
    if not rows:
        return 0
    sample = random.Random(seed).sample(rows, min(sample_size, len(rows)))
    game_ids = sorted({row["game_id"] for row in sample})
    drafts = DraftState.bulk_load(conn, game_ids)
    sequences = _pick_sequences(conn, game_ids)
    for row in sample:
        gid, side, role = row["game_id"], row["side"], row["role"]
        label = f"{gid}/{side}/{role}"
        draft = drafts.get(gid)
        if draft is None:
            raise RuntimeError(f"spot check: no loadable draft for stored row {label}")
        is_blind = _classify_blind_or_counter(sequences.get(gid, []), side, role)
        pool = scope.role_pools.get(role, set())
        alternatives = [a for a in legal_alternatives(draft, predictor.index) if a in pool]
        if row["picked_champ"] not in alternatives:
            alternatives = [row["picked_champ"], *alternatives]
        responses = _resolve_enemy_responses(
            conn, draft, side, role, is_blind=is_blind, as_of=draft.date
        )
        scores = predictor.predict_swaps_marginalized(
            draft,
            side,
            role,
            alternatives,
            enemy_role=role,
            enemy_responses=responses,
            stats_cache=stats_cache,
        )
        slot = {
            "game_id": gid,
            "side": side,
            "role": role,
            "player_id": row["player_id"],
            "picked_champ": row["picked_champ"],
        }
        want = _pick_row(slot, alternatives, scores, model_id)
        got = (
            gid,
            side,
            role,
            model_id,
            row["player_id"],
            row["actual_score"],
            row["alt_count"],
            row["best_score"],
            row["worst_score"],
            row["percentile"],
        )
        if want != got:
            raise RuntimeError(
                f"spot check: stored row {label} is not bitwise reproducible on the "
                f"singular path (stored {got[5:]}, re-scored {tuple(want[5:])}) — "
                "batch scoring drifted or the artifact/DB changed under the run."
            )
    return len(sample)


def _classify_blind_or_counter(
    sequence: Iterable[_SeqStep],
    side: str,
    role: str,
) -> bool:
    """``(is_blind, focal_step)`` for the focal slot; blind iff the enemy same-role pick is later.

    Raises:
        ValueError: the focal slot is not in the sequence — the caller counts it as a
            predict error.
    """
    focal_step: int | None = None
    enemy_step: int | None = None
    enemy_side = "red" if side == "blue" else "blue"
    for pick in sequence:
        if pick.role == role and pick.side == side and focal_step is None:
            focal_step = pick.step
        elif pick.role == role and pick.side == enemy_side and enemy_step is None:
            enemy_step = pick.step
    if focal_step is None:
        raise ValueError(f"focal slot {side}/{role} not in sequence")
    if enemy_step is None:
        # Enemy never picks this role — treat as blind (shouldn't happen on a
        # complete draft, but guard the contract).
        return True
    return enemy_step > focal_step


def _resolve_enemy_responses(
    db: sqlite3.Connection,
    draft: DraftState,
    side: str,
    role: str,
    *,
    is_blind: bool,
    as_of: datetime,
) -> list[tuple[str, float]]:
    """The enemy same-role response distribution a pick marginalizes over.

    Returns:
        The top-K enemy responses for a blind pick with a non-empty meta prior; otherwise
        the KNOWN enemy champion at the focal role, at weight 1.0.
    """
    if is_blind:
        prior = cached_enemy_response_distribution(db, role, as_of, top_k=ENEMY_K)
        if prior:
            return prior
    enemy_comp = draft.red if side == "blue" else draft.blue
    enemy_champ = enemy_comp.slots[ROLE_ORDER.index(role)].champion_id
    return [(enemy_champ, 1.0)]


def _finalize_pick_scores(
    scores: np.ndarray,
    side: str,
    alternatives: list[str],
    picked_champ: str,
) -> tuple[float, float, float, float]:
    """Turn a per-alternative ``prob_blue_wins`` array into the focal side's score tuple.

    Returns:
        ``(actual, best, worst, percentile)`` in the focal side's frame (flipped for red, so
        higher is better); percentile is the pick's rank inside ``[worst, best]`` normalized
        to 0..1, and ``0.5`` on a degenerate flat envelope.
    """
    if side == "red":
        scores = 1.0 - scores
    if picked_champ not in alternatives:
        raise ValueError(f"picked_champ={picked_champ!r} not in alternatives (caller must prepend)")
    actual = float(scores[alternatives.index(picked_champ)])
    best = float(np.max(scores))
    worst = float(np.min(scores))
    pct = (actual - worst) / (best - worst) if best > worst else 0.5
    return actual, best, worst, pct
