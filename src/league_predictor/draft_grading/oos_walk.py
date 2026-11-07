"""The draft model's walk-forward history: one full recipe refit per trailing month."""

from __future__ import annotations

import logging
import sqlite3
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor import artifacts, resources
from league_predictor.clock import trailing_months
from league_predictor.draft_grading import recipe as draft_recipe
from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.live import DraftGrader, load_grader
from league_predictor.draft_grading.state import DraftState
from league_predictor.draft_grading.training.provenance import served_oos_probs
from league_predictor.draft_grading.training.set_fit import fit_candidate

if TYPE_CHECKING:
    from datetime import date

logger = logging.getLogger(__name__)

#: Trailing coverage span, served's own: the stacker reads only games both parents walked.
OOS_MONTHS = 12

#: Per-refit RAM budget, including corpus tensors and the worker's retained caches.
REFIT_WORKER_GB = 8.0
#: Headroom beside the refits: the parent's backfill context and the workers' memo growth
#: over their cutoffs (two refits beside a 3 GB reserve ran a 34 GB box out of memory).
WALK_RESERVE_GB = 8.0

#: Per-refit CUDA budget for the set encoders and input batches; the pool never oversubscribes
#: the card, and a card that cannot hold one refit refuses the walk outright.
REFIT_VRAM_GB = 1.5


def assert_artifact_is_source() -> None:
    """Refuse a walk whose refits would write ``oos`` rows under another model's id.

    Raises:
        RuntimeError: the artifact's persisted recipe and the source recipe differ in
            version or kind.
    """
    stamped = artifacts.persisted_recipe("draft")
    if stamped is None:
        return
    source = draft_recipe.DRAFT_RECIPE.to_dict()
    read = (int(stamped["version"]), str(stamped.get("kind", "blend")))
    current = (int(source["version"]), str(source.get("kind", "blend")))
    if read != current:
        raise RuntimeError(
            f"the draft artifact persists recipe {read} but source is {current}: the walk "
            "refits source, so its rows would come from another model. Run `lol-predict "
            "retrain draft`."
        )


@dataclass(frozen=True, slots=True)
class DraftGameScore:
    """One walked game's blue-oriented score, the ``draft_scores`` row shape."""

    game_id: str
    p_blue: float
    grade_blue: str
    grade_red: str
    edge_pp_blue: float | None


def expected_oos_game_ids(conn: sqlite3.Connection, *, through: date, months: int) -> set[str]:
    """Every finished game with a loadable draft the walk must score exactly once."""
    # The month cut is served's own; the draft filter is bulk_load's (a game whose ten
    # roles do not resolve is unscorable on every draft path).
    through_iso = through.isoformat()
    ids: list[str] = []
    for start, end in trailing_months(through, months):
        rows = conn.execute(
            "SELECT g.game_id FROM games g JOIN series s ON s.series_id = g.series_id "
            "WHERE s.scheduled_ts >= ? AND s.scheduled_ts < ? AND s.scheduled_ts < ? "
            "AND g.winner IS NOT NULL AND g.blue_team IS NOT NULL",
            (start.isoformat(), end.isoformat(), through_iso),
        ).fetchall()
        ids.extend(str(r[0]) for r in rows)
    return set(DraftState.bulk_load(conn, ids))


def walk_worker_count(n_cutoffs: int) -> int:
    """Refits in flight: bounded by RAM, by free VRAM, and by the cutoff count.

    Raises:
        RuntimeError: the card cannot hold even one refit.
    """
    import torch

    by_ram = resources.bounded_workers(REFIT_WORKER_GB, reserve_gb=WALK_RESERVE_GB)
    if not torch.cuda.is_available():
        return max(1, min(by_ram, n_cutoffs))
    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    by_vram = int(free / 1e9 // REFIT_VRAM_GB)
    if by_vram < 1:
        raise RuntimeError(
            f"the draft OOS walk needs ~{REFIT_VRAM_GB:.1f} GB of free GPU memory per refit, "
            f"but only {free / 1e9:.1f} GB is free. Free GPU memory or use a larger CUDA GPU; "
            "no CPU fallback is supported."
        )
    return max(1, min(by_ram, by_vram, n_cutoffs))


def _score_window(
    predictor: DraftGrader, conn: sqlite3.Connection, *, since: date, until: date
) -> list[DraftGameScore]:
    """Score every finished game with a loadable draft in ``[since, until)``."""
    ids = [
        row[0]
        for row in conn.execute(
            "SELECT g.game_id FROM games g JOIN series s ON s.series_id = g.series_id "
            "WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) < ? "
            "AND g.blue_team IS NOT NULL AND g.winner IS NOT NULL",
            (since.isoformat(), until.isoformat()),
        )
    ]
    drafts = DraftState.bulk_load(conn, ids)
    present = [(gid, draft) for gid in ids if (draft := drafts.get(gid)) is not None]
    cache = BulkStatsCache(conn, baseline=predictor.pro_excess_baseline)
    scores = predictor.predict_batch([draft for _gid, draft in present], stats_cache=cache)
    return [
        DraftGameScore(
            gid, float(s.prob_blue_wins), str(s.grade_blue), str(s.grade_red), s.edge_pp_blue
        )
        for (gid, _draft), s in zip(present, scores, strict=True)
    ]


def _refit_and_score(
    cutoff: date, until: date, *, db_path: str, served_id: str, code_fingerprint: str
) -> list[DraftGameScore]:
    """One cutoff: refit the frozen recipe (its own kind) before ``cutoff``, score its month."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        probs = served_oos_probs(conn, served_id)
        with tempfile.TemporaryDirectory(prefix="draft_oos_") as tmp:
            root = Path(tmp) / "draft_OOS.json"
            started = time.perf_counter()
            fit_candidate(
                conn,
                root,
                fit_through=cutoff,
                production_probs=probs,
                code_fingerprint=code_fingerprint,
            )
            predictor = load_grader(conn, artifact=root)
            scores = _score_window(predictor, conn, since=cutoff, until=until)
            logger.info(
                "draft oos cutoff %s: refit + %d games scored in %.0fs",
                cutoff,
                len(scores),
                time.perf_counter() - started,
            )
        return scores
    finally:
        conn.close()


def walk_draft_oos(
    conn: sqlite3.Connection,
    *,
    through: date,
    served_id: str,
    code_fingerprint: str,
    months: int = OOS_MONTHS,
) -> list[DraftGameScore]:
    """Score each trailing month by a refit strictly before it; COST: ``months`` full refits.

    Raises:
        RuntimeError: the artifact the rows land under is not the model source refits.
        RuntimeError: from ``walk_worker_count``.
        RuntimeError: the walk's coverage differs from ``expected_oos_game_ids`` — a
            month's fit or scoring silently dropped games; never a partial write.
    """
    assert_artifact_is_source()
    db_path = next(row[2] for row in conn.execute("PRAGMA database_list") if row[1] == "main")
    slates = [
        (start, min(end, through))
        for start, end in trailing_months(through, months)
        if conn.execute(
            "SELECT 1 FROM games g JOIN series s ON s.series_id = g.series_id "
            "WHERE s.scheduled_ts >= ? AND s.scheduled_ts < ? AND s.scheduled_ts < ? "
            "AND g.winner IS NOT NULL LIMIT 1",
            (start.isoformat(), end.isoformat(), through.isoformat()),
        ).fetchone()
    ]
    if not slates:
        return []
    workers = walk_worker_count(len(slates))
    logger.info("draft oos walk: %d cutoffs, %d refits in flight", len(slates), workers)
    scores: list[DraftGameScore] = []
    refit = partial(
        _refit_and_score, db_path=db_path, served_id=served_id, code_fingerprint=code_fingerprint
    )
    if workers == 1:
        for start, until in slates:
            scores.extend(refit(start, until))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for month_scores in pool.map(refit, *zip(*slates, strict=True)):
                scores.extend(month_scores)
    expected = expected_oos_game_ids(conn, through=through, months=months)
    actual = {s.game_id for s in scores}
    if actual != expected or len(scores) != len(actual):
        missing = sorted(expected - actual)[:5]
        extra = sorted(actual - expected)[:5]
        raise RuntimeError(
            f"draft oos walk covered {len(actual)} games against {len(expected)} expected "
            f"(missing {missing}, unexpected {extra}, duplicates "
            f"{len(scores) - len(actual)}) — refusing a partial oos write."
        )
    return scores
