"""Fit the candidate."""  # sightline-ok: 27 - one boundary composes corpus, fit and artifact owners

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from league_predictor.draft_grading import recipe as draft_recipe
from league_predictor.draft_grading.champion_attributes.ability_store import AbilityFeatureStore
from league_predictor.draft_grading.champion_attributes.power_store import (
    DEFAULT_POWER_CACHE,
    load_power_grid,
)
from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.grader.calibration import apply_temperature, fit_temperature
from league_predictor.draft_grading.grader.dense_labels import dense_targets
from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from league_predictor.draft_grading.grader.grades import (
    EDGE_CUT_A_PP,
    EDGE_CUT_S_PP,
    fit_edge_curve,
)
from league_predictor.draft_grading.grader.knowledge_tables import (
    KnowledgeTables,
    build_knowledge_tables,
)
from league_predictor.draft_grading.grader.loss import compute_importance_weights
from league_predictor.draft_grading.grader.oof_calibration import (
    OofCalibration,
    calibration_tail_start,
    fold_windows,
)
from league_predictor.draft_grading.grader.power_curve_features import build_power_tables
from league_predictor.draft_grading.grader.pro_excess import team_form_offsets
from league_predictor.draft_grading.grader.set_encoder import DraftSetEncoder
from league_predictor.draft_grading.grader.set_training import (
    SetWeights,
    fit_set_encoder,
    set_scores,
)
from league_predictor.draft_grading.grader.slot_inputs import SlotInputs
from league_predictor.draft_grading.set_variant import SetDraftPredictor
from league_predictor.draft_grading.state import DraftState
from league_predictor.draft_grading.training.provenance import stamp_fit_provenance
from league_predictor.probability import sigmoid_array

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping
    from datetime import date

    import torch

    from league_predictor.draft_grading.recipe import DraftRecipe, SetTrainConfig

logger = logging.getLogger(__name__)


class SetCorpus(NamedTuple):
    """One fit's chronological rows: the boards, their outcomes, and the importance weights."""

    match_ids: list[str]
    drafts: list[DraftState]
    outcomes: np.ndarray
    weights: np.ndarray
    skipped: dict[str, int]


OofPool = tuple[list[str], np.ndarray, np.ndarray]
SetFitResult = tuple[SetWeights, KnowledgeTables, OofCalibration, OofPool]


def load_set_corpus(
    conn: sqlite3.Connection,
    *,
    since: str,
    until: str,
    production_probs: Mapping[str, float] | None = None,
) -> SetCorpus:
    """Finished blue-sided games in ``[since, until)`` in facts order, with their weights.

    Args:
        production_probs: served prior per game; a game without one weighs exactly 1.0.
    """
    rows = conn.execute(
        "SELECT g.game_id, g.winner FROM games g JOIN series s ON s.series_id = g.series_id "
        "WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) < ? "
        "AND g.blue_team IS NOT NULL AND g.winner IS NOT NULL "
        "ORDER BY date(s.scheduled_ts), g.game_id",
        (since, until),
    ).fetchall()
    winners = {str(row[0]): row[1] for row in rows}
    loaded = DraftState.bulk_load(conn, list(winners))
    skipped = {"corrupt": 0}
    match_ids: list[str] = []
    drafts: list[DraftState] = []
    outcomes: list[float] = []
    for match_id, winner in winners.items():
        draft = loaded.get(match_id)
        if draft is None:
            skipped["corrupt"] += 1
        else:
            match_ids.append(match_id)
            drafts.append(draft)
            outcomes.append(float(winner == draft.blue.team_id))

    y = np.array(outcomes, dtype=np.float64)
    weights = np.ones(len(match_ids))
    priors = production_probs or {}
    covered = np.array([match_id in priors for match_id in match_ids], dtype=bool)
    if covered.any():
        weights[covered] = compute_importance_weights(
            np.array([priors[m] for m, c in zip(match_ids, covered, strict=True) if c]), y[covered]
        )
    logger.info(
        "set corpus: %d games in [%s, %s), %d with a served prior, skipped %s",
        len(match_ids),
        since,
        until,
        int(covered.sum()),
        skipped,
    )
    return SetCorpus(match_ids, drafts, y, weights, skipped)


def build_set_knowledge(config: SetTrainConfig) -> tuple[KnowledgeTables, ChampionIndex]:
    """Freeze the fit's champion knowledge: the z-matrix always, kit and power per the flags."""
    store = AbilityFeatureStore()
    index = ChampionIndex(store)
    power = None
    if config.power:
        power = build_power_tables(index, load_power_grid(DEFAULT_POWER_CACHE))
    return (
        build_knowledge_tables(index, store, kit=config.kit, power=power),
        index,
    )


def _oof_calibration(
    inputs: torch.Tensor,
    corpus: SetCorpus,
    *,
    offsets: np.ndarray | None,
    dense,
    config: SetTrainConfig,
    n_champions: int,
) -> tuple[OofCalibration, OofPool]:
    """Refit each expanding fold on its own prefix and pool the folds' held-out scores."""
    n = len(corpus.match_ids)
    tail = calibration_tail_start(n)
    windows = fold_windows(tail, n, config.oof_folds)
    probs: list[np.ndarray] = []
    epochs: list[int] = []
    for fold_start, fold_end in windows:
        prefix = slice(0, fold_start)
        weights = fit_set_encoder(
            inputs[prefix],
            corpus.outcomes[prefix],
            None if offsets is None else offsets[prefix],
            None if dense is None else dense.take(prefix),
            config,
            weights=corpus.weights[prefix],
            n_champions=n_champions,
            device=config.device,
        )
        epochs.extend(int(report["best_epoch"]) for report in weights.history)
        # The score is exactly antisymmetric, so sigmoid of the seed mean IS the
        # symmetrized probability; there is no second orientation to average.
        probs.append(sigmoid_array(set_scores(weights, config, inputs[fold_start:fold_end])))
    pool = np.concatenate(probs) if probs else np.array([])
    pool_ids = [m for start, end in windows for m in corpus.match_ids[start:end]]
    pool_y = np.concatenate([corpus.outcomes[start:end] for start, end in windows])
    temperature = fit_temperature(pool, pool_y)  # pyright: ignore[reportArgumentType]  ndarray ok at runtime
    curve = fit_edge_curve(np.asarray(apply_temperature(pool, temperature), dtype=float), pool_y)
    return (
        OofCalibration(
            int(np.median(epochs)) if epochs else 0,
            temperature,
            list(curve.thresholds(EDGE_CUT_S_PP, EDGE_CUT_A_PP)),
            len(windows),
            int(pool.size),
            curve,
            pool_probs=pool,
        ),
        (pool_ids, pool, pool_y),
    )


def fit_set_candidate(
    conn: sqlite3.Connection,
    config: SetTrainConfig,
    *,
    through: date,
    production_probs: Mapping[str, float] | None = None,
    stats_cache: BulkStatsCache | None = None,
) -> SetFitResult:
    """Fit one set-encoder candidate through ``through`` (exclusive) and calibrate it.

    Returns:
        The five-seed weights, the knowledge member they were fit against, the OOF-fit
        temperature/curve/thresholds, and the tail pool behind them.

    Raises:
        RuntimeError: the corpus window holds no loadable game.
    """
    # DECLARED COST: (oof_folds + 1) x seeds encoder fits over the whole corpus, minutes on
    # the GPU at the production recipe; the corpus tensor is about 4 KB a game, 350 MB.

    started = time.perf_counter()
    knowledge, index = build_set_knowledge(config)
    corpus = load_set_corpus(
        conn,
        since=config.outcome_since,
        until=through.isoformat(),
        production_probs=production_probs,
    )
    if not corpus.match_ids:
        raise RuntimeError(
            f"set fit: no loadable game in [{config.outcome_since}, {through}) — the fit "
            f"window is empty (skipped {corpus.skipped})."
        )
    if stats_cache is None:
        stats_cache = BulkStatsCache(conn)
    inputs = SlotInputs(config).build(corpus.drafts, knowledge, stats_cache)
    offsets = team_form_offsets(conn, corpus.match_ids) if config.outcome_offset != "none" else None
    dense = dense_targets(conn, corpus.match_ids) if config.aux_heads else None
    # ``len(index)`` already counts the UNKNOWN row; a spare row nothing indexes would
    # leave the real UNKNOWN trainable.
    n_champions = len(index)
    logger.info(
        "set fit: %d games, input %s, prepared in %.1fs",
        len(corpus.match_ids),
        tuple(inputs.shape),
        time.perf_counter() - started,
    )

    calibration, pool = _oof_calibration(
        inputs, corpus, offsets=offsets, dense=dense, config=config, n_champions=n_champions
    )
    # The folds chose the epoch on rows outside the calibration tail, so the final fit runs
    # exactly that many over every row: it never early-stops inside the pool it is graded on.
    weights = fit_set_encoder(
        inputs,
        corpus.outcomes,
        offsets,
        dense,
        config,
        weights=corpus.weights,
        n_champions=n_champions,
        epochs=max(1, calibration.chosen_rounds),
        device=config.device,
    )
    logger.info(
        "set fit: done in %.1fs, %d epochs, temperature %.4f over %d pooled tail games",
        time.perf_counter() - started,
        calibration.chosen_rounds,
        calibration.temperature,
        calibration.n_pool,
    )
    return weights, knowledge, calibration, pool


def save_set_candidate(root: Path, result: SetFitResult, *, recipe: DraftRecipe) -> None:
    """Write ``result`` as a format-4 family, through the predictor that will serve it.

    Args:
        recipe: the recipe the fit ran under, stamped verbatim into the top meta.
    """
    # Building the predictor runs its antisymmetry gate, so a seed whose half-swap does
    # not negate its score is refused here rather than at serve time.
    weights, knowledge, calibration, _pool = result
    config = recipe.set_recipe()
    modules = []
    for state in weights.states:
        module = DraftSetEncoder(config, weights.d_in, weights.n_champions)
        module.load_state_dict(state)
        modules.append(module.eval())
    SetDraftPredictor(
        modules,
        weights=weights.to_state(),
        build_inputs=SlotInputs.build_for(weights.input_spec, knowledge),
        champion_index=ChampionIndex.from_names(list(knowledge.champions)),
        thresholds=calibration.thresholds,
        temperature=calibration.temperature,
        edge_curve=calibration.edge_curve,
        edge_cuts_pp=(EDGE_CUT_S_PP, EDGE_CUT_A_PP),
        recipe=recipe.to_dict(),
        knowledge=knowledge,
    ).save(Path(root))


def fit_candidate(
    conn: sqlite3.Connection,
    output_root: Path,
    *,
    fit_through: date,
    production_probs: Mapping[str, float],
    code_fingerprint: str,
) -> None:
    """Fit the frozen production set recipe without publishing. COST: one GPU fit."""
    recipe = draft_recipe.DRAFT_RECIPE
    result = fit_set_candidate(
        conn,
        recipe.set_recipe(),
        through=fit_through,
        production_probs=production_probs,
    )
    save_set_candidate(Path(output_root), result, recipe=recipe)
    stamp_fit_provenance(
        Path(output_root),
        fit_through,
        code_fingerprint=code_fingerprint,
    )
