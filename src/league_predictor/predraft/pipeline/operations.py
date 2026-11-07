"""Fit and coverage operations for the served model."""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from league_predictor.asof import AsOf
from league_predictor.db.rows import PredictionRow
from league_predictor.predraft.pipeline.chosen_model import (
    ServedModel,
    chosen_model,
    fit_served_model,
)
from league_predictor.predraft.pipeline.live import build_live_indexes
from league_predictor.predraft.pipeline.model_contract import predict_symmetric
from league_predictor.predraft.pipeline.serve import (
    adjust_map_prob,
    blend_toward_rating,
    predict,
    predict_batch,
    with_series_lead,
)
from league_predictor.predraft.pipeline.served_oos import (
    OOS_MONTHS,
    _corpus_orientation_rows,
    expected_oos_game_ids,
    walk_served_oos,
)
from league_predictor.predraft.pipeline.walk_forward import build_fit_context

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Mapping, Sequence
    from datetime import date

    from league_predictor.predraft.features import FamilyIndex
    from league_predictor.predraft.pipeline.walk_forward import FeaturizedCorpus, FitContext

BACKFILL_GUARD_SAMPLES = 8
BACKFILL_GUARD_TOL = 1e-6
CORPUS_BACKFILL_GUARD_TOL = 1e-5


def build_context(conn: sqlite3.Connection, *, through: date, months: int) -> FitContext:
    """Build the full-history context shared by served operations."""
    return build_fit_context(conn, chosen_model(), through=through, months=months)


def fit_current_model(
    conn: sqlite3.Connection, *, through: date, ctx: FitContext | None = None
) -> ServedModel:
    """Fit the locked Served recipe through ``through``."""
    if ctx is None:
        ctx = build_context(conn, through=through, months=OOS_MONTHS)
    return fit_served_model(conn, through=through, ctx=ctx)


def prediction_rows(
    scores: Iterable[ServedGameScore], model_id: str, context: str, created_at: str
) -> list[PredictionRow]:
    """Stamp served game scores into lifecycle-shaped prediction rows."""
    return [
        PredictionRow(
            subject_kind="game",
            subject_id=score.game_id,
            model_id=model_id,
            p_a=score.p_a,
            asof_date=score.asof_date,
            context=context,
            created_at=created_at,
        )
        for score in scores
    ]


def build_prd_coverage(
    conn: sqlite3.Connection, model: ServedModel, *, through: date, ctx: FitContext
) -> tuple[list[ServedGameScore], list[ServedGameScore]]:
    """Score the complete backfill corpus and walk-forward ``oos`` window off one context.

    Returns:
        ``(backfill_scores, oos_scores)``, each strict-coverage-checked against the games
        its contract owns.
    """
    backfill_scores = list(score_backfill(conn, model, indexes=ctx.indexes, corpus=ctx.corpus))
    oos_scores = list(materialize_oos(conn, through=through, ctx=ctx))
    expected_backfill = {str(game["game_id"]) for game in dated_games(conn)}
    require_complete_coverage("backfill", backfill_scores, expected_backfill)
    require_complete_coverage("oos", oos_scores, expected_oos_game_ids(conn, through=through))
    return backfill_scores, oos_scores


def require_complete_coverage(
    context: str, scores: Collection[ServedGameScore], expected_ids: set[str]
) -> None:
    """Require one unique valid-probability score for every game a coverage contract owns."""
    invalid_probabilities = [
        score.game_id
        for score in scores
        if isinstance(score.p_a, bool)
        or not isinstance(score.p_a, (int, float))
        or not math.isfinite(float(score.p_a))
        or not 0.0 <= float(score.p_a) <= 1.0
    ]
    if invalid_probabilities:
        raise RuntimeError(
            f"strict {context} coverage contains invalid probability values "
            f"({invalid_probabilities[:3]})"
        )
    game_ids = [str(score.game_id) for score in scores]
    if len(game_ids) != len(set(game_ids)):
        raise RuntimeError(f"strict {context} coverage contains duplicate game IDs")
    actual_ids = set(game_ids)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        unexpected = sorted(actual_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing {len(missing)} IDs ({missing[:3]})")
        if unexpected:
            details.append(f"unexpected {len(unexpected)} IDs ({unexpected[:3]})")
        raise RuntimeError(f"strict {context} coverage is incomplete: {', '.join(details)}")


@dataclass(frozen=True, slots=True)
class ServedGameScore:
    """One canonical served probability ready for lifecycle persistence."""

    game_id: str
    p_a: float
    asof_date: str


def dated_games(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return every dated game in canonical facts order."""
    return conn.execute(
        "SELECT g.game_id AS game_id, s.team_a AS team_a, s.team_b AS team_b,"
        " s.scheduled_ts AS scheduled_ts, s.series_id AS series_id, g.game_no AS game_no"
        " FROM games g JOIN series s ON s.series_id = g.series_id"
        " WHERE s.scheduled_ts IS NOT NULL"
        " ORDER BY s.scheduled_ts, s.series_id, g.game_no"
    ).fetchall()


def score_game(
    model: ServedModel, indexes: Mapping[str, FamilyIndex], team_a: str, team_b: str, *, as_of: AsOf
) -> float:
    """Score one matchup through the canonical live served chain."""
    return float(predict(model, indexes, team_a, team_b, as_of=as_of))


def score_games(
    model: ServedModel,
    indexes: Mapping[str, FamilyIndex],
    matchups: list[tuple[str, str, AsOf]],
) -> list[float]:
    """Score many matchups through the batched served chain."""
    return [float(prob) for prob in predict_batch(model, indexes, matchups)]


def guard_batched_backfill(
    model: ServedModel,
    indexes: Mapping[str, FamilyIndex],
    matchups: Sequence[tuple[str, str, AsOf]],
    probabilities: Sequence[float],
    *,
    tolerance: float = BACKFILL_GUARD_TOL,
) -> None:
    """Fail when batched backfill probabilities diverge from live scoring."""
    step = max(1, len(matchups) // BACKFILL_GUARD_SAMPLES)
    for index in range(0, len(matchups), step)[:BACKFILL_GUARD_SAMPLES]:
        team_a, team_b, as_of = matchups[index]
        reference = score_game(model, indexes, team_a, team_b, as_of=as_of)
        if abs(probabilities[index] - reference) > tolerance:
            raise RuntimeError(
                f"batched backfill prob {probabilities[index]} diverged from serve.predict "
                f"{reference} for {team_a} vs {team_b} at {as_of.key}"
            )


def _score_games_from_corpus(
    model: ServedModel, indexes, games: list[sqlite3.Row], *, matchups, corpus
) -> list[float]:
    """Score covered games from persisted orientation rows and misses through live indexes."""
    import numpy as np

    orientation = _corpus_orientation_rows(corpus)
    covered_positions: list[int] = []
    ab_indices: list[int] = []
    ba_indices: list[int] = []
    missing_positions: list[int] = []
    for pos, game in enumerate(games):
        pair = orientation.get(str(game["game_id"]))
        if pair is None:
            missing_positions.append(pos)
            continue
        i0, i1 = pair
        if str(game["team_a"]) > str(game["team_b"]):
            i0, i1 = i1, i0
        covered_positions.append(pos)
        ab_indices.append(i0)
        ba_indices.append(i1)

    probabilities = np.empty(len(games), dtype=np.float64)
    if covered_positions:
        ab_rows = corpus._rows.take(ab_indices)
        ba_rows = corpus._rows.take(ba_indices)
        bare = predict_symmetric(model.bag, ab_rows, ba_rows)

        def symmetrized(name: str):
            col = corpus.feature_names.index(name)
            return 0.5 * (corpus._matrix[ab_indices, col] - corpus._matrix[ba_indices, col])

        rating_logits = symmetrized("rating_logit_diff")
        leads = symmetrized("diff_series_wins") if model.series_lead else rating_logits * 0.0
        blended = [
            with_series_lead(
                blend_toward_rating(float(prob), float(rating), model.rating_blend),
                float(lead),
                model.series_lead,
            )
            for prob, rating, lead in zip(bare, rating_logits, leads, strict=True)
        ]
        for pos, prob in zip(covered_positions, blended, strict=True):
            team_a, team_b, as_of = matchups[pos]
            probabilities[pos] = adjust_map_prob(
                float(prob),
                model.league,
                indexes,
                team_a,
                team_b,
                as_of,
                temperature=model.temperature,
                damp_facts_games=model.damp_facts_games,
            )
    if missing_positions:
        missing = score_games(model, indexes, [matchups[index] for index in missing_positions])
        probabilities[missing_positions] = missing
    return probabilities.tolist()


def score_backfill(
    conn: sqlite3.Connection,
    model: ServedModel,
    *,
    indexes: Mapping[str, FamilyIndex] | None = None,
    corpus: FeaturizedCorpus | None = None,
) -> list[ServedGameScore]:
    """Score every dated game canonically, each at its own query, for backfill persistence."""
    indexes = build_live_indexes(conn, model) if indexes is None else indexes
    games = dated_games(conn)
    matchups = [
        (g["team_a"], g["team_b"], AsOf.game(g["scheduled_ts"], g["series_id"], g["game_no"]))
        for g in games
    ]
    probabilities = (
        score_games(model, indexes, matchups)
        if corpus is None
        else _score_games_from_corpus(model, indexes, games, matchups=matchups, corpus=corpus)
    )
    guard_batched_backfill(
        model,
        indexes,
        matchups,
        probabilities,
        tolerance=BACKFILL_GUARD_TOL if corpus is None else CORPUS_BACKFILL_GUARD_TOL,
    )
    return [
        ServedGameScore(str(game["game_id"]), float(probability), matchup[2].day.isoformat())
        for game, matchup, probability in zip(games, matchups, probabilities, strict=True)
    ]


def score_oos(
    conn: sqlite3.Connection,
    *,
    through: date,
    ctx: FitContext | None = None,
    months: int = OOS_MONTHS,
) -> list[ServedGameScore]:
    """Walk the leak-free served OOS corpus into canonical game scores."""
    return [
        ServedGameScore(str(match_id), float(p_a), str(when)[:10])
        for match_id, when, p_a, _outcome in walk_served_oos(
            conn, through=through, months=months, ctx=ctx
        )
    ]


def materialize_oos(
    conn: sqlite3.Connection, *, through: date, ctx: FitContext | None
) -> list[ServedGameScore]:
    """Recompute the complete OOS product under the fixed recipe."""
    return score_oos(conn, through=through, ctx=ctx, months=OOS_MONTHS)
