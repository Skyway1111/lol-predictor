"""Walk-forward OOS scoring for the fixed Served recipe; its output rows are never persisted."""

from __future__ import annotations

import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

from league_predictor.asof import AsOf
from league_predictor.clock import trailing_months
from league_predictor.predraft.features import build_features
from league_predictor.predraft.pipeline.chosen_model import (
    bag_backend,
    chosen_model,
)
from league_predictor.predraft.pipeline.model_contract import predict_symmetric
from league_predictor.predraft.pipeline.serve import (
    adjust_map_prob,
    blend_toward_rating,
    game_prob,
    rating_logit,
    series_lead,
    with_series_lead,
)
from league_predictor.predraft.pipeline.walk_forward import build_fit_context, fit_window
from league_predictor.predraft.ratings.region_rating import fit_region_ratings

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator
    from datetime import date

    from league_predictor.predraft.pipeline.walk_forward import FitContext

logger = logging.getLogger(__name__)

# Trailing coverage span. Twelve months bounds the refresh at twelve monthly bag fits.
OOS_MONTHS = 12

# How many corpus-scored maps to cross-check against the live ``game_prob`` path per walk. The
# two are float-identical, so a handful of checks catches orientation/eligibility drift cheaply.
_OOS_CORPUS_GUARD_SAMPLES = 8

# Window-fit prefetch, corpus path only. Width 3 is the probed saturation on 16 cores (width 4
# regresses); depth 4 keeps the pool fed while bounding live memory to ~4 fitted bags.
_OOS_FIT_WORKERS = 3
_OOS_FIT_PREFETCH = 4

# A scored match contributes one corpus row per orientation.
_CORPUS_ROWS_PER_MATCH = 2
# The corpus shortcut and the live path are float-identical; anything wider is a divergence.
_PARITY_TOL = 1e-9

#: A scored map's key: its canonical pair and its own query point.
_MapKey = tuple[str, str, AsOf]


def expected_oos_game_ids(
    conn: sqlite3.Connection,
    *,
    through: date,
    months: int = OOS_MONTHS,
) -> set[str]:
    """Return every finished game the OOS walk must materialize exactly once."""
    through_iso = through.isoformat()
    game_ids: set[str] = set()
    for month_start, month_end in trailing_months(through, months):
        rows = conn.execute(
            "SELECT g.game_id FROM games g JOIN series s ON s.series_id=g.series_id "
            "WHERE s.scheduled_ts >= ? AND s.scheduled_ts < ? AND s.scheduled_ts < ? "
            "AND g.winner IS NOT NULL",
            (month_start.isoformat(), month_end.isoformat(), through_iso),
        ).fetchall()
        game_ids.update(str(row[0]) for row in rows)
    return game_ids


def _fit_worker_count(config, n_slates: int) -> int:
    """OOS fit width, bounded by cores, per-window bag width, and 12 concurrent windows."""
    members = max(1, len(config.seeds))
    core_width = max(_OOS_FIT_WORKERS, ((os.cpu_count() or 4) - 2) // members)
    return min(n_slates, 12, core_width)


def _corpus_orientation_rows(corpus) -> dict[str, tuple[int, int]]:
    """``match_id -> (sorted-orientation index, swapped index)`` into the corpus matrix."""
    by_id: dict[str, list[int]] = {}
    for i, meta in enumerate(corpus._meta):
        by_id.setdefault(str(meta["match_id"]), []).append(i)
    return {mid: (ix[0], ix[1]) for mid, ix in by_id.items() if len(ix) == _CORPUS_ROWS_PER_MATCH}


def _oriented_corpus_rows(
    corpus, indices: tuple[int, int], team1_id, team2_id
) -> tuple[dict, dict]:
    """A corpus map's two rows as fresh ``(x_ab, x_ba)`` feature dicts, team1 as team_a."""
    i0, i1 = indices
    if not (str(team1_id) <= str(team2_id)):
        i0, i1 = i1, i0
    return corpus._rows[i0], corpus._rows[i1]


def _month_slates(
    conn: sqlite3.Connection, *, through: date, months: int
) -> list[tuple[date, list[sqlite3.Row]]]:
    """Each trailing month that has games, oldest first, with its finished maps.

    Returns:
        ``(month start, games)`` pairs; ``< through`` caps the final month at the corpus's
        exclusive ``end``, calendar-day exclusive. Months without games are dropped, so they
        never fit.
    """
    through_iso = through.isoformat()
    slates: list[tuple[date, list[sqlite3.Row]]] = []
    for m_start, m_end in trailing_months(through, months):
        games = conn.execute(
            "SELECT g.game_id AS match_id, s.scheduled_ts AS date,"
            " s.series_id AS series_id, g.game_no AS game_no,"
            " s.team_a AS team1_id, s.team_b AS team2_id, g.winner AS winner"
            " FROM games g JOIN series s ON s.series_id = g.series_id"
            " WHERE s.scheduled_ts >= ? AND s.scheduled_ts < ? AND s.scheduled_ts < ?"
            " AND g.winner IS NOT NULL"
            " ORDER BY s.scheduled_ts, s.series_id, g.game_no",
            (m_start.isoformat(), m_end.isoformat(), through_iso),
        ).fetchall()
        if games:
            slates.append((m_start, games))
    return slates


def _map_key(g: sqlite3.Row) -> _MapKey:
    """A slate row's scoring key: the canonical pair at the map's own query point."""
    return (g["team1_id"], g["team2_id"], AsOf.game(g["date"], g["series_id"], g["game_no"]))


def _submit_prefetch(pool: ThreadPoolExecutor, fit_futures: dict, upcoming, fit) -> None:
    """Queue a window fit for each upcoming month slate not already in flight."""
    for m_next, _ in upcoming:
        if m_next not in fit_futures:
            fit_futures[m_next] = pool.submit(fit, m_next)


def _month_queries(
    games, corpus, corpus_rows: None | dict[str, tuple[int, int]]
) -> tuple[dict, dict]:
    """One query per map, and the corpus rows that cover it.

    Returns:
        ``(by_key, stored_rows)``, both keyed by ``(team1, team2, as_of)`` in slate order.
        ``stored_rows`` holds only the maps the corpus stores, as their two oriented feature
        dicts.
    """
    by_key: dict[_MapKey, sqlite3.Row] = {_map_key(g): g for g in games}
    stored_rows = {
        key: _oriented_corpus_rows(
            corpus, corpus_rows[str(g["match_id"])], g["team1_id"], g["team2_id"]
        )
        for key, g in by_key.items()
        if corpus_rows is not None and str(g["match_id"]) in corpus_rows
    }
    return by_key, stored_rows


def _base_terms(x_ab, x_ba) -> tuple[float, float]:
    """A map's symmetrized rating logit and series lead, the served base's two inputs."""
    return rating_logit(x_ab, x_ba), series_lead(x_ab, x_ba)


def _batch_score_stored(stored_rows, month_start: date, *, fitted, blend) -> tuple[dict, dict]:
    """Score every corpus-stored map as one month-wide symmetric predict.

    Returns:
        ``(prob by key, base terms by key)``. Predict is row-independent, so the batch is
        bit-for-bit the per-map calls it replaces. The ``_base_terms`` are filled only under
        ``blend``, always from the same feats this path scored. A failure logs and NaNs the
        whole batch — the same contract as the per-map path, never fatal to the walk.
    """
    if not stored_rows:
        return {}, {}
    ab_rows = [ab for ab, _ in stored_rows.values()]
    ba_rows = [ba for _, ba in stored_rows.values()]
    cache: dict[_MapKey, float] = {}
    try:
        probs = predict_symmetric(fitted, ab_rows, ba_rows)
        cache.update(zip(stored_rows, (float(p) for p in probs), strict=True))
    except Exception as exc:  # noqa: BLE001 - scoring failures NaN, never kill the walk
        logger.warning(
            "Served OOS batched scoring failed for %d maps in %s: %s",
            len(stored_rows),
            month_start,
            exc,
        )
        cache.update((key, float("nan")) for key in stored_rows)
    if not blend:
        return cache, {}
    zipped = zip(stored_rows, ab_rows, ba_rows, strict=True)
    return cache, {key: _base_terms(ab, ba) for key, ab, ba in zipped}


def _guard_corpus_parity(stored_rows, by_key, cache, budget: int, *, fitted, ctx) -> int:
    """Cross-check a budgeted sample of corpus-scored maps against the live ``game_prob``.

    Returns:
        The budget left. Called outside the batch's ``try`` so a divergence is never swallowed
        as a scoring failure.

    Raises:
        RuntimeError: a sampled map's corpus-derived probability differs from the live path,
            meaning the corpus-reuse shortcut has diverged. A hard failure, not ``assert``
            (which ``python -O`` strips).
    """
    family_names = [f.name for f in ctx.fams]
    for key in stored_rows:
        if budget <= 0 or cache[key] != cache[key]:  # batch failed; nothing to cross-check
            break
        budget -= 1
        g = by_key[key]
        t1, t2, as_of = key
        p_fresh = float(game_prob(fitted, ctx.indexes, t1, t2, as_of, families=family_names))
        if abs(cache[key] - p_fresh) > _PARITY_TOL:
            raise RuntimeError(
                f"corpus-derived OOS prob {cache[key]} diverged from game_prob "
                f"{p_fresh} for {g['match_id']} ({t1} vs {t2} at {as_of.key})"
            )
    return budget


def _score_uncached(by_key, cache, *, fitted, ctx, config) -> tuple[dict, dict]:
    """From-DB featurization and scoring for the maps ``cache`` has no probability for.

    Returns:
        ``(prob by key, base terms by key)`` for those maps only — maps absent from the
        corpus, i.e. not training-eligible. Under a blend the feats are built explicitly for
        the base terms. A map whose scoring raises is logged and scored NaN.
    """
    idx = ctx.indexes
    family_names = [f.name for f in ctx.fams]
    blend = _blends(config)
    probs: dict[_MapKey, float] = {}
    base_terms: dict[_MapKey, tuple[float, float]] = {}
    for key in by_key:
        if key in cache:
            continue
        t1, t2, as_of = key
        try:
            if not blend:
                probs[key] = float(game_prob(fitted, idx, t1, t2, as_of, families=family_names))
            else:
                x_ab = build_features(t1, t2, as_of, indexes=idx, families=family_names)
                x_ba = build_features(t2, t1, as_of, indexes=idx, families=family_names)
                probs[key] = float(predict_symmetric(fitted, [x_ab], [x_ba])[0])
                base_terms[key] = _base_terms(x_ab, x_ba)
        except Exception as exc:  # noqa: BLE001 - scoring failures NaN, never kill the walk
            logger.warning(
                "Served OOS scoring failed for %s vs %s at %s: %s", t1, t2, as_of.key, exc
            )
            probs[key] = float("nan")
    return probs, base_terms


def _blends(config) -> bool:
    """Whether the recipe's base reads its rating logit or series lead from the feats."""
    return bool(config.rating_blend or config.series_lead)


def _served_probs(cache, base_terms, league, *, idx) -> dict[_MapKey, float]:
    """Apply the fixed base and post-base transforms live serving applies, per map.

    Returns:
        One probability per map key; a failed map (the NaN sentinel) passes through
        untouched. The base blends toward the rating and adds the series lead through the
        same ``serve`` primitives the live path uses, before the post-base transform.
    """
    recipe = chosen_model()
    served: dict[_MapKey, float] = {}
    for key in cache:
        if cache[key] != cache[key]:  # failed map (NaN sentinel)
            served[key] = cache[key]
            continue
        base = cache[key]
        if _blends(recipe):
            z, lead = base_terms[key]
            base = blend_toward_rating(base, z, recipe.rating_blend)
            base = with_series_lead(base, lead, recipe.series_lead)
        served[key] = adjust_map_prob(
            base,
            league,
            idx,
            key[0],
            key[1],
            key[2],
            temperature=recipe.temperature,
            damp_facts_games=recipe.damp_facts_games,
        )
    return served


def walk_served_oos(
    conn: sqlite3.Connection,
    *,
    through: date,
    months: int = OOS_MONTHS,
    ctx: FitContext | None = None,
) -> Iterator[tuple[str, str, float, int]]:
    """Yield fixed-recipe served-OOS rows oldest month first, fitting each month on data before it.

    Args:
        ctx: a ``walk_forward.FitContext`` supplying shared indexes and the featurized corpus;
            omitted, one is built here so the corpus-reuse scoring path still runs.
    Yields:
        ``(game_id, date, prob, outcome)``. ``prob`` is team1's side-agnostic win probability
        through the full serve transform at the map's own query point, ``date`` the map's
        ``series.scheduled_ts`` string, ``outcome`` 1 iff team1 won. No map dated on or after
        ``through`` is emitted; an empty training window skips the month and a map whose
        scoring raises is logged and skipped.
    """
    config = chosen_model()
    if ctx is None:
        ctx = build_fit_context(conn, config, through=through, months=months)
    idx, drop, corpus = ctx.indexes, ctx.drop, ctx.corpus
    backend = bag_backend()
    blend = _blends(config)
    corpus_rows = _corpus_orientation_rows(corpus) if corpus is not None else None
    guard_budget = _OOS_CORPUS_GUARD_SAMPLES
    # Slates first, so ``conn`` stays on this thread while the fits run on the pool.
    slates = _month_slates(conn, through=through, months=months)

    def _fit(train_end: date):
        return fit_window(
            backend,
            train_end=train_end,
            window=config.window,
            drop_features=drop,
            corpus=corpus,
        )

    # Each fit is deterministic regardless of scheduling, so prefetch is byte-identical to the
    # serial walk. A corpus=None ctx stays serial: its fit path reads the single-thread ``conn``.
    fit_workers = _fit_worker_count(config, len(slates))
    pool = (
        ThreadPoolExecutor(max_workers=fit_workers)
        if corpus is not None and len(slates) > 1
        else None
    )
    fit_futures: dict[date, Future] = {}
    try:
        for i, (month_start, games) in enumerate(slates):
            if pool is not None:
                prefetch = max(_OOS_FIT_PREFETCH, fit_workers + 1)
                _submit_prefetch(pool, fit_futures, slates[i : i + prefetch], _fit)
                fitted = fit_futures.pop(month_start).result()
            else:
                fitted = _fit(month_start)
            if fitted is None:
                continue
            league = fit_region_ratings(conn, through=month_start).with_betas(
                cross_tier=config.region_beta_cross_tier,
                within_tier1=config.region_beta_within_tier1,
            )

            by_key, stored_rows = _month_queries(games, corpus, corpus_rows)
            cache, base_terms = _batch_score_stored(
                stored_rows, month_start, fitted=fitted, blend=blend
            )
            guard_budget = _guard_corpus_parity(
                stored_rows, by_key, cache, guard_budget, fitted=fitted, ctx=ctx
            )
            probs, logits = _score_uncached(by_key, cache, fitted=fitted, ctx=ctx, config=config)
            cache.update(probs)
            base_terms.update(logits)
            served = _served_probs(cache, base_terms, league, idx=idx)

            for key, g in by_key.items():
                if served[key] == served[key]:  # skip failed maps (NaN sentinel)
                    yield (
                        g["match_id"],
                        g["date"],
                        served[key],
                        1 if g["winner"] == g["team1_id"] else 0,
                    )
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
