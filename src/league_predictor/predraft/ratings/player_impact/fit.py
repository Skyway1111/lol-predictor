"""DB-facing Impact orchestration: prepare one cutoff, then fit and publish it."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.predraft.ratings.impact_corpus import (
    build_impact_corpus,
    prepare_impact_window,
)
from league_predictor.predraft.ratings.player_impact.design import (
    _ALPHA_KEY,
    _BETA_KEY,
    _GAMMA_KEY,
    _LAMBDA_CANDIDATES,
    _assign_columns,
    _build_design,
    _extend_full_history_columns,
    _fit_masked_ridge,
    _select_lambda,
    replacement_key,
)
from league_predictor.predraft.ratings.player_impact.predictor import (
    _DEFAULT_ROLE,
    ImpactValues,
    RegionLevels,
)
from league_predictor.predraft.ratings.player_impact.recipe import (
    IMPACT_RECIPE,
    PRIOR_BLOCKS,
    STAT_CONTEXT_KWARGS,
)
from league_predictor.predraft.ratings.player_impact.results import (
    ImpactEstimate,
    ImpactFit,
    ImpactFitContext,
    _percentile_of,
    _value_to_pp,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping, Sequence
    from datetime import date

    from league_predictor.predraft.ratings.impact_corpus import ImpactCorpus
    from league_predictor.predraft.ratings.player_impact.recipe import ImpactRecipe
    from league_predictor.predraft.ratings.region_rating import RegionRatingTable

#: Drift detector on ``REGION_NORMALIZATION_MAP``: a drop means its keys stopped matching the
#: facts' league_id spellings. 0.18 is two-thirds of the 0.276 observed at introduction.
_MIN_REGION_COVERAGE = 0.18


def _latest_domestic_league(
    conn: sqlite3.Connection,
    as_of_date: date,
    *,
    exclude_international: bool,
) -> list[tuple[str, str]]:
    """``(player_id, league_id)`` of every player's most-recent game strictly before the cutoff.

    Args:
        exclude_international: drop WLDs/MSI/FST/EWC series, so an autumn cutoff reads a
            player's home league rather than the tournament they last travelled to. Matches
            the exclusion ``snapshots/stream.py`` applies to team regions.
    """
    from league_predictor.predraft.league_tier import INTERNATIONAL_LEAGUES

    clause, params = "", ()
    if exclude_international:
        clause = f" AND s.league_id NOT IN ({', '.join('?' * len(INTERNATIONAL_LEAGUES))})"
        params = tuple(INTERNATIONAL_LEAGUES)
    # Latest tournament per player resolved in SQL — one window query instead of scanning
    # ~900k rows; canonical (series_id, game_no) order breaks same-day ties.
    return conn.execute(
        f"""
        SELECT player_id, tournament FROM (
            SELECT gp.player_id, s.league_id AS tournament,
                   ROW_NUMBER() OVER (PARTITION BY gp.player_id
                                      ORDER BY date(s.scheduled_ts) DESC,
                                               s.series_id DESC, g.game_no DESC) AS rn
            FROM game_players gp
            JOIN games g ON g.game_id = gp.game_id
            JOIN series s ON s.series_id = g.series_id
            WHERE date(s.scheduled_ts) < ?{clause}
        ) WHERE rn = 1
        """,
        (as_of_date.isoformat(), *params),
    ).fetchall()


def _resolve_player_levels_by_league(
    conn: sqlite3.Connection,
    as_of_date: date,
    table: RegionRatingTable,
) -> tuple[RegionLevels, dict[str, str], dict[str, bool]]:
    """Each player's latest domestic league priced EXACTLY by the region table.

    Returns:
        ``(levels, region_of, pool_hit)``. ``table.rating`` resolves the league to its
        fitted pool level, else its learned tier level — an unresolved league never
        reads as an average major region. ``region_of`` is the published region facet
        (``REGION_NORMALIZATION_MAP``); ``pool_hit`` says whether the LEVEL came from a
        fitted pool row, which is what the coverage gate measures.
    """
    from league_predictor.predraft.league_tier import LEAGUE_TIER_DEFAULT, REGION_NORMALIZATION_MAP

    level_of: dict[str, float] = {}
    region_of: dict[str, str] = {}
    pool_hit: dict[str, bool] = {}
    for pid, tournament in _latest_domestic_league(conn, as_of_date, exclude_international=True):
        league = tournament or ""
        region_of[pid] = REGION_NORMALIZATION_MAP.get(league, "")
        level_of[pid] = table.rating(league or None)
        pool_hit[pid] = table.has_pool(league or None)
    levels = RegionLevels(level=level_of, default_level=table.tier_effect[LEAGUE_TIER_DEFAULT])
    return levels, region_of, pool_hit


def _stat_context_priors(
    conn: sqlite3.Connection,
    as_of_date: date,
    player_ids: set[str],
    *,
    player_role: Mapping[str, str],
    recipe_version: int,
) -> dict[str, float]:
    """Per-player prior from the walk-forward stat context and the versioned weights."""
    from league_predictor.predraft.ratings.stat_context import (
        aggregate_prior,
        build_game_context_z,
        load_team_sum_weights,
        team_sum_prior,
    )

    if not player_ids:
        return {}
    weights = load_team_sum_weights(recipe_version, expected_build=dict(STAT_CONTEXT_KWARGS))
    ctx = build_game_context_z(conn, as_of_date, **STAT_CONTEXT_KWARGS)
    totals = dict.fromkeys(player_ids, 0.0)
    for block, half_life in PRIOR_BLOCKS.items():
        aggregates = aggregate_prior(ctx, as_of_date, players=player_ids, half_life_days=half_life)
        for pid, prior in aggregates.items():
            totals[pid] += team_sum_prior(
                prior, player_role.get(pid, _DEFAULT_ROLE), weights[block]
            )
    return totals


def _pool_hit_fraction(qualified: Sequence[str], pool_hit: Mapping[str, bool]) -> float:
    """Share of qualified players whose level came from a fitted pool row, not a fallback."""
    if not qualified:
        return 1.0
    return sum(1 for p in qualified if pool_hit.get(p, False)) / len(qualified)


def _assert_region_coverage(fraction: float, as_of_date: date, n_qualified: int) -> None:
    """Refuse a fit whose region term would price tier fallbacks instead of fitted pools.

    Raises:
        RuntimeError: below ``_MIN_REGION_COVERAGE``, naming the cause and
            the operator action, because a collapsed resolution changes what the published
            numbers mean without changing anything a green gate can see.
    """
    if fraction >= _MIN_REGION_COVERAGE:
        return
    raise RuntimeError(
        f"Impact region coverage collapsed at as_of {as_of_date.isoformat()}: only "
        f"{fraction:.1%} of {n_qualified} qualified players price through a fitted region-"
        f"rating pool (floor {_MIN_REGION_COVERAGE:.0%}) — the league_id spellings the facts "
        "carry no longer reach the fitted pools; reconcile league_tier.LEAGUES and "
        "RATING_POOL_ALIAS against SELECT DISTINCT league_id FROM series, then refit Impact"
    )


def _prepare_impact_fit_for_recipe(
    conn: sqlite3.Connection,
    as_of_date: date,
    recipe: ImpactRecipe,
    *,
    corpus: ImpactCorpus | None = None,
) -> ImpactFitContext:
    """Reconstruct evaluator inputs from one validated artifact recipe.

    Raises:
        ValueError: the recipe predates the oldest predictor this module still expresses;
            ``impact_artifact.production_recipe`` rejects such an artifact first.
    """
    if corpus is None:
        corpus = build_impact_corpus(conn)
    games, historical_roles = prepare_impact_window(
        corpus,
        as_of_date=as_of_date,
        window_months=recipe.influence_window_months,
    )
    if recipe.qualification_window_months is None:
        qualification_games, player_role = games, historical_roles
    else:
        qualification_games, player_role = prepare_impact_window(
            corpus,
            as_of_date=as_of_date,
            window_months=recipe.qualification_window_months,
        )
    col_index, colkey_of, n_games, n_distinct = _assign_columns(
        qualification_games,
        player_role,
        recipe.min_games,
    )
    qualified = tuple(p for p, key in colkey_of.items() if key == p)
    column_scale_of: dict[str, float] = dict.fromkeys(colkey_of, 1.0)
    if recipe.qualification_window_months is not None:
        col_index, colkey_of, column_scale_of = _extend_full_history_columns(
            games,
            historical_roles,
            recent_col_index=col_index,
            recent_colkey_of=colkey_of,
            history_buckets_per_role=recipe.history_buckets_per_role,
        )

    from league_predictor.predraft.ratings.region_rating import fit_region_ratings

    table = fit_region_ratings(conn, through=as_of_date)
    region, player_region, pool_hit = _resolve_player_levels_by_league(conn, as_of_date, table)
    exact_fraction = _pool_hit_fraction(qualified, pool_hit)
    _assert_region_coverage(exact_fraction, as_of_date, len(qualified))
    skill_priors = _stat_context_priors(
        conn, as_of_date, set(qualified), player_role=player_role, recipe_version=recipe.version
    )

    for key in (_GAMMA_KEY, _ALPHA_KEY, _BETA_KEY):
        col_index[key] = len(col_index)
    X, y, sample_weight = _build_design(
        games,
        col_index,
        colkey_of,
        skill_priors,
        recipe.half_life_days,
        column_scale_of=column_scale_of,
        region=region,
    )
    penalty_mask = np.ones(len(col_index))
    for key in (_GAMMA_KEY, _ALPHA_KEY, _BETA_KEY):
        penalty_mask[col_index[key]] = 0.0

    days = np.array([g.days_before_asof for g in games])
    qualification_days = np.array([g.days_before_asof for g in qualification_games])
    val_mask = days <= np.quantile(qualification_days, 0.2)
    return ImpactFitContext(
        as_of_date=as_of_date,
        recipe=recipe,
        games=games,
        player_role=player_role,
        col_index=col_index,
        colkey_of=colkey_of,
        column_scale_of=column_scale_of,
        n_games=n_games,
        n_distinct=n_distinct,
        qualified=qualified,
        skill_priors=skill_priors,
        player_region=player_region,
        region=region,
        region_exact_fraction=exact_fraction,
        X=X,
        y=y,
        sample_weight=sample_weight,
        val_mask=val_mask,
        penalty_mask=penalty_mask,
    )


def prepare_impact_fit(
    conn: sqlite3.Connection,
    as_of_date: date,
    *,
    corpus: ImpactCorpus | None = None,
) -> ImpactFitContext:
    """Build the fixed production recipe; only evaluators reconstruct older recipes."""
    return _prepare_impact_fit_for_recipe(conn, as_of_date, IMPACT_RECIPE, corpus=corpus)


def fit_prepared_impact(context: ImpactFitContext, *, lam: float | None = None) -> ImpactFit:
    """Fit one lambda policy over an already-prepared cutoff, per the context's recipe."""
    from league_predictor.predraft.league_tier import PREMIER_REGIONS

    candidates = [lam] if lam is not None else _LAMBDA_CANDIDATES
    best_lam, cv_logloss = _select_lambda(
        context.X,
        context.y,
        context.sample_weight,
        val_mask=context.val_mask,
        candidates=candidates,
        penalty_mask=context.penalty_mask,
    )
    theta = _fit_masked_ridge(
        context.X, context.y, context.sample_weight, lam=best_lam, penalty_mask=context.penalty_mask
    )
    col_index = context.col_index
    gamma = float(theta[col_index[_GAMMA_KEY]])
    alpha = float(theta[col_index[_ALPHA_KEY]])
    beta = float(theta[col_index[_BETA_KEY]])

    qualified = context.qualified
    delta = {p: float(theta[col_index[p]]) for p in qualified}
    # Every player the design placed keeps the value that design row gave them, so
    # game_eta reproduces any fit row exactly.
    player_value = {
        p: gamma * context.skill_priors.get(p, 0.0)
        + context.column_scale_of.get(p, 1.0) * float(theta[col_index[key]])
        for p, key in context.colkey_of.items()
    }
    replacement_value = {
        role: float(theta[col_index[replacement_key(role)]])
        for role in {context.player_role.get(p, _DEFAULT_ROLE) for p in context.n_games}
        if replacement_key(role) in col_index
    }
    values = ImpactValues(
        player_value=player_value,
        replacement_value=replacement_value,
        player_role=context.player_role,
        region=context.region,
        alpha=alpha,
        beta=beta,
    )

    # V_p spreads the team-level region term over five players, so a roster's published
    # values re-sum to the strength the linear predictor gives it.
    region_share = {p: alpha * context.region.share(p) for p in qualified}
    value_logodds = {p: player_value[p] + region_share[p] for p in qualified}

    role_of = {p: context.player_role.get(p, _DEFAULT_ROLE) for p in qualified}
    premier = [p for p in qualified if context.player_region.get(p, "") in PREMIER_REGIONS]
    reference_pool = premier or list(qualified)
    premier_mean = (
        float(np.mean([value_logodds[p] for p in reference_pool])) if reference_pool else 0.0
    )
    by_role: dict[str, list[float]] = defaultdict(list)
    for p in reference_pool:
        by_role[role_of[p]].append(value_logodds[p])
    premier_role_means = {role: float(np.mean(vals)) for role, vals in by_role.items()}
    role_reference = {role: sorted(vals) for role, vals in by_role.items()}

    scores: dict[str, ImpactEstimate] = {}
    for p in qualified:
        role = role_of[p]
        reference = premier_role_means.get(role, premier_mean)
        impact_pp = _value_to_pp(value_logodds[p], reference, premier_mean)
        stats_pp = _value_to_pp(value_logodds[p] - delta[p], reference, premier_mean)
        percentile = _percentile_of(role_reference.get(role, []), value_logodds[p])
        scores[p] = ImpactEstimate(
            player_id=p,
            role=role,
            region=context.player_region.get(p, ""),
            level=context.region.of(p),
            n_games=context.n_games.get(p, 0),
            n_distinct_teammates=context.n_distinct.get(p, 0),
            prior_logodds=context.skill_priors.get(p, 0.0),
            delta=delta[p],
            value_logodds=value_logodds[p],
            display_logodds=value_logodds[p] - reference,
            impact_pp=impact_pp,
            stats_pp=stats_pp,
            results_pp=impact_pp - stats_pp,
            percentile=percentile,
            grade=_grade(percentile),
        )
    return ImpactFit(
        values=values,
        scores=scores,
        gamma=gamma,
        alpha=alpha,
        beta=beta,
        lam=best_lam,
        cv_logloss=cv_logloss,
        premier_mean_logodds=premier_mean,
        premier_role_means=premier_role_means,
        region_exact_fraction=context.region_exact_fraction,
        n_games=len(context.games),
        as_of_date=context.as_of_date.isoformat(),
    )


def _grade(percentile: float) -> str:
    """The shared S/A/B/C/D band for a 0-100 percentile."""
    from league_predictor.predraft.ratings.player_scoring import performance_score_to_grade

    return performance_score_to_grade(percentile)


def fit_player_impact(conn: sqlite3.Connection, as_of_date: date) -> ImpactFit:
    """Fit the current full-history Impact recipe through ``as_of_date``."""
    return fit_prepared_impact(prepare_impact_fit(conn, as_of_date))
