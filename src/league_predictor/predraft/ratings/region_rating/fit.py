"""The MAP solve: one cutoff's walk-forward region columns and the fitted pool levels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

import numpy as np
from scipy import optimize, sparse

from league_predictor.decay import half_life_rate
from league_predictor.predraft.league_tier import resolve_domestic_league
from league_predictor.predraft.ratings.region_rating.corpus import (
    _TABLE_CACHE,
    _cache_prefix,
    _fit_corpus,
    _home_structure,
)
from league_predictor.predraft.ratings.region_rating.table import RegionRatingTable, _pool_tier
from league_predictor.vocab.roles import N_ROLES

if TYPE_CHECKING:
    import sqlite3

    from league_predictor.predraft.ratings.region_rating.corpus import _FitCorpus

# Fit dials, pinned by the Phase-1 standalone eval (benchmarks.md).
_HALF_LIFE_DAYS = 1460.0
_L2_PLAYER = 32.0
_L2_POOL = 8.0
_L2_TIER = 0.5
_L2_INTERCEPT = 1.0


def _through_iso(through) -> str:
    """Normalize a date/datetime/ISO-string cutoff to ``YYYY-MM-DD``."""
    if isinstance(through, str):
        return through[:10]
    if isinstance(through, datetime):
        return through.date().isoformat()
    return through.isoformat()


def _side_pool(home, team_id: str, year: str, tournament: str) -> str | None:
    """A side's pool: its modal home, else the (domestic) tournament itself."""
    pool = home.get((team_id, year))
    if pool is not None:
        return pool
    return resolve_domestic_league(tournament)


def fit_region_ratings(
    conn: sqlite3.Connection, *, through: date | datetime | str
) -> RegionRatingTable:
    """Fit the region table on games before ``through`` (~5 s cold, memoized per vintage).

    Args:
        through: exclusive ``YYYY-MM-DD`` cutoff (date, datetime, or ISO string).
    Returns:
        The fitted table; betas are 0.0 until the served retrain stamps them.
    """
    through_day = _through_iso(through)
    prefix = _cache_prefix(conn)
    if prefix is not None:
        key = (*prefix, through_day)
        cached = _TABLE_CACHE.get(key)
        if cached is not None:
            return cached
    corpus = _fit_corpus(conn, prefix)
    table, _warm = _fit(
        corpus,
        through_day,
        half_life_days=_HALF_LIFE_DAYS,
        l2_player=_L2_PLAYER,
        l2_pool=_L2_POOL,
        l2_tier=_L2_TIER,
    )
    if prefix is not None:
        _TABLE_CACHE[key] = table
    return table


@dataclass(frozen=True)
class _CutoffCtx:
    """One cutoff's walk-forward pool context, dial-independent and memoized."""

    home: dict  # (team_id, 'YYYY') -> pool, derived from the prefix only
    pools: tuple[str, ...]
    Xr: sparse.csr_matrix  # k x (n_pools + 3 tiers) signed region columns
    row_pools: np.ndarray  # (k, 2) side-pool ordinals, -1 unmapped
    n_app: int  # appearances belonging to prefix rows (app arrays are row-ordered)
    app_pool: np.ndarray  # (n_app,) side-pool ordinal per prefix appearance, -1 unmapped


def _cutoff_ctx(corpus: _FitCorpus, through_day: str, k: int) -> _CutoffCtx:
    """The cutoff's walk-forward home map and region columns, built from rows[:k] only."""
    cached = corpus.cutoff_cache.get(through_day)
    if cached is not None:
        return cached
    prefix = corpus.games[:k]
    home, _ = _home_structure(prefix, tiers=("tier1", "tier2", "tier3"))

    side_pools: list[tuple[str | None, str | None]] = []
    pool_names: set[str] = set()
    for g in prefix:
        yr = g["date"][:4]
        pa = _side_pool(home, g["t1"], yr, g["t"])
        pb = _side_pool(home, g["t2"], yr, g["t"])
        side_pools.append((pa, pb))
        pool_names.update(p for p in (pa, pb) if p is not None)
    pools = tuple(sorted(pool_names))
    pool_ord = {p: i for i, p in enumerate(pools)}
    tier_col = {t: len(pools) + i for i, t in enumerate((1, 2, 3))}

    data: list[float] = []
    ri: list[int] = []
    ci: list[int] = []
    rp = np.full((k, 2), -1, dtype=np.int64)
    for i, (pa, pb) in enumerate(side_pools):
        if pa is not None:
            rp[i, 0] = pool_ord[pa]
        if pb is not None:
            rp[i, 1] = pool_ord[pb]
        if pa is not None and pb is not None and pa != pb:
            for pool, sign in ((pa, 1.0), (pb, -1.0)):
                data.append(sign)
                ri.append(i)
                ci.append(pool_ord[pool])
                data.append(sign)
                ri.append(i)
                ci.append(tier_col[_pool_tier(pool)])

    n_app = int(np.searchsorted(corpus.app_row, k, side="left"))
    app_pool = rp[corpus.app_row[:n_app], corpus.app_side[:n_app]]
    ctx = _CutoffCtx(
        home=home,
        pools=pools,
        Xr=sparse.csr_matrix((data, (ri, ci)), shape=(k, len(pools) + 3)),
        row_pools=rp,
        n_app=n_app,
        app_pool=app_pool,
    )
    corpus.cutoff_cache[through_day] = ctx
    return ctx


def _warm_start(n: int, n_players: int, pools: tuple[str, ...], x0) -> np.ndarray:
    """Map a previous cutoff's optimum onto this cutoff's parameter layout."""
    start = np.zeros(n)
    if x0 is None:
        return start
    prev_theta, prev_pools = x0
    prev_ord = {p: i for i, p in enumerate(prev_pools)}
    start[:n_players] = prev_theta[:n_players]
    for i, pool in enumerate(pools):
        j = prev_ord.get(pool)
        if j is not None:
            start[n_players + i] = prev_theta[n_players + j]
    start[n_players + len(pools) : n_players + len(pools) + 3] = prev_theta[
        n_players + len(prev_pools) : n_players + len(prev_pools) + 3
    ]
    start[-1] = prev_theta[-1]
    return start


def _fit(
    corpus: _FitCorpus,
    through_day: str,
    *,
    half_life_days: float,
    l2_player: float,
    l2_pool: float,
    l2_tier: float,
    x0=None,
):
    """The MAP solve behind :func:`fit_region_ratings` on the row prefix before the cutoff.

    Args:
        x0: optional ``(theta, pools)`` from a nearby cutoff's fit. The objective is
            strictly convex with one optimum, but L-BFGS stops on a tolerance, so a
            warm start can move the result within that tolerance (~1e-3 log-odds);
            sweep callers accept the drift, byte-determinism callers pass None.
    Returns:
        ``(table, (theta, pools))`` — the second element feeds the next warm start.
    """
    asof = date.fromisoformat(through_day)
    k = int(np.searchsorted(corpus.days, asof.toordinal(), side="left"))
    if k == 0:
        zero = {1: 0.0, 2: 0.0, 3: 0.0}
        return RegionRatingTable(r={}, tier_effect=zero, home={}), None

    ctx = _cutoff_ctx(corpus, through_day, k)
    n_players, pools = corpus.n_players, ctx.pools
    n_pools = len(pools)
    n = n_players + n_pools + 3 + 1
    rate = half_life_rate(half_life_days)
    X = sparse.hstack([corpus.Xp[:k], ctx.Xr, corpus.Xi[:k]], format="csr")
    y = corpus.y[:k]
    w = np.exp(-rate * (asof.toordinal() - corpus.days[:k]))
    l2 = np.empty(n)
    l2[:n_players] = l2_player
    l2[n_players : n_players + n_pools] = l2_pool
    l2[n_players + n_pools : n - 1] = l2_tier
    l2[n - 1] = _L2_INTERCEPT

    def objective(theta):
        z = X @ theta
        # log(1+e^z) - y*z, computed stably for both signs of z
        nll = w @ (np.logaddexp(0.0, z) - y * z)
        grad = X.T @ (w * (1.0 / (1.0 + np.exp(-z)) - y)) + l2 * theta
        return nll + 0.5 * float(theta @ (l2 * theta)), grad

    result = optimize.minimize(
        objective,
        _warm_start(n, n_players, pools, x0),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    theta = result.x

    # Population level per pool: fitted tier+pool terms + N_ROLES x the pool's appearance-
    # weighted mean player effect (short scoresheets still average cleanly).
    app_w = w[corpus.app_row[: ctx.n_app]]
    app_eff = theta[corpus.app_player[: ctx.n_app]]
    app_pool = ctx.app_pool
    all_w = float(app_w.sum())
    global_mean = float(app_w @ app_eff) / all_w if all_w else 0.0
    pooled = app_pool >= 0
    eff_sum = np.bincount(app_pool[pooled], weights=(app_w * app_eff)[pooled], minlength=n_pools)
    eff_w = np.bincount(app_pool[pooled], weights=app_w[pooled], minlength=n_pools)

    tier_of = np.array([_pool_tier(p) for p in pools], dtype=np.int64)
    tiers = {t: n_players + n_pools + i for i, t in enumerate((1, 2, 3))}

    def pool_level(o):
        mean_eff = eff_sum[o] / eff_w[o] if eff_w[o] else global_mean
        return (
            float(theta[tiers[int(tier_of[o])]]) + float(theta[n_players + o]) + N_ROLES * mean_eff
        )

    levels = {o: pool_level(o) for o in range(n_pools)}
    tier1_w = sum(eff_w[o] for o in levels if tier_of[o] == 1)
    origin = (
        sum(levels[o] * eff_w[o] for o in levels if tier_of[o] == 1) / tier1_w if tier1_w else 0.0
    )
    # float() everywhere: np.float64 must not leak into the JSON-bound table.
    r = {pools[o]: float(levels[o] - origin) for o in levels}

    tier_effect: dict[int, float] = {}
    for t in (1, 2, 3):
        member_w = sum(eff_w[o] for o in levels if tier_of[o] == t)
        if member_w:
            tier_effect[t] = float(
                sum((levels[o] - origin) * eff_w[o] for o in levels if tier_of[o] == t) / member_w
            )
    # A tier with no fitted pool prices at the weakest fitted level (never the population
    # average -- the Impact miss-pricing trap); an empty table prices everything at 0.
    floor = min(r.values(), default=0.0)
    floor = min(floor, min(tier_effect.values(), default=floor))
    for t in (1, 2, 3):
        tier_effect.setdefault(t, float(floor))
    home_flat = {
        f"{tid}|{yr}": pool for (tid, yr), pool in ctx.home.items() if int(yr) >= asof.year - 4
    }
    return RegionRatingTable(r=r, tier_effect=tier_effect, home=home_flat), (theta, pools)
