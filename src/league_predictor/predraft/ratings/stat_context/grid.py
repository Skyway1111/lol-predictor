"""The walk-forward grid: per-period expectations, residual sigmas, and the z they produce."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from league_predictor.predraft.ratings.player_scoring import (
    PILLAR_STATS_BY_ROLE,
    STAT_WEIGHTS_BY_ROLE,
)
from league_predictor.predraft.ratings.stat_context.appearances import (
    _Appearances,
    _load_appearances,
)
from league_predictor.predraft.ratings.stat_context.vocab import (
    _ADJUSTED_STATS,
    _EPS,
    _LANE_BT_STATS,
    _MIN_CELL_SUPPORT,
    _MIN_SIGMA,
    _PILLAR_INDEX,
    _ROLE_INDEX,
    _STAT_INDEX,
    _Z_WINSOR,
    GRID_EPOCH,
    GRID_STEP_DAYS,
    GRID_WINDOW_DAYS,
    PILLARS,
    PRIOR_HALF_LIFE_DAYS,
    PRIOR_SHRINKAGE_K,
    STATS,
    ContextZ,
    _assert_scaled_weight_span,
)
from league_predictor.vocab.roles import ROLE_ORDER

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping, Sequence
    from datetime import date


def _cell_means(codes: np.ndarray, values: np.ndarray, present: np.ndarray, size: int):
    """(mean per code, support per code); mean is NaN below the support floor."""
    counts = np.bincount(codes[present], minlength=size)
    sums = np.bincount(codes[present], weights=values[present], minlength=size)
    with np.errstate(invalid="ignore"):
        means = np.where(counts >= _MIN_CELL_SUPPORT, sums / np.maximum(counts, 1), np.nan)
    return means, counts


def _group_offsets(codes: np.ndarray, residual: np.ndarray, present: np.ndarray, size: int):
    """Group-mean offsets with the support floor; 0.0 for thin or absent groups."""
    means, _ = _cell_means(codes, residual, present, size)
    return np.where(np.isnan(means), 0.0, means)


def _expected_for(
    app: _Appearances,
    idx: np.ndarray,
    *,
    mu1: np.ndarray,
    mu2: np.ndarray,
    b_role: np.ndarray,
    win_rate: np.ndarray,
) -> np.ndarray:
    """E[stat] for rows ``idx`` from the window's fitted means (NaN when both levels thin)."""
    roles = app.roles[idx].astype(np.int64)
    tiers = app.tiers[idx].astype(np.int64) - 1
    wins = app.wins[idx].astype(np.int64)
    cell = roles * 3 + tiers
    mu = mu1[cell]
    mu = np.where(np.isnan(mu), mu2[roles], mu)
    rate = win_rate[cell]
    rate = np.where(np.isnan(rate), 0.5, rate)
    return mu + b_role[roles] * (wins - rate)


def _window_fit(  # noqa: PLR0915 - linear per-stat fit assembly, no branching depth
    app: _Appearances, window: np.ndarray, stat: int
):
    """Fit one stat's window parameters: means, outcome gap, adjustments, residual sigma."""
    n_roles = len(ROLE_ORDER)
    vals = app.values[window, stat]
    present = ~np.isnan(vals)
    roles = app.roles[window].astype(np.int64)
    tiers = app.tiers[window].astype(np.int64) - 1
    wins = app.wins[window].astype(np.int64)

    b_role = np.zeros(n_roles)
    win_rate = np.full(n_roles * 3, np.nan)
    mu1, _ = _cell_means(roles * 3 + tiers, vals, present, n_roles * 3)
    mu2, _ = _cell_means(roles, vals, present, n_roles)
    win_means, win_counts = _cell_means(roles * 2 + wins, vals, present, n_roles * 2)
    gap = win_means[1::2] - win_means[0::2]
    supported = np.minimum(win_counts[1::2], win_counts[0::2]) >= _MIN_CELL_SUPPORT
    b_role = np.where(supported & ~np.isnan(gap), np.nan_to_num(gap), 0.0)
    rate_sum = np.bincount(roles * 3 + tiers, weights=wins * present, minlength=n_roles * 3)
    rate_n = np.bincount(roles * 3 + tiers, weights=present.astype(float), minlength=n_roles * 3)
    with np.errstate(invalid="ignore"):
        win_rate = np.where(rate_n > 0, rate_sum / np.maximum(rate_n, 1), np.nan)

    expected = _expected_for(app, window, mu1=mu1, mu2=mu2, b_role=b_role, win_rate=win_rate)
    residual = vals - expected
    fitted = present & ~np.isnan(residual)

    alpha = np.zeros(n_roles)
    strength_mean = 0.0
    side_off = np.zeros(n_roles * 2)
    blind_off = np.zeros(n_roles * 2)
    if STATS[stat] in _ADJUSTED_STATS:
        strength = app.opp_strength[window]
        known = fitted & ~np.isnan(strength)
        if int(known.sum()) >= _MIN_CELL_SUPPORT:
            strength_mean = float(strength[known].mean())
            centered = np.where(known, strength - strength_mean, 0.0)
            var = np.bincount(roles, weights=centered**2 * known, minlength=n_roles)
            cov = np.bincount(
                roles, weights=centered * np.nan_to_num(residual) * known, minlength=n_roles
            )
            counts = np.bincount(roles[known], minlength=n_roles)
            with np.errstate(invalid="ignore"):
                alpha = np.where(
                    (counts >= _MIN_CELL_SUPPORT) & (var > _EPS), cov / np.maximum(var, _EPS), 0.0
                )
            residual = residual - np.where(known, alpha[roles] * centered, 0.0)

        sides = app.sides[window].astype(np.int64)
        side_known = fitted & (sides >= 0)
        side_off = _group_offsets(
            np.where(side_known, roles * 2 + sides, 0), residual, side_known, n_roles * 2
        )
        residual = residual - np.where(side_known, side_off[roles * 2 + np.maximum(sides, 0)], 0.0)

        blinds = app.blinds[window].astype(np.int64)
        blind_known = fitted & (blinds >= 0)
        blind_off = _group_offsets(
            np.where(blind_known, roles * 2 + blinds, 0), residual, blind_known, n_roles * 2
        )
        residual = residual - np.where(
            blind_known, blind_off[roles * 2 + np.maximum(blinds, 0)], 0.0
        )

    sigma1 = np.full(n_roles * 3, np.nan)
    sigma2 = np.full(n_roles, np.nan)
    sq = np.nan_to_num(residual) ** 2
    for codes, size, out in ((roles * 3 + tiers, n_roles * 3, sigma1), (roles, n_roles, sigma2)):
        counts = np.bincount(codes[fitted], minlength=size)
        sums = np.bincount(codes[fitted], weights=sq[fitted], minlength=size)
        with np.errstate(invalid="ignore"):
            variance = np.where(counts >= _MIN_CELL_SUPPORT, sums / np.maximum(counts, 1), np.nan)
        out[:] = np.sqrt(variance)
    sigma1[sigma1 < _MIN_SIGMA] = np.nan
    sigma2[sigma2 < _MIN_SIGMA] = np.nan
    return (
        mu1,
        mu2,
        b_role,
        win_rate,
        alpha,
        strength_mean,
        side_off,
        blind_off,
        sigma1,
        sigma2,
    )


def _apply_fit(app: _Appearances, idx: np.ndarray, stat: int, *, fit) -> np.ndarray:
    """z for period rows ``idx`` under one window fit; NaN where the value or cell is absent."""
    (
        mu1,
        mu2,
        b_role,
        win_rate,
        alpha,
        strength_mean,
        side_off,
        blind_off,
        sigma1,
        sigma2,
    ) = fit
    vals = app.values[idx, stat]
    expected = _expected_for(app, idx, mu1=mu1, mu2=mu2, b_role=b_role, win_rate=win_rate)
    roles = app.roles[idx].astype(np.int64)
    residual = vals - expected
    if STATS[stat] in _ADJUSTED_STATS:
        strength = app.opp_strength[idx]
        known = ~np.isnan(strength)
        residual = residual - np.where(known, alpha[roles] * (strength - strength_mean), 0.0)
        sides = app.sides[idx].astype(np.int64)
        residual = residual - np.where(sides >= 0, side_off[roles * 2 + np.maximum(sides, 0)], 0.0)
        blinds = app.blinds[idx].astype(np.int64)
        residual = residual - np.where(
            blinds >= 0, blind_off[roles * 2 + np.maximum(blinds, 0)], 0.0
        )
    sigma = sigma1[roles * 3 + app.tiers[idx].astype(np.int64) - 1]
    sigma = np.where(np.isnan(sigma), sigma2[roles], sigma)
    with np.errstate(invalid="ignore"):
        return np.clip(residual / sigma, -_Z_WINSOR, _Z_WINSOR)


def _present_weighted_mean(
    stat_z: np.ndarray, rows: np.ndarray, stats: Sequence[str], weights: Mapping[str, float]
) -> np.ndarray:
    """Weighted mean of the present (non-NaN) stat columns; NaN where no stat carries weight."""
    cols = np.array([_STAT_INDEX[s] for s in stats])
    w = np.array([weights.get(s, 1.0) for s in stats])
    z = stat_z[np.ix_(rows, cols)]
    weight_sum = ((~np.isnan(z)) * w).sum(axis=1)
    with np.errstate(invalid="ignore"):
        score = np.nansum(z * w, axis=1) / weight_sum
    return np.where(weight_sum > _EPS, score, np.nan)


def _pillar_scores(roles: np.ndarray, stat_z: np.ndarray) -> np.ndarray:
    """Within-pillar weighted means of present stat z, per each appearance's own role."""
    pillar_z = np.full((len(roles), len(PILLARS)), np.nan)
    for role, role_idx in _ROLE_INDEX.items():
        rows = np.flatnonzero(roles == role_idx)
        if rows.size == 0:
            continue
        for pillar, stats in PILLAR_STATS_BY_ROLE[role].items():
            weights = STAT_WEIGHTS_BY_ROLE[role].get(pillar, {})
            pillar_z[rows, _PILLAR_INDEX[pillar]] = _present_weighted_mean(
                stat_z, rows, stats, weights
            )
    return pillar_z


def _lane_opponents(app: _Appearances) -> np.ndarray:
    """Each appearance's actual lane opponent (same game, same role, other side); -1 if none."""
    opponents = np.full(len(app.players), -1, dtype=np.int64)
    pair: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(len(app.players)):
        if app.team1[i] < 0 or app.roles[i] < 0:
            continue
        pair.setdefault((int(app.games[i]), int(app.roles[i])), []).append((i, int(app.team1[i])))
    for members in pair.values():
        if len(members) == 2 and members[0][1] != members[1][1]:  # noqa: PLR2004 - a lane pair
            opponents[members[0][0]] = members[1][0]
            opponents[members[1][0]] = members[0][0]
    return opponents


def _lane_bt_pillars(app: _Appearances, stat_z: np.ndarray, pillar_z: np.ndarray) -> None:
    """Replace the laning-family pillars with opponent-adjusted lane margins, in date order.

    Returns:
        None; each entry becomes its diff-only margin z PLUS the lane opponent's pre-game
        aggregate (half-life 60d, shrinkage k=20), so the standard aggregation over these
        entries IS an online ridge lane-BT rating — beating a strong laner counts more.
    """
    n = len(app.players)
    margins = np.full(n, np.nan)
    pillar_col = np.full(n, -1, dtype=np.int64)
    for role, role_idx in _ROLE_INDEX.items():
        pillar = "lane" if role == "sup" else "laning"
        rows = np.flatnonzero(app.roles == role_idx)
        weights = STAT_WEIGHTS_BY_ROLE[role].get(pillar, {})
        margins[rows] = _present_weighted_mean(stat_z, rows, _LANE_BT_STATS[pillar], weights)
        pillar_col[rows] = _PILLAR_INDEX[pillar]

    opponents = _lane_opponents(app)
    order = np.argsort(app.dates, kind="stable")
    if not order.size:
        return
    n_players = len(app.player_ids)
    num = np.zeros(n_players)
    den = np.zeros(n_players)
    cnt = np.zeros(n_players)
    t0 = int(app.dates[order[0]])
    _assert_scaled_weight_span(app.dates, t0, PRIOR_HALF_LIFE_DAYS)
    boundaries = np.flatnonzero(np.diff(app.dates[order])) + 1
    for group in np.split(np.arange(len(order)), boundaries):
        rows = order[group]
        with np.errstate(invalid="ignore"):
            rating = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0) * (
                cnt / (cnt + PRIOR_SHRINKAGE_K)
            )
        opp = opponents[rows]
        adjusted = margins[rows] + np.where(opp >= 0, rating[app.players[opp]], 0.0)
        has_col = pillar_col[rows] >= 0
        pillar_z[rows[has_col], pillar_col[rows[has_col]]] = adjusted[has_col]
        # The as-of scale factor cancels in num/den, so no decay pass is ever needed.
        scale = 2.0 ** ((int(app.dates[rows[0]]) - t0) / PRIOR_HALF_LIFE_DAYS)
        valued = ~np.isnan(adjusted)
        players = app.players[rows[valued]]
        np.add.at(num, players, adjusted[valued] * scale)
        np.add.at(den, players, scale)
        np.add.at(cnt, players, 1.0)


def build_game_context_z(
    conn: sqlite3.Connection,
    through: date,
    *,
    lane_bt: bool = False,
) -> ContextZ:
    """Context-adjusted per-appearance z for every game strictly before ``through``.

    Args:
        lane_bt: replace the laning-family pillars with opponent-adjusted lane margins,
            whose aggregate is an online ridge lane-BT rating (recipe v8).

    Returns:
        Appearances scored against expectations whose PARAMETERS are estimated only from
        the trailing window ending at each one's grid-period start (walk-forward at era
        granularity ``GRID_STEP_DAYS``). The game's own outcome/side/blind/champion/
        opponent still enter its own expectation as conditioning variables, so a z is
        outcome-conditioned by design — see the ``ContextZ.stat_z`` contract. Full-corpus
        build cost: ~20 s and ~1.3 GB peak on the reference machine.
    """
    app = _load_appearances(conn, through)
    n = len(app.players)
    stat_z = np.full((n, len(STATS)), np.nan)
    epoch = GRID_EPOCH.toordinal()
    periods = (app.dates - epoch) // GRID_STEP_DAYS

    for period in np.unique(periods):
        grid_start = epoch + int(period) * GRID_STEP_DAYS
        window_lo = int(np.searchsorted(app.dates, grid_start - GRID_WINDOW_DAYS, side="left"))
        window_hi = int(np.searchsorted(app.dates, grid_start, side="left"))
        if window_hi - window_lo < _MIN_CELL_SUPPORT:
            continue
        window = np.arange(window_lo, window_hi)
        rows = np.flatnonzero(periods == period)
        for stat in range(len(STATS)):
            fit = _window_fit(app, window, stat)
            stat_z[rows, stat] = _apply_fit(app, rows, stat, fit=fit)

    pillar_z = _pillar_scores(app.roles, stat_z)
    if lane_bt:
        _lane_bt_pillars(app, stat_z, pillar_z)
    return ContextZ(
        player_ids=app.player_ids,
        appearance_players=app.players,
        appearance_dates=app.dates,
        appearance_roles=app.roles,
        stat_z=stat_z,
        pillar_z=pillar_z,
        through=through,
        appearance_games=app.games,
        appearance_team1=app.team1,
        game_dates=app.game_dates,
        game_team1_won=app.game_team1_won,
        lane_bt=lane_bt,
    )
