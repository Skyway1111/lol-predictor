"""Column assignment, the Impact design matrix, and its penalized-logistic solve."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse
from scipy.optimize import minimize

from league_predictor.decay import half_life_weight
from league_predictor.metrics import logloss_logits
from league_predictor.predraft.ratings.player_impact.predictor import _DEFAULT_ROLE

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from league_predictor.predraft.ratings.impact_corpus import ImpactGame
    from league_predictor.predraft.ratings.player_impact.predictor import RegionLevels

#: Ridge grid for the current recipe, selected by held-out log-loss (scale-aware, unlike AUC).
_LAMBDA_CANDIDATES = [0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
_GAMMA_KEY = "__prior_scale__"
_ALPHA_KEY = "__region_gap__"
_BETA_KEY = "__blue_side__"


# ── Column assignment ────────────────────────────────────────────────────────


def _assign_columns(
    games: Sequence[ImpactGame],
    player_role: Mapping[str, str],
    min_games: int,
) -> tuple[dict[str, int], dict[str, str], dict[str, int], dict[str, int]]:
    """Map qualified players to own columns, sub-threshold ones to role replacements.

    Returns:
        ``(col_index, colkey_of, n_games, n_distinct_teammates)``; ``colkey_of[player]``
        is the player's own id when qualified, else ``"__rep_{role}__"``, and
        ``col_index`` maps every distinct column key to an integer index.
    """
    n_games: dict[str, int] = {}
    teammates: dict[str, set[str]] = {}
    for g in games:
        for side in (g.team1_players, g.team2_players):
            for p in side:
                n_games[p] = n_games.get(p, 0) + 1
                teammates.setdefault(p, set()).update(q for q in side if q != p)

    colkey_of: dict[str, str] = {}
    for p, played in n_games.items():
        if played >= min_games:
            colkey_of[p] = p
        else:
            role = player_role.get(p, _DEFAULT_ROLE)
            colkey_of[p] = replacement_key(role)

    col_index: dict[str, int] = {}
    for key in colkey_of.values():
        if key not in col_index:
            col_index[key] = len(col_index)

    n_distinct = {p: len(t) for p, t in teammates.items()}
    return col_index, colkey_of, n_games, n_distinct


def replacement_key(role: str) -> str:
    """The shared column key every sub-threshold player of ``role`` maps onto."""
    return f"__rep_{role}__"


def _extend_full_history_columns(
    games: Sequence[ImpactGame],
    historical_roles: Mapping[str, str],
    *,
    recent_col_index: dict[str, int],
    recent_colkey_of: dict[str, str],
    history_buckets_per_role: int,
) -> tuple[dict[str, int], dict[str, str], dict[str, float]]:
    """Add bounded nuisance columns for players absent from the qualification window.

    Returns:
        Recent qualified and replacement mappings untouched; historical-only players
        enter a deterministic signed, role-aware hash basis, so the full corpus cannot
        inflate the named-player solve dimension without bound.
    """
    col_index = dict(recent_col_index)
    colkey_of = dict(recent_colkey_of)
    column_scale_of = dict.fromkeys(recent_colkey_of, 1.0)
    historical_players = sorted(
        {
            player_id
            for game in games
            for side in (game.team1_players, game.team2_players)
            for player_id in side
            if player_id not in recent_colkey_of
        }
    )
    for player_id in historical_players:
        role = historical_roles.get(player_id, _DEFAULT_ROLE)
        # A frozen hash salt: its BYTES pin every bucket assignment, so it is a recipe
        # constant, not a name — changing it is a model change gated by a A/B.
        digest = hashlib.blake2b(
            f"{role}\0{player_id}".encode(),
            digest_size=8,
            person=b"hist-bucket-v1",
        ).digest()
        bucket = int.from_bytes(digest[:4], "little") % history_buckets_per_role
        key = f"__history_{role}_{bucket:02d}__"
        colkey_of[player_id] = key
        column_scale_of[player_id] = 1.0 if digest[4] & 1 else -1.0
        if key not in col_index:
            col_index[key] = len(col_index)
    return col_index, colkey_of, column_scale_of


# ── Design matrices ────────────────────────────────────────────────────────────


def _player_entries(
    game: ImpactGame,
    col_index: Mapping[str, int],
    colkey_of: Mapping[str, str],
    column_scale_of: Mapping[str, float],
) -> list[tuple[int, float]]:
    """Signed ``(column, value)`` pairs for one game's two lineups."""
    entries = [(col_index[colkey_of[p]], column_scale_of.get(p, 1.0)) for p in game.team1_players]
    entries += [(col_index[colkey_of[p]], -column_scale_of.get(p, 1.0)) for p in game.team2_players]
    return entries


def _build_design(
    games: Sequence[ImpactGame],
    col_index: dict[str, int],
    colkey_of: dict[str, str],
    priors: Mapping[str, float],
    half_life_days: float,
    *,
    column_scale_of: dict[str, float],
    region: RegionLevels,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Player columns plus the three unpenalized fit columns gamma / alpha / beta.

    Returns:
        ``(X, y, sample_weight)`` with ``eta = X @ theta`` — no fixed offset, because the
        prior enters through its own fitted scale column instead of being asserted at 1. The
        alpha column is ``region.gap`` of the two lineups, the same call ``game_eta``
        makes, so a design row and its scored eta cannot price region differently.
    """
    n_rows, n_cols = len(games), len(col_index)
    gamma_col, alpha_col, beta_col = (
        col_index[_GAMMA_KEY],
        col_index[_ALPHA_KEY],
        col_index[_BETA_KEY],
    )
    rows, cols, vals = [], [], []
    y = np.zeros(n_rows)
    sample_weight = np.zeros(n_rows)

    for i, g in enumerate(games):
        for column, value in _player_entries(g, col_index, colkey_of, column_scale_of):
            rows.append(i)
            cols.append(column)
            vals.append(value)
        prior_gap = sum(priors.get(p, 0.0) for p in g.team1_players) - sum(
            priors.get(p, 0.0) for p in g.team2_players
        )
        gap = region.gap(g.team1_players, g.team2_players)
        for column, value in ((gamma_col, prior_gap), (alpha_col, gap), (beta_col, g.blue_sign)):
            rows.append(i)
            cols.append(column)
            vals.append(float(value))
        y[i] = g.team1_won
        sample_weight[i] = half_life_weight(g.days_before_asof, half_life_days)

    X = sparse.coo_matrix((vals, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    return X, y, sample_weight  # pyright: ignore[reportReturnType]  # scipy: tocsr() typed csr_array


# ── Penalized logistic fit ─────────────────────────────────────────────────────


def _fit_masked_ridge(
    X: sparse.csr_matrix,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    lam: float,
    penalty_mask: np.ndarray,
) -> np.ndarray:
    """Minimize weighted logistic NLL at ``eta = X @ theta``, L2 on masked columns.

    Returns:
        ``theta`` minimizing ``-Sigma w[y log p + (1-y) log(1-p)] + lam ||mask * theta||^2``;
        convex, L-BFGS-B on the analytic gradient. Columns with ``mask = 0`` (the prior
        scale, the region weight, the blue-side weight) are fit free of shrinkage — a
        penalty on them would silently bias the model's calibration toward zero.
    """
    n_cols = X.shape[1]  # pyright: ignore[reportOptionalSubscript]  # scipy: _shape typed None

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = X @ theta
        # Numerically stable log-loss.
        log_p = -np.logaddexp(0.0, -eta)  # log sigmoid(eta)
        log_1mp = -np.logaddexp(0.0, eta)  # log(1 - sigmoid(eta))
        nll = -np.sum(sample_weight * (y * log_p + (1.0 - y) * log_1mp))
        penalized = penalty_mask * theta
        loss = nll + lam * np.dot(penalized, penalized)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (sample_weight * (p - y)) + 2.0 * lam * penalty_mask * theta
        return loss, np.asarray(grad).ravel()

    res = minimize(
        objective,
        np.zeros(n_cols),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-9},
    )
    if not res.success:
        raise RuntimeError(
            f"Impact ridge solve failed: {res.message} after {res.nit} iterations "
            f"(lambda={lam}, objective={res.fun})"
        )
    return res.x


def _select_lambda(
    X: sparse.csr_matrix,
    y: np.ndarray,
    sample_weight: np.ndarray,
    *,
    val_mask: np.ndarray,
    candidates: list[float],
    penalty_mask: np.ndarray,
) -> tuple[float, float]:
    """Walk-forward CV: fit on ``~val_mask`` rows, score the held-out rows.

    Returns:
        ``(best_lam, best_score)`` — held-out log-loss, lower wins. The median candidate
        when the validation slice is one-class.
    """
    train = ~val_mask
    if len(np.unique(y[val_mask])) < 2:  # noqa: PLR2004 - a one-class slice cannot score
        return float(np.median(candidates)), float("nan")

    best_lam, best_score = candidates[0], float("inf")
    for lam in candidates:
        theta = _fit_masked_ridge(
            X[train], y[train], sample_weight[train], lam=lam, penalty_mask=penalty_mask
        )
        score = logloss_logits(y[val_mask], X[val_mask] @ theta)
        if score < best_score:
            best_lam, best_score = lam, score
    return float(best_lam), best_score
