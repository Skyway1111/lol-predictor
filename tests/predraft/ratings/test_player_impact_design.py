"""Synthetic pins on the Impact design matrix and its penalized-logistic solve."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from league_predictor.predraft.ratings.impact_corpus import ImpactGame
from league_predictor.predraft.ratings.player_impact import RegionLevels
from league_predictor.predraft.ratings.player_impact.design import (
    _ALPHA_KEY,
    _BETA_KEY,
    _GAMMA_KEY,
    _assign_columns,
    _build_design,
    _fit_masked_ridge,
)
from league_predictor.predraft.ratings.player_impact.results import _value_to_pp


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _make_games(true_skill, n_games, rng, *, fixed_team_a=None):
    """Generate 5v5 games from a player pool using true BT win probabilities.

    Args:
        fixed_team_a: five players always on team1; otherwise both sides are random
            disjoint fives.
    """
    players = list(true_skill)
    games: list[ImpactGame] = []
    for _ in range(n_games):
        if fixed_team_a is not None:
            t1 = list(fixed_team_a)
            pool = [p for p in players if p not in t1]
            t2 = list(rng.choice(pool, size=5, replace=False))
        else:
            picked = rng.choice(players, size=10, replace=False)
            t1, t2 = list(picked[:5]), list(picked[5:])
        margin = sum(true_skill[p] for p in t1) - sum(true_skill[p] for p in t2)
        won = 1 if rng.random() < _sigmoid(margin) else 0
        games.append(
            ImpactGame(
                tuple(t1), tuple(t2), won, float(rng.uniform(0, 700)), int(rng.choice([-1, 1]))
            )
        )
    return games


def _fit_columns(col_index: dict[str, int]) -> dict[str, int]:
    """Append the three unpenalized fit columns the recipe-3 design expects."""
    for key in (_GAMMA_KEY, _ALPHA_KEY, _BETA_KEY):
        col_index[key] = len(col_index)
    return col_index


def _penalty_mask(col_index: dict[str, int]) -> np.ndarray:
    """Ones on player columns, zeros on gamma / alpha / beta."""
    mask = np.ones(len(col_index))
    for key in (_GAMMA_KEY, _ALPHA_KEY, _BETA_KEY):
        mask[col_index[key]] = 0.0
    return mask


def _design(games, col_index, colkey_of, priors, *, half_life=1e9, region=None):
    """Build the current design over a synthetic game list."""
    return _build_design(
        games,
        col_index,
        colkey_of,
        priors,
        half_life,
        column_scale_of=dict.fromkeys(colkey_of, 1.0),
        region=region or RegionLevels(level={}, default_level=0.0),
    )


def test_recovers_known_player_values_synthetic():
    """With players moving across lineups, the fit recovers true relative skills."""
    rng = np.random.default_rng(7)
    n = 14
    true_skill = {f"p{i}": v for i, v in enumerate(np.linspace(-1.2, 1.2, n))}
    roles = dict.fromkeys(true_skill, "mid")
    games = _make_games(true_skill, 5000, rng)

    priors = dict.fromkeys(true_skill, 0.0)
    col_index, colkey_of, _, _ = _assign_columns(games, roles, min_games=0)
    _fit_columns(col_index)
    X, y, sw = _design(games, col_index, colkey_of, priors)
    theta = _fit_masked_ridge(X, y, sw, lam=1.0, penalty_mask=_penalty_mask(col_index))

    recovered = np.array([theta[col_index[p]] for p in true_skill])
    truth = np.array([true_skill[p] for p in true_skill])
    # Bradley-Terry is identified up to an additive constant; center both.
    recovered -= recovered.mean()
    truth -= truth.mean()
    corr = np.corrcoef(recovered, truth)[0, 1]
    assert corr > 0.9, f"recovered skills correlate {corr:.3f} with truth"


def test_replacement_pooling():
    """Sub-threshold players collapse into a per-role replacement column."""
    rng = np.random.default_rng(1)
    # 10 qualified players (appear in every game) + 2 one-off scrubs.
    qualified = {f"q{i}": 0.0 for i in range(10)}
    roles = {p: ("top" if i % 2 == 0 else "mid") for i, p in enumerate(qualified)}
    games = _make_games(qualified, 60, rng)
    # Two scrubs, one game each, distinct roles.
    games.append(
        ImpactGame(("q0", "q2", "q4", "q6", "scrubT"), ("q1", "q3", "q5", "q7", "q8"), 1, 10.0, 1)
    )
    games.append(
        ImpactGame(("q1", "q3", "q5", "q7", "scrubM"), ("q0", "q2", "q4", "q6", "q9"), 0, 12.0, -1)
    )
    roles["scrubT"] = "top"
    roles["scrubM"] = "mid"

    col_index, colkey_of, n_games, _ = _assign_columns(games, roles, min_games=5)

    assert colkey_of["scrubT"] == "__rep_top__"
    assert colkey_of["scrubM"] == "__rep_mid__"
    assert "__rep_top__" in col_index
    assert "__rep_mid__" in col_index
    assert colkey_of["q0"] == "q0"  # qualified keeps its own column
    assert n_games["scrubT"] == 1


def test_thin_churn_player_stays_near_prior_vs_mover():
    """A collinear (no-churn) player's delta is smaller than a mover's, given a wrong prior."""
    rng = np.random.default_rng(3)
    n = 14
    true_skill = {f"p{i}": v for i, v in enumerate(np.linspace(-1.0, 1.0, n))}
    roles = dict.fromkeys(true_skill, "mid")
    priors = dict.fromkeys(true_skill, 0.0)  # deliberately wrong (true skills are non-zero)

    # No-churn world: p0..p4 always together as team1.
    nochurn = _make_games(true_skill, 4000, rng, fixed_team_a=["p0", "p1", "p2", "p3", "p4"])
    ci_n, ck_n, _, _ = _assign_columns(nochurn, roles, min_games=0)
    _fit_columns(ci_n)
    Xn, yn, swn = _design(nochurn, ci_n, ck_n, priors)
    dn = _fit_masked_ridge(Xn, yn, swn, lam=1.0, penalty_mask=_penalty_mask(ci_n))

    # Churn world: everyone shuffled across lineups.
    churn = _make_games(true_skill, 4000, rng)
    ci_c, ck_c, _, _ = _assign_columns(churn, roles, min_games=0)
    _fit_columns(ci_c)
    Xc, yc, swc = _design(churn, ci_c, ck_c, priors)
    dc = _fit_masked_ridge(Xc, yc, swc, lam=1.0, penalty_mask=_penalty_mask(ci_c))

    # p0 (true skill ~ -1.0) is identifiable only when it moves.
    assert abs(dc[ci_c["p0"]]) > abs(dn[ci_n["p0"]])


def test_penalty_shrinks_players_and_leaves_the_prior_scale_free():
    """Huge lambda drives player deltas to ~0 while gamma stays free to price the prior."""
    rng = np.random.default_rng(5)
    n = 12
    true_skill = {f"p{i}": v for i, v in enumerate(np.linspace(-1.0, 1.0, n))}
    roles = dict.fromkeys(true_skill, "mid")
    priors = dict(true_skill)  # the prior IS the truth, so gamma should land near 1
    games = _make_games(true_skill, 3000, rng)

    col_index, colkey_of, _, _ = _assign_columns(games, roles, min_games=0)
    _fit_columns(col_index)
    X, y, sw = _design(games, col_index, colkey_of, priors)
    theta = _fit_masked_ridge(X, y, sw, lam=1e7, penalty_mask=_penalty_mask(col_index))

    players = [theta[col_index[p]] for p in true_skill]
    assert np.max(np.abs(players)) < 1e-3, "player deltas must shrink under a huge penalty"
    assert 0.5 < theta[col_index[_GAMMA_KEY]] < 1.5, "gamma is unpenalized and must stay free"


def test_ridge_refuses_a_nonconverged_solver_result(monkeypatch):
    from types import SimpleNamespace

    from league_predictor.predraft.ratings.player_impact import design as design_module

    monkeypatch.setattr(
        design_module,
        "minimize",
        lambda *_, **__: SimpleNamespace(
            success=False,
            message="iteration limit",
            nit=500,
            fun=123.0,
            x=np.zeros(1),
        ),
    )

    with pytest.raises(RuntimeError, match=r"iteration limit.*500"):
        _fit_masked_ridge(
            design_module.sparse.csr_matrix([[1.0]]),
            np.array([1.0]),
            np.array([1.0]),
            lam=1.0,
            penalty_mask=np.ones(1),
        )


def test_impact_pp_monotonic_in_value():
    """_value_to_pp is increasing in the player's value log-odds."""
    reference, premier_mean = -0.5, 0.0
    values = np.linspace(-2.0, 2.0, 25)
    impact = [_value_to_pp(v, reference, premier_mean) for v in values]
    assert all(b > a for a, b in itertools.pairwise(impact)), "impact_pp must increase with value"


def test_impact_pp_is_zero_at_the_role_reference():
    """Impact reads as lift over the average premier starter of the role: 0 exactly there."""
    reference, premier_mean = -0.4, 0.1
    assert _value_to_pp(reference, reference, premier_mean) == pytest.approx(0.0)
    assert (
        _value_to_pp(0.6, reference, premier_mean) > 0 > _value_to_pp(-0.9, reference, premier_mean)
    )
