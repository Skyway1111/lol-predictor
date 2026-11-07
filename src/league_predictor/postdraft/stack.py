"""The fitted post-draft stack: served x draft logit weights plus per-team draft utilization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import minimize

from league_predictor.metrics import logloss_logits
from league_predictor.probability import sigmoid

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

#: Team-term penalties the holdout ranks. The objective is TOTAL log loss plus
#: ``penalty * sum(u**2)``, so a penalty of k pulls like k games of evidence.
PENALTY_GRID: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

#: The time-ordered split, applied twice: the outer holdout gates the team terms, the
#: inner split of the outer train picks the penalty, so the gate never scores a tuned arm.
_TRAIN_FRACTION = 0.75

#: The global row's team id — the two coefficients every game shares.
GLOBAL_TEAM_ID = "*"


@dataclass(frozen=True, slots=True)
class StackRow:
    """One fit row: both parents' blue-oriented logits, the two sides, and blue's outcome."""

    date: str
    l_served: float
    l_draft: float
    blue_team: str
    red_team: str
    blue_won: bool


@dataclass(frozen=True, slots=True)
class Stack:
    """The fitted composition ``a*l_served + (beta + u_blue + u_red)*l_draft``.

    Attributes:
        a: the served logit's weight.
        beta: the draft logit's weight every game shares.
        u: per-team utilization added to ``beta``; a team absent from the fit reads 0.
        n_oos: per-team count of the oos rows the fit saw.
        penalty: the grid value the inner time-ordered split chose.
        holdout_gain: global outer-holdout log loss minus team-aware at that penalty;
            positive means u helped, and at or below zero every ``u`` is exactly 0.
        n_rows: the oos rows the final fit saw.
    """

    a: float
    beta: float
    u: Mapping[str, float]
    n_oos: Mapping[str, int]
    penalty: float
    holdout_gain: float
    n_rows: int

    def logit_final(self, l_served: float, l_draft: float, blue_team: str, red_team: str) -> float:
        """The blue-oriented final logit; a team the fit never saw contributes ``u = 0``."""
        coefficient = self.beta + self.u.get(blue_team, 0.0) + self.u.get(red_team, 0.0)
        return self.a * l_served + coefficient * l_draft

    def p_blue(self, l_served: float, l_draft: float, blue_team: str, red_team: str) -> float:
        """:meth:`logit_final` as a probability; the mirrored game reads exactly ``1 - p``."""
        return sigmoid(self.logit_final(l_served, l_draft, blue_team, red_team))


@dataclass(frozen=True, slots=True)
class _Design:
    """The fit arrays: both logit columns, the outcome, and the two team-index columns."""

    l_served: np.ndarray
    l_draft: np.ndarray
    y: np.ndarray
    blue_ix: np.ndarray
    red_ix: np.ndarray
    teams: list[str]

    def slice(self, start: int, stop: int) -> _Design:
        """The row range ``[start, stop)`` over the SAME team index, so ``u`` stays aligned."""
        cut = slice(start, stop)
        return _Design(
            self.l_served[cut],
            self.l_draft[cut],
            self.y[cut],
            self.blue_ix[cut],
            self.red_ix[cut],
            self.teams,
        )


def _design(rows: Sequence[StackRow]) -> _Design:
    """Date-ordered arrays over one team index built from every row, blue then red."""
    ordered = sorted(rows, key=lambda row: row.date)
    teams = sorted({team for row in ordered for team in (row.blue_team, row.red_team)})
    index = {team: i for i, team in enumerate(teams)}
    return _Design(
        np.array([row.l_served for row in ordered], dtype=float),
        np.array([row.l_draft for row in ordered], dtype=float),
        np.array([float(row.blue_won) for row in ordered], dtype=float),
        np.array([index[row.blue_team] for row in ordered], dtype=int),
        np.array([index[row.red_team] for row in ordered], dtype=int),
        teams,
    )


def _eta(design: _Design, a: float, beta: float, u: np.ndarray) -> np.ndarray:
    """The row logits under one parameter vector."""
    coefficient = beta + (u[design.blue_ix] + u[design.red_ix] if u.size else 0.0)
    return a * design.l_served + coefficient * design.l_draft


def _objective(
    theta: np.ndarray, design: _Design, penalty: float, n_teams: int
) -> tuple[float, np.ndarray]:
    """Total logistic NLL plus ``penalty * sum(u**2)``, with its analytic gradient."""
    a, beta, u = theta[0], theta[1], theta[2:]
    eta = _eta(design, a, beta, u)
    residual = 1.0 / (1.0 + np.exp(-eta)) - design.y
    value = float(np.sum(np.logaddexp(0.0, eta) - design.y * eta) + penalty * np.sum(u**2))
    draft_residual = residual * design.l_draft
    grad_u = np.zeros(n_teams)
    if n_teams:
        grad_u = (
            np.bincount(design.blue_ix, draft_residual, minlength=n_teams)
            + np.bincount(design.red_ix, draft_residual, minlength=n_teams)
            + 2.0 * penalty * u
        )
    gradient = np.concatenate(
        ([float(np.sum(residual * design.l_served))], [float(np.sum(draft_residual))], grad_u)
    )
    return value, gradient


def _fit(design: _Design, *, penalty: float, team_aware: bool) -> tuple[float, float, np.ndarray]:
    """L-BFGS over ``[a, beta, u...]``; ``team_aware=False`` drops the team block entirely.

    Raises:
        RuntimeError: the optimizer did not converge, which on this convex objective
            means a corpus the two coefficients separate; the coefficients it stopped at
            are not a fit and are never returned.
    """
    n_teams = len(design.teams) if team_aware else 0
    start = np.concatenate(([1.0, 1.0], np.zeros(n_teams)))
    args = (design, penalty, n_teams)
    result = minimize(_objective, start, args=args, jac=True, method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(
            f"the post-draft stack did not converge on {len(design.y)} oos rows "
            f"(penalty={penalty}, team terms={team_aware}): {result.message}"
        )
    return float(result.x[0]), float(result.x[1]), result.x[2:]


def fit_global(l_served: np.ndarray, l_draft: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """The two-coefficient stack over one row block: no team terms, no penalty.

    Returns:
        ``(a, beta)`` maximizing the likelihood of ``y`` under
        ``sigmoid(a*l_served + beta*l_draft)`` — the global-only stack.
    """
    unused = np.zeros(len(y), dtype=int)
    design = _Design(
        np.asarray(l_served, dtype=float),
        np.asarray(l_draft, dtype=float),
        np.asarray(y, dtype=float),
        unused,
        unused,
        [],
    )
    a, beta, _ = _fit(design, penalty=0.0, team_aware=False)
    return a, beta


def _holdout_logloss(design: _Design, a: float, beta: float, u: np.ndarray) -> float:
    """Mean log loss of one parameter vector on a held-out design."""
    return logloss_logits(design.y, _eta(design, a, beta, u))


def _split(design: _Design, n: int) -> tuple[_Design, _Design]:
    """The time-ordered 75/25 cut of the first ``n`` rows.

    Raises:
        RuntimeError: too few rows for either side of the cut to hold a game.
    """
    split = int(n * _TRAIN_FRACTION)
    if split < 1 or split >= n:
        raise RuntimeError(
            f"the post-draft stack needs at least 2 rows on each side of its nested 75/25 "
            f"date splits, got {n}; run the served oos walk and the draft oos walk for the "
            "current model pair first."
        )
    return design.slice(0, split), design.slice(split, n)


def fit_stack(rows: Sequence[StackRow]) -> Stack:
    """Fit the stack: pick the penalty inside the train block, gate on the outer holdout, refit.

    Args:
        rows: the oos corpus, one row per game; order is irrelevant (the fit sorts by date).

    Returns:
        The stack. The penalty is the grid argmin on the inner split of the outer train
        block; the team terms survive only when a team-aware fit at that penalty beats
        the two-coefficient global fit on the outer holdout, which no penalty was tuned
        on; otherwise every ``u`` is 0 and ``holdout_gain`` is at or below 0.

    Raises:
        RuntimeError: too few oos rows for the nested time-ordered splits.
    """
    n = len(rows)
    design = _design(rows)
    train, holdout = _split(design, n)
    inner_train, inner_val = _split(train, len(train.y))

    scored = [
        (_holdout_logloss(inner_val, *_fit(inner_train, penalty=p, team_aware=True)), p)
        for p in PENALTY_GRID
    ]
    _inner_ll, penalty = min(scored)

    global_a, global_beta = fit_global(train.l_served, train.l_draft, train.y)
    global_ll = _holdout_logloss(holdout, global_a, global_beta, np.empty(0))
    team_ll = _holdout_logloss(holdout, *_fit(train, penalty=penalty, team_aware=True))
    gain = global_ll - team_ll

    a, beta, u = _fit(design, penalty=penalty, team_aware=gain > 0)
    counts = np.bincount(design.blue_ix, minlength=len(design.teams)) + np.bincount(
        design.red_ix, minlength=len(design.teams)
    )
    return Stack(
        a=a,
        beta=beta,
        u={team: float(u[i]) for i, team in enumerate(design.teams)} if u.size else {},
        n_oos={team: int(counts[i]) for i, team in enumerate(design.teams)},
        penalty=float(penalty),
        holdout_gain=float(gain),
        n_rows=n,
    )
