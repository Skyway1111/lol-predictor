"""Canonical series math: map win probability <-> series win probability."""

from __future__ import annotations

from league_predictor.probability import logit, sigmoid
from league_predictor.vocab.series import WINS_NEEDED

#: Logit per map of series lead: the team ahead in a series beats its pre-series map odds.
#: Fit on Bo3 and Bo5 series through 2024 (docs/research/served-2026-10.md).
SERIES_MOMENTUM = 0.232
_EPS = 1e-9

# Formats where a series is more than one map and aggregation actually does something -- exactly
# the ``WINS_NEEDED`` keys. Everything else (Bo1, unknown) has series == map.
_MULTI_GAME_FORMATS = tuple(fmt for fmt, wins in WINS_NEEDED.items() if wins > 1)

#: A series state: maps won by team1 and by team2 so far. The pre-series state.
SeriesState = tuple[int, int]
PRE_SERIES: SeriesState = (0, 0)


def _walk(z: float, wins: SeriesState, need: int, momentum: float) -> float:
    """Team1's series probability from ``wins`` when its pre-series map log-odds are ``z``."""
    a, b = wins
    if a >= need:
        return 1.0
    if b >= need:
        return 0.0
    q = sigmoid(z + momentum * (a - b))
    return q * _walk(z, (a + 1, b), need, momentum) + (1 - q) * _walk(z, (a, b + 1), need, momentum)


def simulate_series(
    game_win_prob: float,
    match_format: str,
    *,
    state: SeriesState = PRE_SERIES,
    momentum: float = SERIES_MOMENTUM,
) -> float:
    """Convert the next map's win probability to the series probability.

    Args:
        state: maps won so far ``(team1, team2)``; ``game_win_prob`` is team1's probability
            for the NEXT map at that state, so the walk backs out its pre-series odds first.
        momentum: logit per map of lead; 0 makes every remaining map the same coin.
    """
    p = float(game_win_prob)
    if match_format not in _MULTI_GAME_FORMATS or p <= 0.0 or p >= 1.0:
        return p
    z = logit(p, eps=_EPS) - momentum * (state[0] - state[1])
    return _walk(z, state, WINS_NEEDED[match_format], momentum)


def symmetric_series_prob(
    map_prob: float,
    match_format: str | None,
    *,
    state: SeriesState = PRE_SERIES,
    momentum: float = SERIES_MOMENTUM,
) -> float:
    """Swap-symmetric series win probability from team1's next-map win probability.

    Returns:
        Team1's series probability, satisfying ``f(p, (a, b)) + f(1-p, (b, a)) == 1`` exactly
        (the walk is symmetric, and averaging both views cancels its rounding); for Bo1 and
        unknown formats the map probability unchanged.
    """
    p = float(map_prob)
    if match_format not in _MULTI_GAME_FORMATS:
        return p
    mirrored = (state[1], state[0])
    s_ab = simulate_series(p, match_format, state=state, momentum=momentum)
    s_ba = simulate_series(1.0 - p, match_format, state=mirrored, momentum=momentum)
    return 0.5 * (s_ab + (1.0 - s_ba))
