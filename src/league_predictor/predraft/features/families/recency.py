"""The ``recency`` family -- recent-form diffs a slow rating system lags behind."""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from typing import TYPE_CHECKING, ClassVar

from league_predictor.predraft.ratings.competitive_maps import load_competitive_maps

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence
    from datetime import date

    from league_predictor.asof import AsOf

# Window sizes for the three members. EMA half-life of 10 games -> per-game decay
# alpha = 1 - 2 ** (-1 / 10) (weight halves every 10 games into the past).
_WINRATE_N = 20
_MOMENTUM_N = 10
_EMA_HALFLIFE = 10.0
_EMA_ALPHA = 1.0 - 2.0 ** (-1.0 / _EMA_HALFLIFE)
_NEUTRAL_RATE = 0.5  # "no games" -> neutral form, so the diff is 0.0 not undefined


class _RecencyFamilyIndex:
    """Form index over what a query sees; emits antisymmetric a-minus-b diffs.

    Attributes:
        _keys: per team, the timeline's positions in the facts order. The games a query sees
            are a prefix (every series that started strictly before the query's) followed by
            one run (the query's own earlier maps), addressed as one visible sequence.
    """

    def __init__(self, timelines: dict[str, list[tuple[AsOf, int]]]) -> None:
        # Per team: bisect position list, per-game outcomes, win prefix-sum, EMA-after-each-
        # prefix — one O(history) pass at build so every query is O(log n + own maps).
        self._keys: dict[str, list[tuple[str, str, int]]] = {}
        self._wins: dict[str, list[int]] = {}
        self._winsum: dict[str, list[int]] = {}
        self._ema: dict[str, list[float]] = {}
        for team, tl in timelines.items():
            self._keys[team] = [q.position for q, _ in tl]
            self._wins[team] = [w for _, w in tl]
            winsum = [0]
            ema_arr = [_NEUTRAL_RATE]
            ema = _NEUTRAL_RATE
            for _, w in tl:
                winsum.append(winsum[-1] + w)
                ema += _EMA_ALPHA * (w - ema)
                ema_arr.append(ema)
            self._winsum[team] = winsum
            self._ema[team] = ema_arr

    def _visible(self, team: str, as_of: AsOf) -> tuple[int, int, int]:
        """``(lo, run_lo, run_hi)``: the prefix ``[0, lo)`` and the own-series run."""
        keys = self._keys[team]
        lo = bisect_left(keys, (as_of.ts, "", 0))
        if not as_of.series_id:
            return lo, lo, lo
        run_lo = bisect_left(keys, (as_of.ts, as_of.series_id, 0))
        run_hi = bisect_left(keys, as_of.position)
        return lo, run_lo, run_hi

    def _rate(self, team: str, *, lo: int, run_lo: int, start: int, end: int) -> float:
        """Win rate over visible indices ``[start, end)``, or the neutral rate when empty."""
        if end <= start:
            return _NEUTRAL_RATE
        s = self._winsum[team]
        wins = 0
        if start < lo:
            wins += s[min(end, lo)] - s[start]
        if end > lo:
            wins += s[run_lo + end - lo] - s[run_lo + max(start, lo) - lo]
        return wins / (end - start)

    def _team_form(self, team: str, as_of: AsOf) -> tuple[float, float, float]:
        """``(last-20 winrate, EMA winrate, last-10 minus prior-10 momentum)`` at the query."""
        if team not in self._keys:
            return _NEUTRAL_RATE, _NEUTRAL_RATE, 0.0
        lo, run_lo, run_hi = self._visible(team, as_of)
        n = lo + run_hi - run_lo

        winrate20 = self._rate(team, lo=lo, run_lo=run_lo, start=max(n - _WINRATE_N, 0), end=n)
        ema = self._ema[team][lo]
        for w in self._wins[team][run_lo:run_hi]:
            ema += _EMA_ALPHA * (w - ema)
        recent10 = self._rate(team, lo=lo, run_lo=run_lo, start=max(n - _MOMENTUM_N, 0), end=n)
        prior10 = self._rate(
            team,
            lo=lo,
            run_lo=run_lo,
            start=max(n - 2 * _MOMENTUM_N, 0),
            end=max(n - _MOMENTUM_N, 0),
        )
        return winrate20, ema, recent10 - prior10

    def features(self, team_a: str, team_b: str, as_of: AsOf) -> dict[str, float | str]:
        wr_a, ema_a, mom_a = self._team_form(team_a, as_of)
        wr_b, ema_b, mom_b = self._team_form(team_b, as_of)
        return {
            "recency_winrate20_diff": wr_a - wr_b,
            "recency_ema_diff": ema_a - ema_b,
            "recency_momentum_diff": mom_a - mom_b,
        }


class RecencyFamily:
    """Recent-form family: three antisymmetric form-diff members over win timelines."""

    name = "recency"
    feature_names: ClassVar[list] = [
        "recency_winrate20_diff",
        "recency_ema_diff",
        "recency_momentum_diff",
    ]
    categorical_names: ClassVar[list[str]] = []

    def build(
        self, conn: sqlite3.Connection, *, through: date, query_dates: Sequence[date]
    ) -> _RecencyFamilyIndex:
        """Replay rating-eligible maps before ``through`` into per-team win timelines; no writes."""
        del query_dates
        timelines: dict[str, list[tuple[AsOf, int]]] = defaultdict(list)
        for q, team_a, team_b, a_won in load_competitive_maps(conn, through=through):
            timelines[team_a].append((q, a_won))
            timelines[team_b].append((q, 1 - a_won))
        return _RecencyFamilyIndex(dict(timelines))
