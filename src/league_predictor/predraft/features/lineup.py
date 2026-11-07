"""The served ``lineup`` family: expected-fielded-lineup team strength and continuity."""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import date, timedelta
from typing import TYPE_CHECKING, ClassVar

from league_predictor.predraft.cleaning import eligible_games_by_day
from league_predictor.probability import ELO_ORIGIN

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence

    from league_predictor.asof import AsOf

# --- pre-registered knobs (fixed 2026-06-10, before the family's first evaluation) ---
_ELO_K = 20.0  # internal Elo: standard K/init/scale, day-batched
_ELO_INIT = ELO_ORIGIN
_ELO_SCALE = 400.0
_CLEAN_SIZE = 5  # exactly-5 distinct players == a clean lineup observation
_ROOKIE_GAMES = 10  # career games below which a fielded player counts as a rookie
_CHURN_SHORT_DAYS = 30  # churn reference horizons: the lineup fielded ~30/~90 days ago
_CHURN_LONG_DAYS = 90
_DAYS_CAP = 365.0  # day counts are capped here, then log1p-compressed

# 9 antisymmetric diffs (negate under swap) + 2 symmetric context values (unchanged).
_DIFF_NAMES = [
    "lineup_mean_elo_diff",  # mean accrued Elo of the expected 5, a - b
    "lineup_max_elo_diff",  # best player, a - b
    "lineup_min_elo_diff",  # weakest link, a - b
    "lineup_vs_org_diff",  # (lineup mean - org team Elo), a - b: academy-rotation delta
    "lineup_stack_games_diff",  # log1p(games this exact 5 has played together), a - b
    "lineup_days_same_diff",  # log1p(days since this exact 5 last changed), a - b
    "lineup_churn30_diff",  # expected-5 players NOT in the lineup ~30d ago, a - b
    "lineup_churn90_diff",  # same vs ~90d ago, a - b
    "lineup_rookie_count_diff",  # expected-5 players with <10 career games, a - b
]
_SYMMETRIC_NAMES = [
    "lineup_gap_days_max",  # log1p(capped max days since either team last played)
    "lineup_churn30_max",  # max of the two 30d churn counts (season-boundary flag)
]
_FEATURE_NAMES = _DIFF_NAMES + _SYMMETRIC_NAMES

# Fielded lineups per (game, team); the ``rosters`` table is an append-only snapshot with no
# history and must never feed this. Date-only ``through`` bound: lexicographic-prefix compare.
_PLAYERS_SQL = """
    SELECT gp.game_id, gp.team_id, gp.player_id
    FROM game_players gp
    JOIN games g ON g.game_id = gp.game_id
    JOIN series s ON s.series_id = g.series_id
    WHERE s.scheduled_ts < ?
"""


class _History:
    """One entity's day-close value timeline; ``asof`` reads the last day strictly before."""

    __slots__ = ("dates", "values")

    def __init__(self) -> None:
        self.dates: list[date] = []
        self.values: list[float] = []

    def append(self, d: date, v: float) -> None:
        self.dates.append(d)
        self.values.append(v)

    def asof(self, as_of: date, default: float) -> float:
        i = bisect_left(self.dates, as_of)
        return self.values[i - 1] if i else default


class _TeamTimeline:
    """One team's game dates plus the clean-lineup timeline and each entry's run-start index."""

    __slots__ = ("clean_dates", "clean_sets", "game_dates", "run_start")

    def __init__(self) -> None:
        self.game_dates: list[date] = []
        self.clean_dates: list[date] = []
        self.clean_sets: list[frozenset] = []
        self.run_start: list[int] = []

    def add_game(self, d: date, players: frozenset) -> None:
        self.game_dates.append(d)
        if len(players) == _CLEAN_SIZE:
            same_run = bool(self.clean_sets) and self.clean_sets[-1] == players
            self.run_start.append(self.run_start[-1] if same_run else len(self.clean_sets))
            self.clean_dates.append(d)
            self.clean_sets.append(players)


def _log_days(days: float) -> float:
    return math.log1p(min(days, _DAYS_CAP))


class _LineupIndex:
    """As-of reader over the replayed timelines; every member derives from per-team blocks."""

    def __init__(self, *, team_elo, player_elo, player_apps, timelines, stack_dates) -> None:
        self._team_elo = team_elo  # team -> _History
        self._player_elo = player_elo  # player -> _History
        self._player_apps = player_apps  # player -> sorted [date] (one per appearance)
        self._timelines = timelines  # team -> _TeamTimeline
        self._stack_dates = stack_dates  # (team, frozenset) -> sorted [date]

    def days_since_last_game(self, team: str, as_of: date) -> float:
        """Uncapped days from ``team``'s last game strictly before ``as_of``; ``inf`` if none."""
        tl = self._timelines.get(team)
        if tl is None or not (i := bisect_left(tl.game_dates, as_of)):
            return math.inf
        return float((as_of - tl.game_dates[i - 1]).days)

    def _team_block(self, team: str, as_of: date) -> dict[str, float]:
        """The per-team scalars every emitted member diffs or maxes.

        Returns:
            Neutral defaults where data is missing: no clean lineup yet -> aggregates collapse
            to the org Elo (vs_org/stack/days/churn/rookies all 0); no game at all -> gap 0.
        """
        hist = self._team_elo.get(team)
        org_elo = hist.asof(as_of, _ELO_INIT) if hist else _ELO_INIT
        tl = self._timelines.get(team)

        gap = 0.0
        if tl and (i := bisect_left(tl.game_dates, as_of)):
            gap = _log_days((as_of - tl.game_dates[i - 1]).days)

        j = bisect_left(tl.clean_dates, as_of) - 1 if tl else -1
        if j < 0:
            return {
                "mean": org_elo,
                "max": org_elo,
                "min": org_elo,
                "vs_org": 0.0,
                "stack": 0.0,
                "days_same": 0.0,
                "churn30": 0.0,
                "churn90": 0.0,
                "rookies": 0.0,
                "gap": gap,
            }

        lineup = tl.clean_sets[j]
        elos = [
            self._player_elo[p].asof(as_of, _ELO_INIT) if p in self._player_elo else _ELO_INIT
            for p in lineup
        ]
        mean = sum(elos) / len(elos)
        stack = bisect_left(self._stack_dates[(team, lineup)], as_of)
        days_same = (as_of - tl.clean_dates[tl.run_start[j]]).days
        rookies = sum(
            1 for p in lineup if bisect_left(self._player_apps.get(p, []), as_of) < _ROOKIE_GAMES
        )

        def churn(horizon_days: int) -> float:
            k = bisect_right(tl.clean_dates, as_of - timedelta(days=horizon_days)) - 1
            return float(len(lineup - tl.clean_sets[k])) if k >= 0 else 0.0

        return {
            "mean": mean,
            "max": max(elos),
            "min": min(elos),
            "vs_org": mean - org_elo,
            "stack": math.log1p(stack),
            "days_same": _log_days(days_same),
            "churn30": churn(_CHURN_SHORT_DAYS),
            "churn90": churn(_CHURN_LONG_DAYS),
            "rookies": float(rookies),
            "gap": gap,
        }

    def features(self, team_a: str, team_b: str, as_of: AsOf) -> dict[str, float | str]:
        day = as_of.day  # day-strict by construction: the replay closes at day end
        a, b = self._team_block(team_a, day), self._team_block(team_b, day)
        return {
            "lineup_mean_elo_diff": a["mean"] - b["mean"],
            "lineup_max_elo_diff": a["max"] - b["max"],
            "lineup_min_elo_diff": a["min"] - b["min"],
            "lineup_vs_org_diff": a["vs_org"] - b["vs_org"],
            "lineup_stack_games_diff": a["stack"] - b["stack"],
            "lineup_days_same_diff": a["days_same"] - b["days_same"],
            "lineup_churn30_diff": a["churn30"] - b["churn30"],
            "lineup_churn90_diff": a["churn90"] - b["churn90"],
            "lineup_rookie_count_diff": a["rookies"] - b["rookies"],
            "lineup_gap_days_max": max(a["gap"], b["gap"]),
            "lineup_churn30_max": max(a["churn30"], b["churn30"]),
        }


class LineupFamily:
    """Expected-fielded-lineup strength/continuity -- 9 diff + 2 symmetric members."""

    name = "lineup"
    feature_names: ClassVar[list] = list(_FEATURE_NAMES)
    categorical_names: ClassVar[list[str]] = []

    def build(
        self, conn: sqlite3.Connection, *, through: date, query_dates: Sequence[date]
    ) -> _LineupIndex:
        """One day-batched walk-forward replay of all rating-eligible games < ``through``.

        Returns:
            An index whose Elo math uses START-of-day ratings, applying team and player deltas
            only at day close; queries bisect on strict ``date < as_of``, so same-day results
            and lineups are never visible to that day's own matchups.
        """
        del query_dates
        players_by_key: dict[tuple[str, str], set[str]] = defaultdict(set)
        for mid, tid, pid in conn.execute(_PLAYERS_SQL, (through.isoformat(),)).fetchall():
            players_by_key[(str(mid), str(tid))].add(str(pid))

        by_day = eligible_games_by_day(conn, through=through)

        elo: dict[str, float] = {}
        p_elo: dict[str, float] = {}
        team_elo: dict[str, _History] = {}
        player_elo: dict[str, _History] = {}
        player_apps: dict[str, list[date]] = defaultdict(list)
        timelines: dict[str, _TeamTimeline] = {}
        stack_dates: dict[tuple[str, frozenset], list[date]] = defaultdict(list)

        for day in sorted(by_day):
            d_team: dict[str, float] = defaultdict(float)
            d_player: dict[str, float] = defaultdict(float)
            for m in by_day[day]:
                mid, winner = str(m["match_id"]), str(m["winner"])
                t1, t2 = str(m["team1_id"]), str(m["team2_id"])
                r1, r2 = elo.get(t1, _ELO_INIT), elo.get(t2, _ELO_INIT)
                exp1 = 1.0 / (1.0 + 10.0 ** ((r2 - r1) / _ELO_SCALE))
                delta = _ELO_K * ((1.0 if winner == t1 else 0.0) - exp1)
                for tid, signed in ((t1, delta), (t2, -delta)):
                    d_team[tid] += signed
                    fielded = frozenset(players_by_key.get((mid, tid), ()))
                    timelines.setdefault(tid, _TeamTimeline()).add_game(day, fielded)
                    if len(fielded) == _CLEAN_SIZE:
                        stack_dates[(tid, fielded)].append(day)
                    for p in fielded:
                        d_player[p] += signed
                        player_apps[p].append(day)
            for tid, dv in d_team.items():
                elo[tid] = elo.get(tid, _ELO_INIT) + dv
                team_elo.setdefault(tid, _History()).append(day, elo[tid])
            for p, dv in d_player.items():
                p_elo[p] = p_elo.get(p, _ELO_INIT) + dv
                player_elo.setdefault(p, _History()).append(day, p_elo[p])

        return _LineupIndex(
            team_elo=team_elo,
            player_elo=player_elo,
            player_apps=dict(player_apps),
            timelines=timelines,
            stack_dates=dict(stack_dates),
        )
