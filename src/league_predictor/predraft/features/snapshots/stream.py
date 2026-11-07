"""The ``original`` family's snapshot stack: timelines in memory."""  # sightline-ok: 27 - one stack

# file-length-ok: the groups share one timeline shape and one visibility rule; splitting them
# apart would put the store's row layout in two files.

from __future__ import annotations

import heapq
import logging
import math
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime, timedelta
from itertools import islice
from statistics import fmean
from typing import TYPE_CHECKING, NamedTuple, TypedDict

from league_predictor import config
from league_predictor.asof import AsOf
from league_predictor.db.draft_scores_read import preferred_context_predicate
from league_predictor.decay import half_life_rate, half_life_weight
from league_predictor.predraft.league_tier import INTERNATIONAL_EVENTS, resolve_domestic_league
from league_predictor.predraft.ratings.player_performance import (
    _as_of_datetime,
    _parse_match_datetime,
)
from league_predictor.predraft.ratings.player_performance_prefix import (
    all_player_performance_prefixes,
    build_player_performance_prefixes,
)
from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Collection, Iterator, Mapping, Sequence

    from league_predictor.predraft.ratings.player_performance_prefix import (
        _PlayerPerformancePrefix,
    )

logger = logging.getLogger(__name__)

#: The composite's and the synergy's recency half-life in days: faster than the 60-day
#: player-profile default, a deliberate form signal.
_HALF_LIFE_DAYS = 20.0

#: Window sizes, in games, of the rolling groups.
_FORM_WINDOW = 15
_MOMENTUM_WINDOW = 20
_SLOPE_WINDOW = 7
_MIN_SLOPE_RESULTS = 3
_OBJECTIVE_WINDOW = 15
_PACE_WINDOW = 15
_VARIANCE_WINDOW = 7
_UPSET_WINDOW = 30
_SIDE_WINDOW = 20
_WIN_MARGIN_WINDOW = 15
_CHAMP_POOL_GAMES = 20  # per roster player, pooled across the roster
_GAME_PHASE_WINDOW = 15
_EMA_WINDOW = 40
_BAYESIAN_WINDOW = 30
_BAYESIAN_PRIOR = 8.0
_DRAFT_WINDOW = 20
_H2H_DECAY_DAYS = 180.0
#: The observed-roster scan reads this many role rows (about fifty five-role games).
_ROSTER_ROWS = 250
#: Roster stability reads this many distinct ``(series start, player)`` pairs.
_STABILITY_PAIRS = 200

_EMA_HALF_LIVES = (("ema_wr_fast", 5.0), ("ema_wr_medium", 12.0), ("ema_wr_slow", 25.0))
_EMA_RATES = tuple(half_life_rate(hl) for _, hl in _EMA_HALF_LIVES)

#: Game-length cuts, in seconds: a win under the stomp cut is a stomp, a game over the slow
#: cut is a slow one, a game over the late cut counts toward the late-game index.
_STOMP_SECONDS = 1500
_SLOW_GAME_SECONDS = 2100
_LATE_GAME_SECONDS = 1800
#: Missing game lengths read as a typical game rather than dropping the row.
_TYPICAL_GAME_SECONDS = 1800
#: Below this many long games the late-game index has no support and reads as the coin.
_MIN_LATE_GAMES = 3
_GRADE_POINTS = {"S": 5.0, "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}

#: Above this many requested teams the store scans the whole facts DB instead of filtering.
_BULK_TEAMS = 200


def _v(x) -> float:
    """Coerce a nullable DB value to float, defaulting None to 0.0."""
    return float(x) if x is not None else 0.0


# --- the timelines --------------------------------------------------------------------


class _Game(NamedTuple):
    """One team's view of one map."""

    key: tuple[str, str, int]  # (series start, series_id, game_no): the facts order
    day: date
    when: datetime  # the series start as naive UTC
    game_id: str
    opp: str
    league: str | None
    winner: str | None
    won: int  # winner == team; an unknown winner reads as a loss where a group takes every game
    on_blue: bool | None  # None when the map carries no blue side
    stats: tuple | None  # game_team_stats: see _STATS_COLUMNS
    opp_gold: float | None
    opp_has_stats: bool
    rows: tuple[tuple[str | None, str], ...]  # the team's (role, player_id) rows, rowid order
    result: int | None  # the first row's result
    first_blood: int | None  # MAX(firstblood) over the rows
    at15: tuple[float | None, float | None, float | None]  # AVG of the three @15 diffs
    draft: tuple[float, str, str] | None  # p_blue, grade_blue, grade_red


_STATS_COLUMNS = (
    "gamelength",
    "teamkills",
    "dragons",
    "barons",
    "towers",
    "void_grubs",
    "firstdragon",
    "firstbaron",
    "firsttower",
    "firstherald",
    "totalgold",
)
_ST_LENGTH, _ST_KILLS, _ST_DRAGONS, _ST_BARONS, _ST_TOWERS, _ST_GRUBS = range(6)
_ST_FIRST_DRAGON, _ST_FIRST_BARON, _ST_FIRST_TOWER, _ST_FIRST_HERALD, _ST_GOLD = range(6, 11)


class _Timeline:
    """One team's games in the facts order plus the position lists each group bisects."""

    __slots__ = (
        "by_lineup",
        "by_opp",
        "drafted",
        "every",
        "games",
        "keys",
        "known",
        "nonintl",
        "phase",
        "rostered",
        "sided",
        "stats",
        "stats_known",
    )

    def __init__(self, games: list[_Game]) -> None:
        self.games = games
        self.keys = [g.key for g in games]
        self.every = list(range(len(games)))
        self.known = [i for i, g in enumerate(games) if g.winner is not None]
        self.sided = [i for i in self.known if games[i].on_blue is not None]
        self.stats = [i for i, g in enumerate(games) if g.stats is not None]
        self.rostered = [i for i, g in enumerate(games) if g.rows]
        self.stats_known = [i for i in self.stats if games[i].winner is not None]
        self.phase = [i for i in self.stats_known if games[i].rows]
        self.drafted = [i for i in self.sided if games[i].draft is not None]
        # nonintl[i]: the latest position <= i whose league resolves domestically, else -1.
        nonintl: list[int] = []
        last = -1
        for i, g in enumerate(games):
            if g.league is not None and g.league not in INTERNATIONAL_EVENTS:
                last = i
            nonintl.append(last)
        self.nonintl = nonintl
        by_lineup: dict[frozenset[str], list[int]] = defaultdict(list)
        by_opp: dict[str, list[int]] = defaultdict(list)
        for i, g in enumerate(games):
            if g.rows:
                by_lineup[frozenset(pid for _, pid in g.rows)].append(i)
            if g.winner is not None:
                by_opp[g.opp].append(i)
        self.by_lineup = dict(by_lineup)
        self.by_opp = dict(by_opp)


class _View(NamedTuple):
    """What a query sees of a timeline: the prefix ``[0, lo)`` and the own-series run."""

    lo: int
    run_lo: int
    run_hi: int


def _view(keys: Sequence[tuple[str, str, int]], as_of: AsOf) -> _View:
    """The one visibility rule as three bisects (``recency`` carries the same three)."""
    lo = bisect_left(keys, (as_of.ts, "", 0))
    if not as_of.series_id:
        return _View(lo, lo, lo)
    run_lo = bisect_left(keys, (as_of.ts, as_of.series_id, 0))
    run_hi = bisect_left(keys, as_of.position)
    return _View(lo, run_lo, run_hi)


def _newest_first(sub: Sequence[int] | None, view: _View) -> Iterator[int]:
    """Visible positions of ``sub`` (a sorted position list; None for all) newest first."""
    if sub is None:
        yield from range(view.run_hi - 1, view.run_lo - 1, -1)
        yield from range(view.lo - 1, -1, -1)
        return
    for j in range(bisect_left(sub, view.run_hi) - 1, bisect_left(sub, view.run_lo) - 1, -1):
        yield sub[j]
    for j in range(bisect_left(sub, view.lo) - 1, -1, -1):
        yield sub[j]


def _tail(sub: Sequence[int] | None, view: _View, limit: int | None) -> list[int]:
    """The newest ``limit`` visible positions of ``sub`` (all of them for None)."""
    return list(islice(_newest_first(sub, view), limit))


# --- the loader -----------------------------------------------------------------------


def _scope(teams: set[str] | None) -> tuple[str, tuple]:
    """The SQL fragment restricting a facts read to ``teams``' series, and its bindings."""
    if teams is None:
        return "", ()
    ids = sorted(teams)
    marks = ", ".join("?" * len(ids))
    return f" AND (s.team_a IN ({marks}) OR s.team_b IN ({marks}))", (*ids, *ids)


_FACTS = " FROM games g JOIN series s ON s.series_id = g.series_id WHERE s.scheduled_ts IS NOT NULL"


def _draft_scores(conn, scope: str, params: tuple) -> dict[str, tuple[float, str, str]]:
    """``game_id -> (p_blue, grade_blue, grade_red)`` for the one ``draft_scores`` model.

    Raises:
        RuntimeError: ``draft_scores`` carries more than one model_id (a torn backfill).
    """
    has_table = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'draft_scores'"
    ).fetchone()
    if has_table is None:
        return {}
    # Resolved from draft_scores itself, not the models table: a rolled-back grader stays
    # registered there (out-of-horizon pick_scores keep it), so "newest models row" lies.
    ids = [str(r[0]) for r in conn.execute("SELECT DISTINCT model_id FROM draft_scores LIMIT 2")]
    if not ids:
        return {}
    if len(ids) > 1:
        raise RuntimeError(
            "draft_scores carries more than one model_id (a torn backfill?): "
            "it is single-id by construction; rerun the draft backfill to reconcile"
        )
    return {
        str(gid): (float(p_blue), str(grade_blue), str(grade_red))
        for gid, p_blue, grade_blue, grade_red in conn.execute(
            "SELECT ds.game_id, ds.p_blue, ds.grade_blue, ds.grade_red"  # noqa: S608 - code-literal fragments
            " FROM draft_scores ds JOIN games g ON g.game_id = ds.game_id"
            " JOIN series s ON s.series_id = g.series_id"
            f" WHERE ds.model_id = ? AND {preferred_context_predicate()}"
            f" AND s.scheduled_ts IS NOT NULL{scope}",
            (ids[0], *params),
        )
    }


def _avg(values: list) -> float | None:
    """SQL ``AVG``: the mean of the non-NULL values, NULL when there are none."""
    present = [float(v) for v in values if v is not None]
    return fmean(present) if present else None


def _load_timelines(conn, teams: set[str] | None) -> dict[str, _Timeline]:
    """Every team's timeline (``teams`` None) or the requested teams', from four ordered reads."""
    scope, params = _scope(teams)
    stats = {
        (gid, tid): tuple(rest)
        for gid, tid, *rest in conn.execute(
            f"SELECT t.game_id, t.team_id, {', '.join('t.' + c for c in _STATS_COLUMNS)}"  # noqa: S608 - code-literal fragments
            " FROM game_team_stats t JOIN games g ON g.game_id = t.game_id"
            f" JOIN series s ON s.series_id = g.series_id WHERE s.scheduled_ts IS NOT NULL{scope}",
            params,
        )
    }
    rows_by: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for gid, tid, *row in conn.execute(
        "SELECT gp.game_id, gp.team_id, gp.role, gp.player_id, gp.result, gp.firstblood,"  # noqa: S608 - code-literal fragments
        " gp.golddiffat15, gp.csdiffat15, gp.xpdiffat15"
        " FROM game_players gp JOIN games g ON g.game_id = gp.game_id"
        f" JOIN series s ON s.series_id = g.series_id WHERE s.scheduled_ts IS NOT NULL{scope}"
        " ORDER BY gp.rowid",
        params,
    ):
        rows_by[(gid, tid)].append(tuple(row))
    drafts = _draft_scores(conn, scope, params)

    timelines: dict[str, list[_Game]] = defaultdict(list)
    for gid, sid, ts, gno, team_a, team_b, winner, blue, league in conn.execute(
        "SELECT g.game_id, s.series_id, s.scheduled_ts, g.game_no, s.team_a, s.team_b,"
        f" g.winner, g.blue_team, s.league_id{_FACTS}{scope}"
        " ORDER BY s.scheduled_ts, s.series_id, g.game_no",
        params,
    ):
        key = (str(ts), str(sid), int(gno))
        day = date.fromisoformat(key[0][:10])
        when = _parse_match_datetime(key[0])
        for team, opp in ((team_a, team_b), (team_b, team_a)):
            if teams is not None and team not in teams:
                continue
            rows = rows_by.get((gid, team), [])
            first_bloods = [r[3] for r in rows if r[3] is not None]
            opp_stats = stats.get((gid, opp))
            timelines[team].append(
                _Game(
                    key=key,
                    day=day,
                    when=when,
                    game_id=str(gid),
                    opp=opp,
                    league=league,
                    winner=winner,
                    won=int(winner == team),
                    on_blue=None if blue is None else blue == team,
                    stats=stats.get((gid, team)),
                    opp_gold=None if opp_stats is None else opp_stats[_ST_GOLD],
                    opp_has_stats=opp_stats is not None,
                    rows=tuple((r[0], str(r[1])) for r in rows),
                    result=rows[0][2] if rows else None,
                    first_blood=max(first_bloods) if first_bloods else None,
                    at15=(
                        _avg([r[4] for r in rows]),
                        _avg([r[5] for r in rows]),
                        _avg([r[6] for r in rows]),
                    ),
                    draft=drafts.get(gid),
                )
            )
    return {team: _Timeline(games) for team, games in timelines.items()}


def _roster_table_lineup(conn, team_id: str, day: str) -> dict[str, str]:
    """The ``rosters`` snapshot rows active on ``day``; ``{}`` without the table or rows."""
    # The facts schema carries no rosters table (a team with no fielded history has no known
    # roster), so the append-only snapshot fallback is probed: absent means empty.
    has_rosters = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='rosters'"
    ).fetchone()
    if has_rosters is None:
        return {}
    roster: dict[str, str] = {}
    # Several rows per role (a mid-period transfer): the latest start wins.
    for role, player_id in conn.execute(
        "SELECT role, player_id FROM rosters WHERE team_id = ? AND start_date <= ?"
        " AND (end_date IS NULL OR end_date > ?) ORDER BY start_date DESC",
        (team_id, day, day),
    ):
        roster.setdefault(role, player_id)
    return roster


# --- the groups -----------------------------------------------------------------------


# Keep the fitted feature arithmetic: statistics.linear_regression changes float rounding.
def _linear_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Compute the slope of a simple linear regression (OLS) on xs vs ys."""
    n = len(xs)
    if n < 2:  # noqa: PLR2004 - a line needs two points
        return 0.0
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    den = sum((x - x_mean) ** 2 for x in xs)
    return num / den if den > 0 else 0.0


def recent_form(tl: _Timeline, view: _View) -> dict[str, float]:
    """win_rate_10, win_rate_20 (the FULL window despite the name), win_streak, loss_streak."""
    tail = _tail(tl.every, view, _FORM_WINDOW)
    if not tail:
        return {"win_rate_10": 0.0, "win_rate_20": 0.0, "win_streak": 0, "loss_streak": 0}
    results = [tl.games[i].won for i in tail]
    last10 = results[:10]
    win_streak = loss_streak = 0
    for r in results:
        if r != 1:
            break
        win_streak += 1
    for r in results:
        if r != 0:
            break
        loss_streak += 1
    return {
        "win_rate_10": sum(last10) / len(last10),
        # Win rate over the full window; the key name is fixed by features/manifest.py.
        "win_rate_20": sum(results) / len(results),
        "win_streak": win_streak,
        "loss_streak": loss_streak,
    }


def momentum(tl: _Timeline, view: _View, as_of: AsOf) -> dict[str, float]:
    """win_rate_5, performance_slope over the last seven results, days_since_last."""
    tail = _tail(tl.known, view, _MOMENTUM_WINDOW)
    if not tail:
        return {"win_rate_5": 0.0, "performance_slope": 0.0, "days_since_last": 999.0}
    results = [tl.games[i].won for i in tail]
    last5 = results[:5]
    last7 = results[:_SLOPE_WINDOW]
    slope = (
        _linear_slope(list(range(len(last7))), last7) if len(last7) >= _MIN_SLOPE_RESULTS else 0.0
    )
    return {
        "win_rate_5": sum(last5) / len(last5),
        "performance_slope": slope,
        "days_since_last": float((as_of.day - tl.games[tail[0]].day).days),
    }


def _stats_rows(tl: _Timeline, view: _View, limit: int) -> list[tuple]:
    """The newest ``limit`` visible team-stats rows."""
    return [st for i in _tail(tl.stats, view, limit) if (st := tl.games[i].stats) is not None]


def objective_control(tl: _Timeline, view: _View) -> dict[str, float]:
    """Objective-control rates from game_team_stats; NULL objective columns contribute 0."""
    rows = _stats_rows(tl, view, _OBJECTIVE_WINDOW)
    if not rows:
        return dict.fromkeys(
            (
                "dragon_rate",
                "baron_rate",
                "tower_rate",
                "first_dragon_rate",
                "first_baron_rate",
                "first_tower_rate",
                "first_herald_rate",
                "grub_rate",
            ),
            0.0,
        )
    n = len(rows)
    return {
        "dragon_rate": sum(_v(r[_ST_DRAGONS]) for r in rows) / n,
        "baron_rate": sum(_v(r[_ST_BARONS]) for r in rows) / n,
        "tower_rate": sum(_v(r[_ST_TOWERS]) for r in rows) / n,
        "first_dragon_rate": sum(1 for r in rows if r[_ST_FIRST_DRAGON] == 1) / n,
        "first_baron_rate": sum(1 for r in rows if r[_ST_FIRST_BARON] == 1) / n,
        "first_tower_rate": sum(1 for r in rows if r[_ST_FIRST_TOWER] == 1) / n,
        "first_herald_rate": sum(1 for r in rows if r[_ST_FIRST_HERALD] == 1) / n,
        "grub_rate": sum(_v(r[_ST_GRUBS]) for r in rows) / n,
    }


def pace(tl: _Timeline, view: _View) -> dict[str, float]:
    """avg_game_duration, kills_per_min over team stats; first_blood_rate over player rows."""
    rows = _stats_rows(tl, view, _PACE_WINDOW)
    if not rows:
        return {"avg_game_duration": 0.0, "kills_per_min": 0.0, "first_blood_rate": 0.0}
    n = len(rows)
    avg_dur = sum(_v(r[_ST_LENGTH]) for r in rows) / n
    total_kills = sum(_v(r[_ST_KILLS]) for r in rows)
    kpm = (total_kills / n) / (avg_dur / 60.0) if avg_dur > 0 else 0.0
    fielded = _tail(tl.rostered, view, _PACE_WINDOW)
    fb_rate = sum(1 for i in fielded if tl.games[i].first_blood) / len(fielded) if fielded else 0.0
    return {"avg_game_duration": avg_dur, "kills_per_min": kpm, "first_blood_rate": fb_rate}


# Keep the fitted rounding and hot-path cost; see benchmarks.md.
def _std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:  # noqa: PLR2004 - a variance needs two observations
        return 0.0
    mu = sum(vals) / n
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / n)


def performance_variance(tl: _Timeline, view: _View) -> dict[str, float]:
    """Game-duration and kills-per-minute standard deviations over the last seven games."""
    rows = _stats_rows(tl, view, _VARIANCE_WINDOW)
    durations = [_v(r[_ST_LENGTH]) for r in rows if _v(r[_ST_LENGTH]) > 0]
    kpms = [_v(r[_ST_KILLS]) / (_v(r[_ST_LENGTH]) / 60.0) for r in rows if _v(r[_ST_LENGTH]) > 0]
    return {"gamedur_variance": _std(durations), "kpm_variance": _std(kpms)}


def synergy(tl: _Timeline, view: _View, player_ids: Sequence[str], as_of: AsOf) -> dict[str, float]:
    """Games the listed players all played for the team, their win rate, and its decayed form."""
    empty = {"games_together": 0, "lineup_win_rate": 0.0, "weighted_win_rate": 0.0}
    if not player_ids:
        return empty
    need, wanted = len(player_ids), set(player_ids)
    positions: list[int] = []
    for lineup, at in tl.by_lineup.items():
        if len(lineup & wanted) == need:
            positions.extend(_tail(at, view, None))
    if not positions:
        return empty
    positions.sort(reverse=True)  # newest first: the order the decayed sum runs in
    games = [tl.games[i] for i in positions]
    wins = sum(1 for g in games if g.result == 1)
    as_of_dt = _as_of_datetime(as_of.day)
    total_weight = weighted_wins = 0.0
    for g in games:
        days_ago = max(0.0, (as_of_dt - g.when).total_seconds() / 86400.0)
        w = half_life_weight(days_ago, _HALF_LIFE_DAYS)
        total_weight += w
        if g.result == 1:
            weighted_wins += w
    return {
        "games_together": len(games),
        "lineup_win_rate": wins / len(games),
        "weighted_win_rate": weighted_wins / total_weight if total_weight else 0.0,
    }


def team_region(tl: _Timeline, view: _View) -> str | None:
    """The region of the team's latest visible non-international league, or None."""
    if view.run_hi > view.run_lo:
        own = tl.games[view.run_hi - 1].league
        if own is not None and own not in INTERNATIONAL_EVENTS:
            return resolve_domestic_league(own)
    j = tl.nonintl[view.lo - 1] if view.lo else -1
    return resolve_domestic_league(tl.games[j].league) if j >= 0 else None


def schedule_density(tl: _Timeline, view: _View, as_of: AsOf) -> dict[str, int]:
    """How many decided games the team played in the last 7 / 14 days."""
    own = bisect_left(tl.known, view.run_hi) - bisect_left(tl.known, view.run_lo)
    before = bisect_left(tl.known, view.lo)

    def since(days: int) -> int:
        first = bisect_left(tl.keys, ((as_of.day - timedelta(days=days)).isoformat(), "", 0))
        return max(0, before - bisect_left(tl.known, first)) + own

    return {"games_7d": since(7), "games_14d": since(14)}


def observed_roster(tl: _Timeline, view: _View) -> dict[str, str]:
    """The latest visible game carrying all five roles, within the ``_ROSTER_ROWS`` scan."""
    budget = _ROSTER_ROWS
    for i in _newest_first(tl.rostered, view):
        rows = [(role, pid) for role, pid in tl.games[i].rows if role is not None][:budget]
        budget -= len(rows)
        lineup: dict[str, str] = {}
        for role, pid in rows:
            lineup.setdefault(role, pid)
        if all(role in lineup for role in ROLE_ORDER):
            return {role: lineup[role] for role in ROLE_ORDER}
        if budget <= 0:
            break
    return {}


def _role_weighted_means(
    profiles: Mapping[str, dict], role_weights: Mapping[str, float] | None
) -> dict[tuple[str, str], float]:
    """``(pillar, stat) -> role-weighted mean`` over the roster profiles that report the stat."""
    total: dict[tuple[str, str], float] = {}
    weight: dict[tuple[str, str], float] = {}
    for role, profile in profiles.items():
        w = (role_weights or {}).get(role, 1.0)
        for pillar, stats in _COMPOSITE_STATS.items():
            section = profile.get(pillar, {})
            for stat in stats:
                if stat in section:
                    pair = (pillar, stat)
                    total[pair] = total.get(pair, 0.0) + section[stat] * w
                    weight[pair] = weight.get(pair, 0.0) + w
    return {pair: (value / weight[pair] if weight[pair] else 0.0) for pair, value in total.items()}


_COMPOSITE_STATS = {
    "laning": ("golddiffat15", "csdiffat15", "xpdiffat15", "cspm"),
    "teamfighting": ("dpm", "damageshare", "killparticipation", "kda", "earnedgoldshare"),
    "vision": ("vspm", "controlwardsbought"),
}


def team_composite(
    roster: Mapping[str, str], prefixes: Mapping[str, _PlayerPerformancePrefix], as_of: AsOf
) -> dict[str, dict[str, float]]:
    """Role-weighted means of the roster's player profiles as of the query's series start.

    Returns:
        ``avg_laning`` / ``avg_teamfighting`` / ``avg_vision``, or ``{}`` when no roster
        player has a game before the query's series (its own earlier maps never count).
    """
    profiles: dict[str, dict] = {}
    for role, player_id in roster.items():
        prefix = prefixes.get(player_id)
        profile = prefix.profile_at(as_of.ts) if prefix is not None else None
        if profile is not None:
            profiles[role] = profile
    if not profiles:
        return {}
    means = _role_weighted_means(profiles, config.ROLE_WEIGHTS)
    return {
        f"avg_{pillar}": {stat: means.get((pillar, stat), 0.0) for stat in stats}
        for pillar, stats in _COMPOSITE_STATS.items()
    }


def _newest_pairs(tl: _Timeline, view: _View) -> dict[str, set[str]]:
    """The newest ``_STABILITY_PAIRS`` distinct (series start, player) pairs, grouped by day."""
    seen: set[tuple[str, str]] = set()
    day_players: dict[str, set[str]] = {}
    for i in _newest_first(tl.rostered, view):
        game = tl.games[i]
        for _, pid in game.rows:
            pair = (game.key[0], pid)
            if pair in seen:
                continue
            seen.add(pair)
            day_players.setdefault(game.key[0][:10], set()).add(pid)
            if len(seen) == _STABILITY_PAIRS:
                return day_players
    return day_players


def roster_stability(
    tl: _Timeline, view: _View, roster: Mapping[str, str], as_of: AsOf
) -> dict[str, int]:
    """How long the current five-man roster has been together, in match days and days."""
    player_ids = set(roster.values())
    if len(player_ids) < N_ROLES:
        return {"roster_days_stable": 0, "roster_total_games": 0}
    day_players = _newest_pairs(tl, view)
    stable_games = 0
    first_stable: str | None = None
    for match_day in sorted(day_players, reverse=True):
        fielded = day_players[match_day]
        if len(fielded) >= N_ROLES and fielded == player_ids:
            stable_games += 1
            first_stable = match_day
        elif len(fielded) >= N_ROLES:
            break
    if first_stable is None:
        return {"roster_days_stable": 0, "roster_total_games": stable_games}
    return {
        "roster_days_stable": (as_of.day - date.fromisoformat(first_stable)).days,
        "roster_total_games": stable_games,
    }


def upset(tl: _Timeline, view: _View) -> dict[str, float]:
    """Windowed loss rate (``upset_rate``) and win rate (``favorite_rate``), which sum to 1."""
    tail = _tail(tl.known, view, _UPSET_WINDOW)
    if not tail:
        return {"upset_rate": 0.5, "favorite_rate": 0.5}
    wr = sum(tl.games[i].won for i in tail) / len(tail)
    return {"upset_rate": 1.0 - wr, "favorite_rate": wr}


def side(tl: _Timeline, view: _View) -> dict[str, float]:
    """Team-specific blue/red side win rates."""
    tail = _tail(tl.sided, view, _SIDE_WINDOW)
    if not tail:
        return {"blue_wr": 0.5, "red_wr": 0.5, "side_diff": 0.0}
    blue = [tl.games[i].won for i in tail if tl.games[i].on_blue]
    red = [tl.games[i].won for i in tail if not tl.games[i].on_blue]
    blue_wr = sum(blue) / len(blue) if blue else 0.5
    red_wr = sum(red) / len(red) if red else 0.5
    return {"blue_wr": blue_wr, "red_wr": red_wr, "side_diff": blue_wr - red_wr}


def win_margin(tl: _Timeline, view: _View) -> dict[str, float]:
    """How decisively the team wins or loses, over maps where both sides have team stats."""
    empty = {"avg_gold_diff_end": 0.0, "stomp_rate": 0.0, "close_game_rate": 0.0}
    rows = [
        tl.games[i]
        for i in _tail(tl.stats_known, view, _WIN_MARGIN_WINDOW)
        if tl.games[i].opp_has_stats
    ]
    if not rows:
        return empty
    gold_diffs: list[float] = []
    stomps = close_games = 0
    for g in rows:
        assert g.stats is not None
        my_gold, gamelength = g.stats[_ST_GOLD], g.stats[_ST_LENGTH]
        if my_gold is None or g.opp_gold is None:
            continue
        diff = float(my_gold) - float(g.opp_gold)
        gold_diffs.append(diff if g.won else -diff)
        gl = float(gamelength) if gamelength else _TYPICAL_GAME_SECONDS
        if gl < _STOMP_SECONDS and g.won:
            stomps += 1
        elif gl > _SLOW_GAME_SECONDS:
            close_games += 1
    if not gold_diffs:
        return empty
    # stomp_rate / close_game_rate divide by ALL rows; quirk locked in features/manifest.py.
    return {
        "avg_gold_diff_end": sum(gold_diffs) / len(gold_diffs),
        "stomp_rate": stomps / len(rows),
        "close_game_rate": close_games / len(rows),
    }


def champion_pool(
    prefixes: Mapping[str, _PlayerPerformancePrefix], player_ids: Sequence[str], as_of: AsOf
) -> dict[str, float]:
    """How diverse the roster's champion pool is over its newest pooled games."""
    # The newest _CHAMP_POOL_GAMES rows per roster player on any team, pooled across the
    # roster and cut as one list; rows at one facts position rank by player id.
    if not player_ids:
        return {"unique_champs": 0, "avg_champ_diversity": 0.0}
    limit = _CHAMP_POOL_GAMES * len(player_ids)
    ranked = sorted(set(player_ids))
    candidates: list[tuple[tuple[str, str, int], int, int]] = []
    for rank, pid in enumerate(ranked):
        prefix = prefixes.get(pid)
        if prefix is None:
            continue
        view = _view(prefix.keys, as_of)
        candidates.extend((prefix.keys[j], -rank, j) for j in _tail(None, view, limit))
    top = heapq.nlargest(limit, candidates) if len(candidates) > limit else candidates
    player_champs: dict[str, set[str]] = defaultdict(set)
    for _key, neg_rank, j in top:
        pid = ranked[-neg_rank]
        champ = prefixes[pid].champions[j]
        if champ:
            player_champs[pid].add(str(champ))
    if not player_champs:
        return {"unique_champs": 0, "avg_champ_diversity": 0.0}
    diversities = [len(champs) for champs in player_champs.values()]
    return {
        "unique_champs": len(set().union(*player_champs.values())),
        "avg_champ_diversity": sum(diversities) / len(diversities),
    }


def game_phase(tl: _Timeline, view: _View) -> dict[str, float]:
    """Early vs late game team strength index."""
    rows = [tl.games[i] for i in _tail(tl.phase, view, _GAME_PHASE_WINDOW)]
    if not rows:
        return {"early_game_index": 0.0, "late_game_index": 0.0}
    # Each stat is scaled by a typical @15 advantage (500 gold, 10 cs, 500 xp)
    # so the three land on comparable ~±1 units before averaging.
    early_game, n_early = 0.0, 0
    for column, scale in ((0, 500.0), (1, 10.0), (2, 500.0)):
        values = [v for g in rows if (v := g.at15[column]) is not None]
        if values:
            early_game += sum(values) / len(values) / scale
            n_early += 1
    if n_early:
        early_game /= n_early
    long_total = long_wins = 0
    for g in rows:
        assert g.stats is not None
        gl = g.stats[_ST_LENGTH]
        if gl is not None and float(gl) > _LATE_GAME_SECONDS:
            long_total += 1
            long_wins += g.won
    late_game = long_wins / long_total if long_total >= _MIN_LATE_GAMES else 0.5
    return {"early_game_index": early_game, "late_game_index": late_game}


def ema_win_rates(tl: _Timeline, view: _View, as_of: AsOf) -> dict[str, float]:
    """EMA win rates at 5/12/25-day half-lives (``ema_wr_fast``/``_medium``/``_slow``)."""
    tail = _tail(tl.known, view, _EMA_WINDOW)
    if not tail:
        return {key: 0.5 for key, _ in _EMA_HALF_LIVES}
    sums = [0.0, 0.0, 0.0]
    totals = [0.0, 0.0, 0.0]
    for i in tail:
        g = tl.games[i]
        days_ago = max((as_of.day - g.day).days, 0)
        for idx, rate in enumerate(_EMA_RATES):
            w = math.exp(-rate * days_ago)
            sums[idx] += w * g.won
            totals[idx] += w
    return {
        key: sums[idx] / totals[idx] if totals[idx] > 0 else 0.5
        for idx, (key, _) in enumerate(_EMA_HALF_LIVES)
    }


def bayesian_win_rate(tl: _Timeline, view: _View) -> dict[str, float]:
    """Win rate shrunk toward 0.5 with ``_BAYESIAN_PRIOR`` pseudocounts, plus the sample size."""
    tail = _tail(tl.known, view, _BAYESIAN_WINDOW)
    wins = sum(tl.games[i].won for i in tail)
    return {
        "bayesian_wr": (wins + _BAYESIAN_PRIOR / 2.0) / (len(tail) + _BAYESIAN_PRIOR),
        "sample_size_30d": float(len(tail)),
    }


def draft_tendency(tl: _Timeline, view: _View) -> dict[str, float]:
    """Rolling draft-quality metrics from ``draft_scores``, side-corrected for the team."""
    rows = [tl.games[i] for i in _tail(tl.drafted, view, _DRAFT_WINDOW)]
    if not rows:
        return {
            "draft_score_avg": 0.5,
            "draft_grade_avg": 3.0,
            "draft_win_rate": 0.5,
            "draft_wr_5": 0.5,
            "draft_wr_10": 0.5,
        }
    scores: list[float] = []
    grades: list[float] = []
    draft_wins: list[float] = []
    for g in rows:
        assert g.draft is not None
        p_blue, grade_blue, grade_red = g.draft
        score = p_blue if g.on_blue else 1.0 - p_blue
        scores.append(score)
        grades.append(_GRADE_POINTS.get(grade_blue if g.on_blue else grade_red, 3.0))
        draft_wins.append(1.0 if score > 0.5 else 0.0)  # noqa: PLR2004 - the two-class argmax cut
    n = len(scores)
    return {
        "draft_score_avg": sum(scores) / n,
        "draft_grade_avg": sum(grades) / n,
        "draft_win_rate": sum(draft_wins) / n,
        "draft_wr_5": sum(draft_wins[:5]) / min(5, n),
        "draft_wr_10": sum(draft_wins[:10]) / min(10, n),
    }


def head_to_head(tl: _Timeline, opponent: str, as_of: AsOf) -> dict[str, float]:
    """H2H games, raw and recency-weighted win rate for the timeline's team, days since."""
    positions = tl.by_opp.get(opponent)
    tail = _tail(positions, _view(tl.keys, as_of), None) if positions else []
    if not tail:
        return {"h2h_games": 0, "h2h_win_rate": 0.5, "h2h_recency_wr": 0.5, "h2h_days_since": 365}
    total_weight = weighted_wins = 0.0
    raw_wins = 0
    for i in tail:
        g = tl.games[i]
        weight = math.exp(-max((as_of.day - g.day).days, 0) / _H2H_DECAY_DAYS)
        total_weight += weight
        weighted_wins += weight * g.won
        raw_wins += g.won
    return {
        "h2h_games": len(tail),
        "h2h_win_rate": raw_wins / len(tail),
        "h2h_recency_wr": weighted_wins / total_weight if total_weight > 0 else 0.5,
        "h2h_days_since": (as_of.day - tl.games[tail[0]].day).days,
    }


# --- the store ------------------------------------------------------------------------


class TeamSnapshot(TypedDict):
    """The single-team feature groups returned together at one query point."""

    composite: dict[str, dict[str, float]]
    form: dict[str, float]
    momentum: dict[str, float]
    objective: dict[str, float]
    pace: dict[str, float]
    variance: dict[str, float]
    synergy: dict[str, float]
    roster: dict[str, str]
    region: str | None
    schedule: dict[str, int]
    roster_stability: dict[str, int]
    upset: dict[str, float]
    side: dict[str, float]
    win_margin: dict[str, float]
    champ_pool: dict[str, float]
    game_phase: dict[str, float]
    ema_wr: dict[str, float]
    bayesian_wr: dict[str, float]
    draft: dict[str, float]


class SnapshotStore:
    """Timelines for a set of teams (or every team) and the snapshots they answer.

    Args:
        teams: the teams to load now; ``None`` or more than ``_BULK_TEAMS`` scans the whole
            facts DB once (every team, every player's career). A smaller set loads only
            those teams' series and reads roster players' careers on demand, cached per
            process; a team asked for later loads on first use.
    """

    def __init__(self, conn: sqlite3.Connection, teams: Collection[str] | None = None) -> None:
        self._conn = conn
        bulk = teams is None or len(teams) > _BULK_TEAMS
        self._timelines = _load_timelines(conn, None if bulk else set(teams or ()))
        self._prefixes = all_player_performance_prefixes(conn, _HALF_LIFE_DAYS) if bulk else None

    def _timeline(self, team_id: str) -> _Timeline:
        timeline = self._timelines.get(team_id)
        if timeline is None:
            if self._prefixes is None:
                self._timelines.update(_load_timelines(self._conn, {team_id}))
            timeline = self._timelines.setdefault(team_id, _Timeline([]))
        return timeline

    def _prefixes_for(self, player_ids: Collection[str]) -> dict[str, _PlayerPerformancePrefix]:
        if self._prefixes is None:
            return build_player_performance_prefixes(self._conn, list(player_ids), _HALF_LIFE_DAYS)
        return {pid: p for pid in player_ids if (p := self._prefixes.get(pid)) is not None}

    def roster(self, team_id: str, as_of: AsOf | date | datetime) -> dict[str, str]:
        """``{role: player_id}`` the query sees: the observed lineup, else the rosters table."""
        as_of = AsOf.of(as_of)
        timeline = self._timeline(team_id)
        return observed_roster(timeline, _view(timeline.keys, as_of)) or _roster_table_lineup(
            self._conn, team_id, as_of.day.isoformat()
        )

    def team_snapshot(self, team_id: str, as_of: AsOf | date | datetime) -> TeamSnapshot:
        """Every single-team group as one query sees it.

        Returns:
            Group name -> that group's block, heterogeneous by group; the caller narrows.
        """
        as_of = AsOf.of(as_of)
        tl = self._timeline(team_id)
        view = _view(tl.keys, as_of)
        roster = self.roster(team_id, as_of)
        player_ids = list(roster.values())
        prefixes = self._prefixes_for(set(player_ids))
        return {
            "composite": team_composite(roster, prefixes, as_of),
            "form": recent_form(tl, view),
            "momentum": momentum(tl, view, as_of),
            "objective": objective_control(tl, view),
            "pace": pace(tl, view),
            "variance": performance_variance(tl, view),
            "synergy": synergy(tl, view, player_ids, as_of),
            "roster": roster,
            "region": team_region(tl, view),
            "schedule": schedule_density(tl, view, as_of),
            "roster_stability": roster_stability(tl, view, roster, as_of),
            "upset": upset(tl, view),
            "side": side(tl, view),
            "win_margin": win_margin(tl, view),
            "champ_pool": champion_pool(prefixes, player_ids, as_of),
            "game_phase": game_phase(tl, view),
            "ema_wr": ema_win_rates(tl, view, as_of),
            "bayesian_wr": bayesian_win_rate(tl, view),
            "draft": draft_tendency(tl, view),
        }

    def h2h(self, team1_id: str, team2_id: str, as_of: AsOf | date | datetime) -> dict[str, float]:
        """The pair's head-to-head block from ``team1_id``'s side."""
        return head_to_head(self._timeline(team1_id), team2_id, AsOf.of(as_of))
