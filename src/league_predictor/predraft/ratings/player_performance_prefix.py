"""Prefix-sum acceleration for player performance profiles (ratings-architecture.md)."""

from __future__ import annotations

import sqlite3
from bisect import bisect_left
from itertools import groupby
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from league_predictor.predraft.feature_cache import get_cache
from league_predictor.predraft.ratings.player_performance import (
    OPTIONAL_STATS,
    PerformanceAggregate,
    _parse_match_datetime,
    assemble_performance_profile,
)
from league_predictor.predraft.ratings.stat_context.vocab import _AT10_SCALE

if TYPE_CHECKING:
    from collections.abc import Iterator

# Prefix keys include the cache's connection identity and transaction-visible facts revision.
NAMESPACE_PLAYER_PERFORMANCE_PREFIX = "player_rating_prefix"

# Columns of the "all-game" weighted cumulative sum (val() semantics: None -> 0.0).
# Index order is load-bearing — the profile reconstruction reads these by number.
_COL_WEIGHT = 0
_COL_CSPM = 1
_COL_DPM = 2
_COL_DAMAGESHARE = 3
_COL_EARNEDGOLDSHARE = 4
_COL_KILLPARTICIPATION = 5
_COL_VSPM = 6
_COL_CONTROLWARDS = 7
_COL_FIRSTBLOOD = 8
_COL_FIRSTTOWER = 9
_COL_FIRSTDRAGON = 10
_COL_FIRSTBARON = 11
_COL_KDA_NUM = 12  # weighted (kills + assists)
_COL_DEATHS = 13
_N_ALL = 14

# Columns of the laning-only weighted cumulative sum (accumulated over games that
# actually have @15 or @10 data, matching laning_weight in compute_performance_profile_from_rows).
_LAN_WEIGHT = 0
_LAN_GD15 = 1
_LAN_CS15 = 2
_LAN_XP15 = 3
_N_LAN = 4

# The per-game columns compute_performance_profile_from_rows consumes plus the game's place
# in the facts order; omits result/teamkills/teamdeaths/gamelength and the optional stats.
_CAREER_COLUMNS = """
    gp.player_id, gp.champion_id, gp.kills, gp.deaths, gp.assists,
    gp.damageshare, gp.killparticipation, gp.vspm, gp.controlwardsbought,
    gp.golddiffat15, gp.csdiffat15, gp.xpdiffat15,
    gp.golddiffat10, gp.csdiffat10, gp.xpdiffat10,
    gp.dpm, gp.cspm, gp.earnedgoldshare,
    gp.firstblood, gp.firsttower, gp.firstdragon, gp.firstbaron,
    s.scheduled_ts AS match_date, s.series_id, g.game_no"""


def _val(x) -> float:
    """float(x) with None -> 0.0, matching compute_performance_profile_from_rows._val."""
    return float(x) if x is not None else 0.0


class _PlayerPerformancePrefix(NamedTuple):
    """One player's full career reduced to as_of-queryable prefix sums.

    Attributes:
        keys: each game's ``(series start, series_id, game_no)``, ascending in the facts
            order; the bisect boundary for a cutoff and the timeline the champion pool reads.
        champions: the champion played in each game, row-aligned with ``keys``.
        cum_all: carries a leading zero row, so ``cum_all[k]`` sums the oldest ``k``
            games — exactly the games with ``series start < cutoff``. ``cum_lan`` matches.
    """

    keys: list[tuple[str, str, int]]
    champions: list[str | None]
    cum_all: np.ndarray  # shape (n+1, _N_ALL)
    cum_lan: np.ndarray  # shape (n+1, _N_LAN)
    n15: np.ndarray  # (n+1,) cumulative count of games with @15 data
    n10: np.ndarray  # (n+1,) cumulative count of games with @10 fallback data
    champ_positions: dict[str, list[int]]  # champion -> ascending game indices

    def profile_at(self, cutoff: str) -> dict | None:
        """Rebuild the profile over the series that started strictly before ``cutoff``.

        Returns:
            None when no game precedes the cutoff. The query's own earlier maps and any
            series that started at the same instant are excluded.
        """
        k = bisect_left(self.keys, (cutoff, "", 0))  # == count of games with start < cutoff
        if k == 0:
            return None

        totals = self.cum_all[k]
        aw = float(totals[_COL_WEIGHT])  # >= k >= 1 (a_i >= 1), never zero
        lan = self.cum_lan[k]
        lanw = float(lan[_LAN_WEIGHT])

        return assemble_performance_profile(
            PerformanceAggregate(
                total_weight=aw,
                laning_weight=lanw,
                sums={
                    "cspm": float(totals[_COL_CSPM]),
                    "dpm": float(totals[_COL_DPM]),
                    "damageshare": float(totals[_COL_DAMAGESHARE]),
                    "earnedgoldshare": float(totals[_COL_EARNEDGOLDSHARE]),
                    "killparticipation": float(totals[_COL_KILLPARTICIPATION]),
                    "kda_numerator": float(totals[_COL_KDA_NUM]),
                    "deaths": float(totals[_COL_DEATHS]),
                    "vspm": float(totals[_COL_VSPM]),
                    "controlwardsbought": float(totals[_COL_CONTROLWARDS]),
                    "firstblood": float(totals[_COL_FIRSTBLOOD]),
                    "firsttower": float(totals[_COL_FIRSTTOWER]),
                    "firstdragon": float(totals[_COL_FIRSTDRAGON]),
                    "firstbaron": float(totals[_COL_FIRSTBARON]),
                },
                laning_sums={
                    "golddiffat15": float(lan[_LAN_GD15]),
                    "csdiffat15": float(lan[_LAN_CS15]),
                    "xpdiffat15": float(lan[_LAN_XP15]),
                },
                optional_sums=dict.fromkeys(OPTIONAL_STATS, 0.0),
                optional_weights=dict.fromkeys(OPTIONAL_STATS, 0.0),
                champions=self._champion_stats(k),
                games=k,
                full_laning_games=int(self.n15[k]),
                fallback_laning_games=int(self.n10[k]),
            )
        )

    def _champion_stats(self, k: int) -> dict[str, tuple[int, int]]:
        """Per champion, ``(count, last_included_index)`` over games before index ``k``.

        Returns:
            ``last_index`` breaks count ties newest-first, matching the row-based profile.
        """
        stats: dict[str, tuple[int, int]] = {}
        for champ, positions in self.champ_positions.items():
            cnt = bisect_left(positions, k)  # appearances with index < k
            if cnt:
                stats[champ] = (cnt, positions[cnt - 1])
        return stats


def _weighted_all_row(r, a: float) -> list[float]:
    """One game's non-laning channels, each scaled by the game's recency weight ``a``."""
    row = [0.0] * _N_ALL
    row[_COL_WEIGHT] = a
    row[_COL_CSPM] = a * _val(r["cspm"])
    row[_COL_DPM] = a * _val(r["dpm"])
    row[_COL_DAMAGESHARE] = a * _val(r["damageshare"])
    row[_COL_EARNEDGOLDSHARE] = a * _val(r["earnedgoldshare"])
    row[_COL_KILLPARTICIPATION] = a * _val(r["killparticipation"])
    row[_COL_VSPM] = a * _val(r["vspm"])
    row[_COL_CONTROLWARDS] = a * _val(r["controlwardsbought"])
    row[_COL_FIRSTBLOOD] = a * _val(r["firstblood"])
    row[_COL_FIRSTTOWER] = a * _val(r["firsttower"])
    row[_COL_FIRSTDRAGON] = a * _val(r["firstdragon"])
    row[_COL_FIRSTBARON] = a * _val(r["firstbaron"])
    row[_COL_KDA_NUM] = a * (_val(r["kills"]) + _val(r["assists"]))
    row[_COL_DEATHS] = a * _val(r["deaths"])
    return row


def _weighted_laning_row(r, a: float) -> tuple[int, int, list[float]]:
    """``(has @15, has @10, weighted laning row)``; prefer @15, fall back to @10 scaled.

    Returns:
        An all-zero row when neither window is complete — the game is not a laning game
        and contributes nothing to the laning-only sums.
    """
    gd15, cs15, xp15 = r["golddiffat15"], r["csdiffat15"], r["xpdiffat15"]
    gd10, cs10, xp10 = r["golddiffat10"], r["csdiffat10"], r["xpdiffat10"]
    has15 = gd15 is not None and cs15 is not None and xp15 is not None
    has10 = gd10 is not None and cs10 is not None and xp10 is not None
    if has15:
        assert gd15 is not None
        assert cs15 is not None
        assert xp15 is not None
        row = [a, a * float(gd15), a * float(cs15), a * float(xp15)]
    elif has10:
        assert gd10 is not None
        assert cs10 is not None
        assert xp10 is not None
        row = [
            a,
            a * float(gd10) * _AT10_SCALE,
            a * float(cs10) * _AT10_SCALE,
            a * float(xp10) * _AT10_SCALE,
        ]
    else:
        row = [0.0, 0.0, 0.0, 0.0]
    return int(has15), int(has10), row


def _build_prefix(rows: list, half_life: float) -> _PlayerPerformancePrefix:
    """Reduce one player's career rows, ascending in the facts order, into a prefix."""
    keys = [(r["match_date"], r["series_id"], int(r["game_no"])) for r in rows]
    champions = [r["champion_id"] for r in rows]
    if not rows:
        return _PlayerPerformancePrefix(
            keys=[],
            champions=[],
            cum_all=np.zeros((1, _N_ALL)),
            cum_lan=np.zeros((1, _N_LAN)),
            n15=np.zeros(1, dtype=np.int64),
            n10=np.zeros(1, dtype=np.int64),
            champ_positions={},
        )

    parsed = [_parse_match_datetime(k[0]) for k in keys]
    t0 = min(parsed)  # first-game anchor; ratios are invariant to this choice
    inv_hl = 1.0 / half_life

    all_rows: list[list[float]] = []
    lan_rows: list[list[float]] = []
    has15_list: list[int] = []
    has10_list: list[int] = []
    champ_positions: dict[str, list[int]] = {}

    for i, r in enumerate(rows):
        days = (parsed[i] - t0).total_seconds() / 86400.0
        a = 2.0 ** (days * inv_hl)

        all_rows.append(_weighted_all_row(r, a))

        has15, has10, lan_row = _weighted_laning_row(r, a)
        has15_list.append(has15)
        has10_list.append(has10)
        lan_rows.append(lan_row)

        champ = r["champion_id"] or "Unknown"
        champ_positions.setdefault(champ, []).append(i)

    zero_all = np.zeros((1, _N_ALL))
    zero_lan = np.zeros((1, _N_LAN))
    cum_all = np.vstack([zero_all, np.cumsum(np.array(all_rows, dtype=np.float64), axis=0)])
    cum_lan = np.vstack([zero_lan, np.cumsum(np.array(lan_rows, dtype=np.float64), axis=0)])
    n15 = np.concatenate([[0], np.cumsum(np.array(has15_list, dtype=np.int64))])
    n10 = np.concatenate([[0], np.cumsum(np.array(has10_list, dtype=np.int64))])
    return _PlayerPerformancePrefix(keys, champions, cum_all, cum_lan, n15, n10, champ_positions)


def _careers(conn: sqlite3.Connection, player_ids: list[str] | None) -> Iterator[tuple[str, list]]:
    """``(player_id, career rows)`` per player, rows ascending in the facts order.

    Args:
        player_ids: the players to read; ``None`` streams every player's career.
    """
    where = ""
    params: tuple = ()
    if player_ids is not None:
        where = f" WHERE gp.player_id IN ({', '.join('?' * len(player_ids))})"
        params = tuple(player_ids)
    cursor = conn.cursor()
    cursor.row_factory = sqlite3.Row
    cursor.execute(
        f"""
        SELECT {_CAREER_COLUMNS}
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id{where}
        ORDER BY gp.player_id, s.scheduled_ts, s.series_id, g.game_no
        """,
        params,
    )
    for player_id, rows in groupby(cursor, key=lambda r: r["player_id"]):
        yield player_id, list(rows)


def all_player_performance_prefixes(
    conn: sqlite3.Connection, half_life: float
) -> dict[str, _PlayerPerformancePrefix]:
    """Every player's prefix from one ordered scan of ``game_players``; uncached (~1M rows)."""
    return {pid: _build_prefix(rows, float(half_life)) for pid, rows in _careers(conn, None)}


def build_player_performance_prefixes(
    conn: sqlite3.Connection, player_ids: list[str], half_life: float
) -> dict[str, _PlayerPerformancePrefix]:
    """Return {player_id: prefix state}, building and caching cold players once per process.

    Returns:
        Career-wide, as_of-independent state cached under
        NAMESPACE_PLAYER_PERFORMANCE_PREFIX, keyed on the write token + half_life.
    """
    cache = get_cache()
    token = cache.make_token(conn)
    hl = float(half_life)

    out: dict[str, _PlayerPerformancePrefix] = {}
    missing: list[str] = []
    for pid in dict.fromkeys(player_ids):  # dedupe, preserve order
        cached = cache.get(NAMESPACE_PLAYER_PERFORMANCE_PREFIX, (*token, pid, hl))
        if cached is None:
            missing.append(pid)
        else:
            out[pid] = cached

    if missing:
        rows_by_player = dict(_careers(conn, missing))
        for pid in missing:
            state = _build_prefix(rows_by_player.get(pid, []), hl)
            cache.set(NAMESPACE_PLAYER_PERFORMANCE_PREFIX, (*token, pid, hl), state)
            out[pid] = state
    return out
