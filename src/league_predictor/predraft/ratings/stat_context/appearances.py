"""One date-ordered facts scan into the column-factorized appearance rows the grid fits."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.predraft.ratings.player_scoring import resolve_tournament_tier
from league_predictor.predraft.ratings.stat_context.vocab import (
    _AT10_SCALE,
    _ROLE_INDEX,
    _STAT_INDEX,
    STATS,
    game_kda,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Collection


def _load_team_strengths(conn: sqlite3.Connection) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per team: (asof ordinals ascending, strengths) from ``team_state``.

    Raises:
        RuntimeError: no rows at all (the control would silently price every
            opponent as average), or rows under more than one owning model id (mixed
            strength scales) — both change what the published numbers mean.
    """
    owners = [row[0] for row in conn.execute("SELECT DISTINCT model_id FROM team_state")]
    if len(owners) > 1:
        raise RuntimeError(
            f"team_state carries rows under {len(owners)} model ids ({sorted(owners)}); "
            "the single-owner contract is broken — run the analytics refresh "
            "(drop_foreign_team_state) and retry"
        )
    rows = conn.execute(
        "SELECT team_id, asof_date, strength FROM team_state "
        "WHERE strength IS NOT NULL ORDER BY team_id, asof_date"
    ).fetchall()
    if not rows:
        raise RuntimeError(
            "team_state is empty, so the opponent-strength control has nothing to read; "
            "run `backfill rating` (or the analytics refresh) to materialize it, then retry"
        )
    by_team: dict[str, tuple[list[int], list[float]]] = {}
    for team_id, asof, strength in rows:
        days, values = by_team.setdefault(team_id, ([], []))
        days.append(date.fromisoformat(asof).toordinal())
        values.append(float(strength))
    return {
        team: (np.asarray(days, dtype=np.int64), np.asarray(values))
        for team, (days, values) in by_team.items()
    }


def _blind_lookup(conn: sqlite3.Connection) -> tuple[set[str], set[tuple[str, str, str]]]:
    """(games with a stored draft, (game_id, side, role) triples picked blind)."""
    known = {
        row[0]
        for row in conn.execute(
            "SELECT DISTINCT game_id FROM draft_actions WHERE kind = 'pick' AND role IS NOT NULL"
        )
    }
    blind = {
        (row[0], row[1], row[2])
        for row in conn.execute(
            """
            SELECT game_id, side, role FROM (
                SELECT game_id, side, role,
                       ROW_NUMBER() OVER (PARTITION BY game_id ORDER BY seq) AS seq_rank
                FROM draft_actions WHERE kind = 'pick' AND role IS NOT NULL
            ) WHERE seq_rank <= 2
            """
        )
    }
    return known, blind


@dataclass(frozen=True, slots=True)
class _Appearances:
    """Column-factorized appearance rows plus the context factors the grid estimates over."""

    player_ids: tuple[str, ...]
    players: np.ndarray
    dates: np.ndarray
    roles: np.ndarray
    tiers: np.ndarray
    wins: np.ndarray
    sides: np.ndarray  # 1 blue, 0 red, -1 unknown
    blinds: np.ndarray  # 1 blind, 0 counter, -1 unknown
    opp_strength: np.ndarray  # NaN unknown
    values: np.ndarray  # [n, len(STATS)], NaN missing
    games: np.ndarray
    team1: np.ndarray  # 1 team_a, 0 team_b, -1 neither
    game_dates: np.ndarray
    game_team1_won: np.ndarray
    champions: np.ndarray  # factorized champion index, -1 unknown
    n_champions: int


# Row-column layout of the appearance query, resolved to stat slots once at import.
_LANING_COLUMNS = tuple(
    (_STAT_INDEX[s15], a15, a10)
    for s15, a15, a10 in (
        ("golddiffat15", 4, 7),
        ("csdiffat15", 5, 8),
        ("xpdiffat15", 6, 9),
    )
)
_PLAIN_COLUMNS = tuple(
    (_STAT_INDEX[name], column)
    for name, column in (
        ("cspm", 10),
        ("dpm", 11),
        ("killparticipation", 12),
        ("firstblood", 16),
        ("vspm", 17),
        ("earnedgoldshare", 18),
    )
)
_KDA_COLUMN = _STAT_INDEX["kda"]


def _fill_stat_values(out_row: np.ndarray, row) -> None:
    """Write one appearance's stats into ``out_row`` (@15 over scaled @10); absent stays NaN."""
    for stat_idx, c15, c10 in _LANING_COLUMNS:
        if row[c15] is not None:
            out_row[stat_idx] = float(row[c15])
        elif row[c10] is not None:
            out_row[stat_idx] = float(row[c10]) * _AT10_SCALE
    for stat_idx, column in _PLAIN_COLUMNS:
        if row[column] is not None:
            out_row[stat_idx] = float(row[column])
    kills, deaths, assists = row[13], row[14], row[15]
    # All three or absent: a partially NULL box line must not score a real (wrong) KDA.
    if kills is not None and deaths is not None and assists is not None:
        out_row[_KDA_COLUMN] = game_kda(float(kills), float(deaths), float(assists))


def _side_and_blind(
    row, known_drafts: Collection[str], blind_triples: Collection[tuple[str, str, str]]
) -> tuple[int, int]:
    """``(side, blind)`` codes for one appearance; -1 where the side or the draft is unknown."""
    team_id, blue_team, game_id = row[1], row[20], row[19]
    if blue_team is None or team_id not in (row[23], row[24]):
        return -1, -1
    side = 1 if team_id == blue_team else 0
    if game_id not in known_drafts:
        return side, -1
    side_name = "blue" if side == 1 else "red"
    return side, (1 if (game_id, side_name, row[2]) in blind_triples else 0)


def _load_appearances(  # noqa: PLR0915 - linear factorization of sixteen parallel columns
    conn: sqlite3.Connection, through: date
) -> _Appearances:
    """One date-ordered scan of every scored appearance strictly before ``through``."""
    strengths = _load_team_strengths(conn)
    known_drafts, blind_triples = _blind_lookup(conn)
    rows = conn.execute(
        """
        SELECT gp.player_id, gp.team_id, gp.role, gp.result,
               gp.golddiffat15, gp.csdiffat15, gp.xpdiffat15,
               gp.golddiffat10, gp.csdiffat10, gp.xpdiffat10,
               gp.cspm, gp.dpm, gp.killparticipation,
               gp.kills, gp.deaths, gp.assists, gp.firstblood, gp.vspm, gp.earnedgoldshare,
               gp.game_id, g.blue_team, s.league_id, date(s.scheduled_ts) AS day,
               s.team_a, s.team_b, g.winner, gp.champion_id
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE date(s.scheduled_ts) < ? AND gp.role IS NOT NULL AND gp.result IN (0, 1)
          AND gp.player_id IS NOT NULL
        ORDER BY date(s.scheduled_ts), s.series_id, g.game_no, gp.player_id
        """,
        (through.isoformat(),),
    ).fetchall()

    n = len(rows)
    player_ids: list[str] = []
    player_index: dict[str, int] = {}
    game_index: dict[str, int] = {}
    game_dates: list[int] = []
    game_team1_won: list[int] = []
    champion_index: dict[str, int] = {}
    champions = np.empty(n, dtype=np.int32)
    games = np.empty(n, dtype=np.int32)
    team1 = np.empty(n, dtype=np.int8)
    players = np.empty(n, dtype=np.int32)
    dates = np.empty(n, dtype=np.int64)
    roles = np.empty(n, dtype=np.int16)
    tiers = np.empty(n, dtype=np.int8)
    wins = np.empty(n, dtype=np.int8)
    sides = np.empty(n, dtype=np.int8)
    blinds = np.empty(n, dtype=np.int8)
    opp_strength = np.full(n, np.nan)
    values = np.full((n, len(STATS)), np.nan)

    tier_cache: dict[str, int] = {}
    date_cache: dict[str, int] = {}

    for i, row in enumerate(rows):
        pid = row[0]
        idx = player_index.get(pid)
        if idx is None:
            idx = len(player_ids)
            player_index[pid] = idx
            player_ids.append(pid)
        players[i] = idx
        day = row[22]
        ordinal = date_cache.get(day)
        if ordinal is None:
            ordinal = date.fromisoformat(day).toordinal()
            date_cache[day] = ordinal
        dates[i] = ordinal
        roles[i] = _ROLE_INDEX[row[2]]
        league = row[21] or ""
        tier = tier_cache.get(league)
        if tier is None:
            tier = min(max(resolve_tournament_tier(league), 1), 3)
            tier_cache[league] = tier
        tiers[i] = tier
        wins[i] = int(row[3])

        team_id, game_id = row[1], row[19]
        gidx = game_index.get(game_id)
        if gidx is None:
            gidx = len(game_dates)
            game_index[game_id] = gidx
            game_dates.append(ordinal)
            winner = row[25]
            game_team1_won.append(1 if winner == row[23] else 0 if winner == row[24] else -1)
        games[i] = gidx
        team1[i] = 1 if team_id == row[23] else 0 if team_id == row[24] else -1
        champion = row[26]
        if champion is None:
            champions[i] = -1
        else:
            cidx = champion_index.get(champion)
            if cidx is None:
                cidx = len(champion_index)
                champion_index[champion] = cidx
            champions[i] = cidx
        sides[i], blinds[i] = _side_and_blind(row, known_drafts, blind_triples)

        opp = row[24] if team_id == row[23] else row[23] if team_id == row[24] else None
        if opp is not None and opp in strengths:
            days_arr, vals_arr = strengths[opp]
            pos = int(np.searchsorted(days_arr, ordinal)) - 1  # strictly before the game day
            if pos >= 0:
                opp_strength[i] = vals_arr[pos]

        _fill_stat_values(values[i], row)

    return _Appearances(
        player_ids=tuple(player_ids),
        players=players,
        dates=dates,
        roles=roles,
        tiers=tiers,
        wins=wins,
        sides=sides,
        blinds=blinds,
        opp_strength=opp_strength,
        values=values,
        games=games,
        team1=team1,
        game_dates=np.asarray(game_dates, dtype=np.int64),
        game_team1_won=np.asarray(game_team1_won, dtype=np.int8),
        champions=champions,
        n_champions=len(champion_index),
    )
