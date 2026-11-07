"""The vintage-cached fit corpus: games, per-year home pools, and the player matrix."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from league_predictor.db.facts_prefix import (
    RATING_REPLAY_INPUTS,
    cache_prefix_digests,
    main_db_path,
)
from league_predictor.predraft.league_tier import RATING_POOL_ALIAS, league_tier

if TYPE_CHECKING:
    import sqlite3

# What makes a tournament international: this many Tier-1 home regions on the board, in at
# least this share of the years it ran.
_MIN_INTL_REGIONS = 2
_MIN_INTL_YEAR_SHARE = 0.5
_MIN_GAMES = 30


def load_games(conn: sqlite3.Connection) -> list[dict]:
    """All finished games as light dicts with the league name pool-folded.

    Returns:
        Dicts with ``players1``/``players2`` scoresheet tuples (possibly short:
        facts rows are not guaranteed ten players per game).
    """
    raw = conn.execute(
        # canonical facts order: (day, series_id, game_no)
        "SELECT g.game_id AS game_id, s.scheduled_ts AS date, s.league_id AS tournament, "
        "s.team_a AS team1_id, s.team_b AS team2_id, g.winner AS winner, "
        "g.blue_team AS blue_team "
        "FROM games g JOIN series s ON s.series_id = g.series_id "
        "WHERE g.winner IS NOT NULL "
        "ORDER BY date(s.scheduled_ts), s.series_id, g.game_no"
    ).fetchall()
    players: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in conn.execute("SELECT game_id, team_id, player_id FROM game_players"):
        players[(row["game_id"], row["team_id"])].append(row["player_id"])
    return [
        {
            "date": g["date"],
            "t": RATING_POOL_ALIAS.get(g["tournament"], g["tournament"]),
            "t1": g["team1_id"],
            "t2": g["team2_id"],
            "w": g["winner"],
            "blue": g["blue_team"],
            "players1": tuple(players.get((g["game_id"], g["team1_id"]), ())),
            "players2": tuple(players.get((g["game_id"], g["team2_id"]), ())),
        }
        for g in raw
    ]


def _home_leagues(games, real_set):
    """Each ``(team, year)``'s modal league among ``real_set`` — its home pool that year."""
    team_year_leagues = defaultdict(Counter)
    for g in games:
        for tid in (g["t1"], g["t2"]):
            if g["t"] in real_set:
                team_year_leagues[(tid, g["date"][:4])][g["t"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in team_year_leagues.items() if c}


def _tier1_regions_by_event(games, home_map):
    """``(tournament, year) -> {Tier-1 home league}`` over the teams that played it."""
    regions_by_event = defaultdict(set)
    for g in games:
        yr = g["date"][:4]
        for tid in (g["t1"], g["t2"]):
            lg = home_map.get((tid, yr))
            if lg and league_tier(lg) == "tier1":
                regions_by_event[(g["t"], yr)].add(lg)
    return regions_by_event


def _home_structure(games, *, tiers=("tier1", "tier2")):
    """Each team's modal home pool by year and the international (tournament, year) events.

    Args:
        games: the games the derivation may see — walk-forward callers pass the
            pre-cutoff prefix ONLY; schedule facts are read, never ``winner``.
        tiers: which tiers count as home-league candidates. The default reproduces
            the skill-filter corpus's map byte-for-byte; the region fit widens to
            all three so feeder leagues become pools.
    Returns:
        ``(home, international)``.
    """
    played = Counter(g["t"] for g in games)
    real_leagues = {t for t, c in played.items() if c >= _MIN_GAMES and league_tier(t) in tiers}

    # EVENT tournaments: >= 2 Tier-1 regions in >= half their years (ignores flukes).
    tier1_by_event = _tier1_regions_by_event(games, _home_leagues(games, real_leagues))
    years_by_tourney, intl_years_by_tourney = defaultdict(set), defaultdict(set)
    for t, yr in {(g["t"], g["date"][:4]) for g in games}:
        years_by_tourney[t].add(yr)
        if len(tier1_by_event[(t, yr)]) >= _MIN_INTL_REGIONS:
            intl_years_by_tourney[t].add(yr)
    event_names = {
        t
        for t in years_by_tourney
        if intl_years_by_tourney[t]
        and len(intl_years_by_tourney[t]) / len(years_by_tourney[t]) >= _MIN_INTL_YEAR_SHARE
    }

    home = _home_leagues(games, real_leagues - event_names)
    reg = _tier1_regions_by_event(games, home)
    international = {
        event for event in reg if event[0] in event_names and len(reg[event]) >= _MIN_INTL_REGIONS
    }
    return home, international


# Both caches key off the corpus vintage, so an ingest invalidates them together; in-memory
# DBs have no stable identity and are never cached.
_STRUCTURE_CACHE: dict = {}
_TABLE_CACHE: dict = {}


# Deliberately full-corpus: an outcome-independent context label for the rating corpus,
# not a fit input of the region model (whose homes derive per cutoff inside ``_fit``).


# A cutoff selects the row prefix ``days < through``; homes and pool/tier columns derive
# from that prefix ONLY (walk-forward) and are memoized per cutoff in ``cutoff_cache``.
@dataclass(frozen=True)
class _FitCorpus:
    """The vintage-cached fit inputs: date-sorted player matrix + per-cutoff context cache."""

    Xp: sparse.csr_matrix  # rows x n_players signed scoresheet indicators
    Xi: sparse.csr_matrix  # rows x 1 signed blue-side intercept column
    days: np.ndarray  # per-row ordinal day, non-decreasing
    y: np.ndarray
    players: tuple[str, ...]
    app_row: np.ndarray  # one entry per (game, scoresheet player) appearance, row-ordered
    app_player: np.ndarray  # that appearance's player column
    app_side: np.ndarray  # that appearance's side, 0 = t1 / 1 = t2
    games: tuple  # the row-aligned light game dicts (home derivation per cutoff)
    cutoff_cache: dict = field(default_factory=dict, compare=False)

    @property
    def n_players(self) -> int:
        return len(self.players)


def _build_fit_corpus(games) -> _FitCorpus:
    """Assemble the cutoff-invariant player matrix behind every fit of this vintage."""

    def _day(g):
        try:
            return date.fromisoformat(g["date"][:10]).toordinal()
        except ValueError:
            return None

    rows = [(g, d) for g in games if (d := _day(g)) is not None]
    players = sorted({p for g, _ in rows for p in g["players1"] + g["players2"]})
    p_idx = {p: i for i, p in enumerate(players)}

    pdata: list[float] = []
    pri: list[int] = []
    pci: list[int] = []
    idata: list[float] = []
    iri: list[int] = []
    days, ys = [], []
    app_row, app_player, app_side = [], [], []
    for i, (g, day) in enumerate(rows):
        days.append(day)
        ys.append(1.0 if g["w"] == g["t1"] else 0.0)
        for side, (pids, sign) in enumerate(((g["players1"], 1.0), (g["players2"], -1.0))):
            for pid in pids:
                pdata.append(sign)
                # sightline-ok: 11 - a sparse design-matrix triple; deer's is an appearance table
                pri.append(i)
                pci.append(p_idx[pid])
                app_row.append(i)
                app_player.append(p_idx[pid])
                app_side.append(side)
        if g["blue"] is not None:
            idata.append(1.0 if g["blue"] == g["t1"] else -1.0)
            iri.append(i)

    n_rows = len(rows)
    return _FitCorpus(
        Xp=sparse.csr_matrix((pdata, (pri, pci)), shape=(n_rows, len(players))),
        Xi=sparse.csr_matrix((idata, (iri, [0] * len(iri))), shape=(n_rows, 1)),
        days=np.asarray(days, dtype=np.int64),
        y=np.asarray(ys),
        players=tuple(players),
        app_row=np.asarray(app_row, dtype=np.int64),
        app_player=np.asarray(app_player, dtype=np.int64),
        app_side=np.asarray(app_side, dtype=np.int64),
        games=tuple(g for g, _ in rows),
    )


def _cache_prefix(conn):
    """Committed and transaction-visible facts identity, absent for in-memory databases."""
    path = main_db_path(conn)
    return (
        (path, cache_prefix_digests(conn, [date.max], RATING_REPLAY_INPUTS)[date.max])
        if path
        else None
    )


def _fit_corpus(conn, prefix):
    """Cache one full-history matrix and discard tables from older facts."""
    if prefix is None:
        return _build_fit_corpus(load_games(conn))
    if prefix not in _STRUCTURE_CACHE:
        _STRUCTURE_CACHE.clear()
        _TABLE_CACHE.clear()
        _STRUCTURE_CACHE[prefix] = _build_fit_corpus(load_games(conn))
    return _STRUCTURE_CACHE[prefix]
