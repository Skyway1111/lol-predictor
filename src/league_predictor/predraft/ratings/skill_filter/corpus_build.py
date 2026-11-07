"""The DB scan behind the corpus: one chronological pass over rating-eligible maps."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING

from league_predictor.predraft.cleaning import (
    clear_enriched_cache,
    competitive_map_eligible,
    enriched_matches,
)
from league_predictor.predraft.league_tier import INTERNATIONAL_EVENTS, league_tier
from league_predictor.predraft.ratings.skill_filter.corpus_schema import (
    CTX_CHANNELS,
    PLAYER_CHANNELS,
    TEAM_CHANNELS,
    Corpus,
    DayBatch,
    NormStats,
    Vocab,
)
from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Mapping

_TIER_VALUE = {"tier1": 1.0, "tier2": 2.0, "tier3": 3.0}


def corpus_signature(conn: sqlite3.Connection, *, through: date) -> tuple[int, str, str]:
    """``Corpus.signature`` without building tensors, vocab, or normalization.

    Returns:
        (count, latest day, max match id) over the same ordered ``enriched_matches`` rows and
            ``competitive_map_eligible`` predicate :func:`_build_corpus_uncached` scans — the
            seconds-scale sidecar check that validates a materialized serving replay.
    """
    try:
        rows = [
            match
            for match in enriched_matches(conn, end=through.isoformat())
            if competitive_map_eligible(match)
        ]
        return (
            len(rows),
            max((str(match["date"])[:10] for match in rows), default=""),
            max((str(match["match_id"]) for match in rows), default=""),
        )
    finally:
        clear_enriched_cache()


def _build_corpus_uncached(  # noqa: C901, PLR0915 - one pass over shared day bookkeeping
    conn, *, through: date, norm_through: date
) -> Corpus:
    """The pure DB scan behind ``corpus.build_corpus`` (one chronological pass, no caching)."""
    import torch

    rows = [
        m for m in enriched_matches(conn, end=through.isoformat()) if competitive_map_eligible(m)
    ]
    extras = _match_extras(conn, through=through)
    lineups = _lineup_stats(conn, through=through)

    player_index: dict[str, int] = {}
    org_index: dict[str, int] = {}
    league_index: dict[str, int] = {}
    debut_league: list[int] = []

    last_seen: dict[str, date] = {}  # player → last appearance day
    roster_run: dict[str, int] = defaultdict(int)  # team → games its current five has run together
    team_last_day: dict[str, date] = {}
    team_last_five: dict[str, frozenset] = {}
    team_last_patch: dict[str, object] = {}
    team_games: dict[str, int] = defaultdict(int)

    by_day: dict[date, list[dict]] = {}
    n_games = 0
    max_match_id = ""

    # Bookkeeping updates flush at END of day, so every same-day appearance reads start-of-day
    # facts — the day-batching semantics the replay's predictions and updates assume.
    pending_day: date | None = None
    pending_players: dict[str, date] = {}
    pending_teams: list[dict] = []

    def _flush_day():
        """Commit the pending day's last-seen and roster bookkeeping."""
        last_seen.update(pending_players)
        seen_tids: set[str] = set()
        for upd in pending_teams:
            tid = upd["tid"]
            if tid not in seen_tids:  # roster run grows day-by-day: one update per team per day
                seen_tids.add(tid)
                same = (
                    team_last_five.get(tid) is not None
                    and upd["five"] is not None
                    and team_last_five[tid] == upd["five"]
                )
                roster_run[tid] = roster_run[tid] + 1 if same else 1
            team_last_day[tid] = upd["day"]
            if upd["five"] is not None:
                team_last_five[tid] = upd["five"]
            if upd["patch"] is not None:
                team_last_patch[tid] = upd["patch"]
            team_games[tid] += 1
        pending_players.clear()
        pending_teams.clear()

    for m in rows:
        mid = str(m["match_id"])
        day = date.fromisoformat(str(m["date"])[:10])
        if pending_day is not None and day != pending_day:
            _flush_day()
        pending_day = day
        t1, t2 = str(m["team1_id"]), str(m["team2_id"])
        team_a, team_b = (t1, t2) if t1 <= t2 else (t2, t1)
        tournament = str(m["tournament"] or "")
        extra = extras.get(mid, {})
        league_idx = league_index.setdefault(tournament, len(league_index))

        sides: list[dict] = []
        for tid in (team_a, team_b):
            role_map = lineups.get((mid, tid), {})
            slot_pids, slot_gaps, slot_debut = [], [], []
            slot_perf, slot_perf_mask = [], []
            current_five = []
            for role in ROLE_ORDER:
                entry = role_map.get(role)
                if entry is None:
                    slot_pids.append(-1)
                    slot_gaps.append(0.0)
                    slot_debut.append(1.0)
                    slot_perf.append([0.0] * len(PLAYER_CHANNELS))
                    slot_perf_mask.append([0.0] * len(PLAYER_CHANNELS))
                    continue
                pid, stats = entry
                current_five.append(pid)
                if pid not in player_index:
                    player_index[pid] = len(player_index)
                    debut_league.append(league_idx)
                slot_pids.append(player_index[pid])
                seen = last_seen.get(pid)
                slot_gaps.append(float((day - seen).days) if seen else 0.0)
                slot_debut.append(0.0 if seen else 1.0)
                vals, mask = [], []
                for ch in PLAYER_CHANNELS:
                    v = _f(stats.get(ch))
                    vals.append(v if v is not None else 0.0)
                    mask.append(1.0 if v is not None else 0.0)
                slot_perf.append(vals)
                slot_perf_mask.append(mask)

            sides.append(
                {
                    "tid": tid,
                    "pids": slot_pids,
                    "gaps": slot_gaps,
                    "debut": slot_debut,
                    "perf": slot_perf,
                    "perf_mask": slot_perf_mask,
                    "five": frozenset(current_five) if len(current_five) == N_ROLES else None,
                    "role_map": role_map,
                }
            )

        for side, opp in ((sides[0], sides[1]), (sides[1], sides[0])):
            tid = side["tid"]
            side["team_feat"], side["team_mask"] = _team_channels(
                side["role_map"], opp["role_map"], m
            )
            prev_five = team_last_five.get(tid)
            churn = (
                float(len(prev_five - side["five"]))
                if prev_five is not None and side["five"] is not None
                else 0.0
            )
            # Lineup continuity (start-of-day): prior consecutive games of the CURRENT five; 0 on
            # a churn game or incomplete five. The serving record stores this + 1 (run-through).
            side["roster_cont"] = (
                float(roster_run.get(tid, 0)) if churn == 0.0 and side["five"] is not None else 0.0
            )
            prev_day = team_last_day.get(tid)
            prev_patch = team_last_patch.get(tid)
            patch = extra.get("patch")
            blue = extra.get("blue_side_team")
            raw_gap = float((day - prev_day).days) if prev_day else 0.0
            side["team_gap"] = raw_gap
            side["ctx"] = [
                _TIER_VALUE[league_tier(tournament)],
                1.0 if tournament in INTERNATIONAL_EVENTS else 0.0,
                float(_f(extra.get("game_number")) or 1.0),
                0.0 if blue is None else (1.0 if str(blue) == tid else -1.0),
                1.0
                if (prev_patch is not None and patch is not None and patch != prev_patch)
                else 0.0,
                _log1p(raw_gap),
                churn,
                _log1p(float(team_games[tid])),
            ]

        # Queue bookkeeping for the end-of-day flush.
        for side in sides:
            tid = side["tid"]
            for pid, _stats in side["role_map"].values():
                pending_players[pid] = day
            pending_teams.append(
                {
                    "tid": tid,
                    "day": day,
                    "five": side["five"],
                    "patch": extra.get("patch"),
                }
            )
            side["org"] = org_index.setdefault(tid, len(org_index))

        by_day.setdefault(day, []).append(
            {
                "mid": mid,
                "position": (str(m["series_id"]), int(m["game_no"])),
                "a_won": 1.0 if str(m["winner"]) == team_a else 0.0,
                "sides": sides,
            }
        )
        n_games += 1
        max_match_id = max(max_match_id, mid)

    if pending_day is not None:
        _flush_day()

    days = [_stack_day(torch, d, games) for d, games in sorted(by_day.items())]
    norm = _norm_stats(torch, days, norm_through)
    vocab = Vocab(player_index, org_index, league_index, debut_league)
    last_day = days[-1].date.isoformat() if days else ""
    # Release the ``enriched_matches`` memo this full-history scan populated (~2.0 GB of row
    # dicts); later same-window consumers repopulate it with one repeat scan.
    clear_enriched_cache()
    return Corpus(days=days, vocab=vocab, norm=norm, signature=(n_games, last_day, max_match_id))


def _f(x) -> float | None:
    """Robust numeric coercion (SQLite TEXT affinity can hand back strings)."""
    if x is None or x == "":
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _log1p(x: float) -> float:
    """log1p clamped at zero (the fixed representation for gap/count ctx channels)."""
    import math

    return math.log1p(max(0.0, x))


def _match_extras(conn, *, through: date) -> dict[str, dict]:
    """match_id → {game_number, patch, blue_side_team} for finished games before ``through``."""
    rows = conn.execute(
        """SELECT g.game_id AS match_id, g.game_no AS game_number,
                  g.patch AS patch, g.blue_team AS blue_side_team
           FROM games g JOIN series s ON s.series_id = g.series_id
           WHERE g.winner IS NOT NULL AND s.scheduled_ts < ?""",
        (through.isoformat(),),
    ).fetchall()
    return {
        str(r["match_id"]): {
            "game_number": r["game_number"],
            "patch": r["patch"],
            "blue_side_team": r["blue_side_team"],
        }
        for r in rows
    }


# Exactly the stats-row keys read downstream; projecting to them avoids a full ``dict(r)`` per
# row (~900k rows). Dropping one that IS read would silently zero a channel.
_LINEUP_STAT_KEYS: tuple[str, ...] = (
    *PLAYER_CHANNELS,
    "teamkills",
    "firsttower",
    "firstdragon",
    "firstbaron",
)


def _lineup_stats(conn, *, through: date) -> dict[tuple[str, str], dict]:
    """(match_id, team_id) → {role: (player_id, stats projected to ``_LINEUP_STAT_KEYS``)}."""
    cols = ", ".join(f"gp.{c}" for c in PLAYER_CHANNELS)
    rows = conn.execute(
        f"""SELECT gp.game_id AS match_id, gp.team_id, gp.player_id, gp.role,
                   gp.teamkills, gp.teamdeaths,
                   gp.firsttower, gp.firstdragon, gp.firstbaron, {cols}
            FROM game_players gp
            JOIN games g ON g.game_id = gp.game_id
            JOIN series s ON s.series_id = g.series_id
            WHERE g.winner IS NOT NULL AND s.scheduled_ts < ? AND gp.role IS NOT NULL""",
        (through.isoformat(),),
    ).fetchall()
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (str(r["match_id"]), str(r["team_id"]))
        role_map = out.setdefault(key, {})
        # A duplicate-role glitch keeps the first row seen.
        role_map.setdefault(
            str(r["role"]), (str(r["player_id"]), {k: r[k] for k in _LINEUP_STAT_KEYS})
        )
    return out


def _team_channels(role_map: Mapping, opp_role_map: Mapping, match: Mapping):
    """One side's TEAM_CHANNELS values + mask from its player rows."""
    own_rows = [stats for _pid, stats in role_map.values()]
    opp_rows = [stats for _pid, stats in opp_role_map.values()]

    feat = [0.0] * len(TEAM_CHANNELS)
    mask = [0.0] * len(TEAM_CHANNELS)

    own_tk = _first_present(own_rows, "teamkills")
    opp_tk = _first_present(opp_rows, "teamkills")
    if own_tk is not None and opp_tk is not None:
        feat[0], mask[0] = own_tk - opp_tk, 1.0

    gd15 = [_f(s.get("golddiffat15")) for s in own_rows]
    if len(gd15) == N_ROLES and all(v is not None for v in gd15):
        feat[1], mask[1] = sum(v for v in gd15 if v is not None), 1.0

    length = _f(match.get("gamelength"))
    if length is not None:
        feat[2], mask[2] = length / 60.0, 1.0

    for i, ch in enumerate(("firsttower", "firstdragon", "firstbaron"), start=3):
        vals = [_f(s.get(ch)) for s in own_rows]
        present = [v for v in vals if v is not None]
        if present:
            feat[i], mask[i] = max(present), 1.0
    return feat, mask


def _first_present(rows: Iterable[dict], key: str) -> float | None:
    """The first numeric ``key`` across a side's rows (team-level columns repeat per row)."""
    for s in rows:
        v = _f(s.get(key))
        if v is not None:
            return v
    return None


def _stack_day(torch, day: date, games: Iterable[dict]) -> DayBatch:
    """Stack one day's accumulated per-game dicts into DayBatch tensors."""

    def per_side(field):
        return [[g["sides"][0][field], g["sides"][1][field]] for g in games]

    return DayBatch(
        date=day,
        match_ids=[g["mid"] for g in games],
        positions=[g["position"] for g in games],
        players=torch.tensor(
            [[s["pids"] for s in (g["sides"][0], g["sides"][1])] for g in games], dtype=torch.int64
        ),
        orgs=torch.tensor(
            [[g["sides"][0]["org"], g["sides"][1]["org"]] for g in games], dtype=torch.int64
        ),
        a_won=torch.tensor([g["a_won"] for g in games], dtype=torch.float32),
        gaps=torch.tensor(per_side("gaps"), dtype=torch.float32),
        debut=torch.tensor(per_side("debut"), dtype=torch.float32),
        roster_cont=torch.tensor(per_side("roster_cont"), dtype=torch.float32),
        team_gaps=torch.tensor(per_side("team_gap"), dtype=torch.float32),
        perf=torch.tensor(per_side("perf"), dtype=torch.float32),
        perf_mask=torch.tensor(per_side("perf_mask"), dtype=torch.float32),
        team_feat=torch.tensor(per_side("team_feat"), dtype=torch.float32),
        team_mask=torch.tensor(per_side("team_mask"), dtype=torch.float32),
        ctx=torch.tensor(per_side("ctx"), dtype=torch.float32),
    )


def _norm_stats(torch, days: Iterable[DayBatch], norm_through: date) -> NormStats:
    """Mask-aware mean/std per channel over games strictly before ``norm_through``.

    Returns:
        ``NormStats`` with std floored at 1e-6, unseen channels 0/1, and global player stats
            pooled over slots (``[P]``).
    """
    in_span = [d for d in days if d.date < norm_through]

    def moments(v, mk, reduce_dims):
        count = mk.sum(dim=reduce_dims).clamp(min=1.0)
        mean = (v * mk).sum(dim=reduce_dims) / count
        var = ((v - mean) ** 2 * mk).sum(dim=reduce_dims) / count
        seen = mk.sum(dim=reduce_dims) > 0
        mean = torch.where(seen, mean, torch.zeros_like(mean))
        std = torch.where(seen, var.sqrt().clamp(min=1e-6), torch.ones_like(var))
        return mean, std

    if not in_span:  # no games before the cutoff -> served 0/1, no variants (the off path is safe)
        nP, nT, nC = len(PLAYER_CHANNELS), len(TEAM_CHANNELS), len(CTX_CHANNELS)
        return NormStats(
            player_mean=torch.zeros(nP),
            player_std=torch.ones(nP),
            team_mean=torch.zeros(nT),
            team_std=torch.ones(nT),
            ctx_mean=torch.zeros(nC),
            ctx_std=torch.ones(nC),
        )

    # Concatenate each tensor across in-span days ONCE, then reduce every stat from it.
    perf = torch.cat([d.perf for d in in_span])
    perf_mask = torch.cat([d.perf_mask for d in in_span])
    team_feat = torch.cat([d.team_feat for d in in_span])
    team_mask = torch.cat([d.team_mask for d in in_span])
    ctx = torch.cat([d.ctx for d in in_span])

    t_mean, t_std = moments(team_feat, team_mask, (0, 1))
    c_mean, c_std = moments(ctx, torch.ones_like(ctx), (0, 1))
    # Global player stats (``[P]``, counts), pooled over games, sides, and slots.
    p_mean, p_std = moments(perf, perf_mask, (0, 1, 2))  # global [P]
    return NormStats(
        player_mean=p_mean,
        player_std=p_std,
        team_mean=t_mean,
        team_std=t_std,
        ctx_mean=c_mean,
        ctx_std=c_std,
    )
