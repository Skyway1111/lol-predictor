"""What a serving replay records: the end-of-day snapshots the query path bisects."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import torch


@dataclass
class TeamRecord:
    """One team appearance day, as the query path reads it.

    Attributes:
        five: the fielded five as player vocab ids (-1 unknown).
        tier: the raw league-tier value of that game.
        games: the team's cumulative eligible-map count through that day.
        cont: games this five has played together THROUGH this day (run-through).
    """

    day: date
    five: list[int]
    tier: float
    games: int
    cont: float


@dataclass
class ServingTimelines:
    """End-of-day snapshots the query path bisects (all strictly-before-as_of reads).

    Attributes:
        player_states/org_states: lists of state tensors parallel to their date lists.
        prior: every player's model prior row (the debut/unseen fallback at query time).
    """

    player_dates: dict[int, list[date]] = field(default_factory=dict)
    player_states: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    org_dates: dict[int, list[date]] = field(default_factory=dict)
    org_states: dict[int, list[torch.Tensor]] = field(default_factory=dict)
    team_records: dict[int, list[TeamRecord]] = field(default_factory=dict)
    prior: torch.Tensor | None = None


@dataclass
class BagServingTimelines:
    """K members' ``ServingTimelines`` from one K-batched serving replay, stored K-WIDE.

    Attributes:
        player_states/org_states: one ``[K,...]`` row per (entity, day), never K per-member
            tensors — that fan-out measures ~6.6 GB of object overhead at K=5 vs ~0.1 GB flat.
        prior: the stacked ``[K, n_players, d]`` prior. Date lists and ``TeamRecord``s are
            member-independent (functions of the shared corpus only) and stored once.
    """

    K: int
    player_dates: dict[int, list[date]] = field(default_factory=dict)
    player_states: dict[int, list] = field(default_factory=dict)  # lists of [K, d]
    org_dates: dict[int, list[date]] = field(default_factory=dict)
    org_states: dict[int, list] = field(default_factory=dict)  # lists of [K, d_org]
    team_records: dict[int, list[TeamRecord]] = field(default_factory=dict)
    prior: torch.Tensor | None = None  # [K, n_players, d]

    def member(self, k: int) -> ServingTimelines:
        """Member ``k`` in the solo shape: shared dates/records + zero-copy per-member views.

        Returns:
            A deliberately transient view — retaining K of these recreates the per-member
                object fan-out this class exists to avoid.
        """
        assert self.prior is not None  # bag serving replay always populates prior
        return ServingTimelines(
            player_dates=self.player_dates,
            player_states={p: [r[k] for r in rows] for p, rows in self.player_states.items()},
            org_dates=self.org_dates,
            org_states={o: [r[k] for r in rows] for o, rows in self.org_states.items()},
            team_records=self.team_records,
            prior=self.prior[k],
        )


def _record_team_days(
    tl: BagServingTimelines | ServingTimelines, db, day: date, n_org_games: list[int]
) -> None:
    """Append each org's ``TeamRecord`` for one day (same-day entries collapse to the last)."""
    tier_ch = 0  # CTX_CHANNELS[0] == "league_tier" (raw, unnormalized in db.ctx)
    n_games = db.a_won.shape[0]
    for g in range(n_games):
        for side in (0, 1):
            oi = int(db.orgs[g, side])
            n_org_games[oi] += 1
            five = [int(p) for p in db.players[g, side].tolist()]
            rec = TeamRecord(
                day=day,
                five=five,
                tier=float(db.ctx[g, side, tier_ch]),
                games=n_org_games[oi],
                cont=float(db.roster_cont[g, side]) + 1.0,  # run-through = run-before + 1
            )
            recs = tl.team_records.setdefault(oi, [])
            if recs and recs[-1].day == day:
                recs[-1] = rec
            else:
                recs.append(rec)


def _snapshot_day(
    tl: ServingTimelines,
    db,
    H,
    H_org,
    *,
    n_org_games: list[int],
) -> None:
    """End-of-day state snapshots + per-team records (same-day entries collapse to the last)."""
    day = db.date
    seen_pids = {int(p) for p in db.players.reshape(-1).tolist() if p >= 0}
    for pid in seen_pids:
        tl.player_dates.setdefault(pid, []).append(day)
        tl.player_states.setdefault(pid, []).append(H[pid].detach().clone())

    _record_team_days(tl, db, day, n_org_games)

    for oi in {int(o) for o in db.orgs.reshape(-1).tolist()}:
        tl.org_dates.setdefault(oi, []).append(day)
        tl.org_states.setdefault(oi, []).append(H_org[oi].detach().clone())


def _snapshot_day_bag(
    tl: BagServingTimelines,
    db,
    H,
    H_org,
    *,
    n_org_games: list[int],
) -> None:
    """The K-batched mirror of ``_snapshot_day``: end-of-day snapshots for ALL K members.

    Args:
        tl: receives states cloned K-WIDE — one ``[K, d]`` row per (entity, day), never split
            into per-member tensors; the member-independent structure is stored once. Same-day
            collapse and the known-five carry mirror the solo function exactly.
    """
    day = db.date
    seen_pids = {int(p) for p in db.players.reshape(-1).tolist() if p >= 0}
    for pid in seen_pids:
        tl.player_dates.setdefault(pid, []).append(day)
        tl.player_states.setdefault(pid, []).append(H[:, pid].detach().clone())

    _record_team_days(tl, db, day, n_org_games)

    for oi in {int(o) for o in db.orgs.reshape(-1).tolist()}:
        tl.org_dates.setdefault(oi, []).append(day)
        tl.org_states.setdefault(oi, []).append(H_org[:, oi].detach().clone())
