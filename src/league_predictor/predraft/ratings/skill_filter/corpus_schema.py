"""The corpus's shape: which channels exist and the day-grouped tensors that carry them."""

# Orientation is ``backend.load_competitive_maps``' reference convention: side 0 = the
# lexicographically smaller team id, ``a_won`` side-0-relative. Tensors stay RAW + masked.

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import torch

# Per-player performance channels from game_players (post-game facts the UPDATE consumes,
# never the prediction). LPL has 0% @10/@15 coverage; those channels are masked, not imputed.
PLAYER_CHANNELS: tuple[str, ...] = (
    "kills",
    "deaths",
    "assists",
    "golddiffat10",
    "csdiffat10",
    "xpdiffat10",
    "golddiffat15",
    "csdiffat15",
    "xpdiffat15",
    "damageshare",
    "vspm",
    "earnedgoldshare",
)

# Team-level margin channels (side-relative where signed).
TEAM_CHANNELS: tuple[str, ...] = (
    "kill_diff",  # own teamkills - opponent teamkills
    "gold_diff_at15",  # sum of the five golddiffat15 (masked unless all five present)
    "length_minutes",
    "firsttower",
    "firstdragon",
    "firstbaron",
)

# The signed side-0-relative DOMINANCE margins the auxiliary head regresses. gold_diff_at15 is
# LPL-masked (~7% missing) and the composite is a masked mean, so it degrades to kill_diff there.
MARGIN_CHANNELS: tuple[str, ...] = ("kill_diff", "gold_diff_at15")
MARGIN_IDX: list[int] = [TEAM_CHANNELS.index(c) for c in MARGIN_CHANNELS]

# Pre-game prediction context, one vector per side. ``gap_days``/``games_played`` are stored
# log1p-transformed.
CTX_CHANNELS: tuple[str, ...] = (
    "league_tier",  # 1.0 / 2.0 / 3.0 via league_tier()
    "international",  # tournament in INTERNATIONAL_EVENTS
    "game_number",
    "side_blue",  # +1 blue, -1 red, 0 unknown
    "patch_changed",  # patch differs from this team's previous map
    "gap_days",  # log1p(days since the team's previous map)
    "churn_count",  # last role-five vs today's five
    "games_played",  # log1p(team's prior eligible maps)
)


@dataclass(frozen=True)
class Vocab:
    """Contiguous ids for players/orgs/leagues in first-appearance order; -1 means unknown."""

    player_index: dict[str, int]
    org_index: dict[str, int]
    league_index: dict[str, int]
    debut_league: list[int]  # player idx → league idx of first appearance

    @property
    def n_players(self) -> int:
        return len(self.player_index)

    @property
    def n_orgs(self) -> int:
        return len(self.org_index)

    @property
    def n_leagues(self) -> int:
        return len(self.league_index)


def _move(obj, device: str):
    """Copy of a frozen dataclass with every tensor field moved to ``device``."""
    import torch

    moved = {f.name: getattr(obj, f.name) for f in fields(obj)}
    for k, v in moved.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
    return dataclasses.replace(obj, **moved)


@dataclass(frozen=True)
class NormStats:
    """Mask-aware deployed-channel mean/std over games strictly before ``norm_through``."""

    player_mean: torch.Tensor  # torch.Tensor [len(PLAYER_CHANNELS)]
    player_std: torch.Tensor
    team_mean: torch.Tensor  # [len(TEAM_CHANNELS)]
    team_std: torch.Tensor
    ctx_mean: torch.Tensor  # [len(CTX_CHANNELS)]
    ctx_std: torch.Tensor

    def to(self, device: str) -> NormStats:
        return _move(self, device)


@dataclass(frozen=True)
class DayBatch:
    """All eligible maps of one day (G games), stacked."""

    date: date
    match_ids: list[str]
    positions: list[tuple[str, int]]  # (series_id, game_no): the facts position an AsOf names
    players: torch.Tensor  # [G,2,5] int64 in role order, -1 unknown
    orgs: torch.Tensor  # [G,2] int64
    a_won: torch.Tensor  # [G] side-0-relative
    gaps: torch.Tensor  # [G,2,5] RAW days since the slot's previous map
    debut: torch.Tensor  # [G,2,5] 1 where the slot has no prior map
    roster_cont: (
        torch.Tensor  # [G,2] games the team's current five has played together (start-of-day)
    )
    team_gaps: torch.Tensor  # [G,2] RAW days since the team's previous day (org evolve)
    perf: torch.Tensor  # [G,2,5,P]
    perf_mask: torch.Tensor
    team_feat: torch.Tensor  # [G,2,T]
    team_mask: torch.Tensor
    ctx: torch.Tensor  # [G,2,C]

    def to(self, device: str) -> DayBatch:
        return _move(self, device)


@dataclass(frozen=True)
class Corpus:
    """The replay's chronological input; ``signature`` = (n_games, last_day_iso, max_match_id)."""

    days: list[DayBatch]
    vocab: Vocab
    norm: NormStats
    signature: tuple

    def to(self, device: str) -> Corpus:
        """Move every day tensor + the norm stats to ``device``; vocab/signature stay Python."""
        return dataclasses.replace(
            self, days=[d.to(device) for d in self.days], norm=self.norm.to(device)
        )
