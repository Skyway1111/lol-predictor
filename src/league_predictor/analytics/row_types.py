"""Typed row dataclasses for analytics tables; INSERT SQL generates from the fields."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import ClassVar


class RowBase:
    """Mixin for typed analytics rows; generates INSERT SQL from field definitions."""

    __table__: ClassVar[str]

    @classmethod
    def insert_sql(cls) -> str:
        """Auto-generate INSERT statement from field names."""
        names = [f.name for f in dataclasses.fields(cls)]  # pyright: ignore[reportArgumentType]  # concrete subclasses are dataclasses
        cols = ", ".join(names)
        placeholders = ", ".join("?" for _ in names)
        return f"INSERT INTO {cls.__table__} ({cols}) VALUES ({placeholders})"  # noqa: S608 - table and columns are fixed dataclass declarations


@dataclass(frozen=True, slots=True)
class TeamDailySnapshotRow(RowBase):
    __table__ = "team_daily_snapshot"

    team_id: str
    date: str
    display_name: str | None
    region: str | None
    roster_json: str
    win_rate_10: float
    win_rate_5: float
    win_streak: int
    loss_streak: int
    dragon_rate: float
    baron_rate: float
    tower_rate: float
    first_dragon_rate: float
    first_baron_rate: float
    first_tower_rate: float
    avg_gold_diff_15: float
    avg_cs_diff_15: float
    avg_xp_diff_15: float
    # power_rating is the skill-filter-derived team power (see analytics/team_snapshot.py).
    power_rating: float


@dataclass(frozen=True, slots=True)
class PlayerDailySnapshotRow(RowBase):
    __table__ = "player_daily_snapshot"

    player_id: str
    date: str
    team_id: str | None
    role: str | None
    display_name: str | None
    champion_pool_json: str
    games_played: int
    # Role-specific pillar structure: 3-pillar for top/jng/mid/bot, 4-pillar for sup.
    pillars_json: str | None = None
    # Draft radar axes, within-role percentile 0-100; the ``_n`` sample counts gate
    # display confidence (consistency_n is the pillar-list length, others count picks).
    pool_depth: float | None = None
    blind_quality: float | None = None
    counter_quality: float | None = None
    counter_rate: float | None = None
    consistency: float | None = None
    overall_quality: float | None = None
    pool_depth_n: int | None = None
    blind_quality_n: int | None = None
    counter_quality_n: int | None = None
    counter_rate_n: int | None = None
    consistency_n: int | None = None


@dataclass(frozen=True, slots=True)
class TeamPowerDisplayRow(RowBase):
    __table__ = "power_rating_snapshots"

    date: str
    team_id: str
    display_name: str | None
    region: str | None
    rating: float
    global_rank: int
