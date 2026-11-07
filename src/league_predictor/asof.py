"""Dependency-leaf query point: a prediction's place in the facts order plus its match context."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_ISO_DATE_LEN = 10


@dataclass(frozen=True, slots=True)
class MatchContext:
    """The match facts of a game the facts DB does not hold: format, stage, maps won so far."""

    format: str = "Bo1"  # 'Bo1' / 'Bo3' / 'Bo5'
    stage: str = "regular"
    wins: tuple[tuple[str, int], ...] = ()  # (team_id, maps won in this series), sorted

    @classmethod
    def of(cls, format: str, stage: str, wins: Mapping[str, int] | None = None) -> MatchContext:  # noqa: A002 - the match format field
        """Build a context from a ``team_id -> maps won`` mapping."""
        return cls(format, stage, tuple(sorted((wins or {}).items())))

    def wins_for(self, team_id: str) -> int:
        """Maps ``team_id`` has won in this series so far; 0 for a team not listed."""
        return dict(self.wins).get(team_id, 0)


@dataclass(frozen=True, slots=True)
class AsOf:
    """The game being predicted, as a walk-forward query.

    Attributes:
        ts: its series' ISO-8601 start, or a bare ``YYYY-MM-DD`` for a query at the start of
            a day. The query sees every game whose series started strictly before ``ts``,
            plus the earlier maps of its own series; two series that start at one instant
            never see each other, so the rule is leak-free under any tie order.
        series_id: the query's own series; empty for a day or an instant.
        game_no: the query's map number within its series; its earlier maps are visible.
        context: set only for a game the facts do not hold (a scheduled or hypothetical
            match); a facts game's context derives from its rows. Part of equality, never
            of ordering.
    """

    ts: str
    series_id: str = ""
    game_no: int = 0
    context: MatchContext | None = field(default=None)

    @classmethod
    def of(cls, value: AsOf | date | datetime | str) -> AsOf:
        """Normalize a public-entry argument: an AsOf, its key, an instant, or a day's start."""
        if isinstance(value, AsOf):
            return value
        if isinstance(value, str):
            return cls.parse(value)
        if isinstance(value, datetime):
            utc = value if value.tzinfo is None else value.astimezone(UTC).replace(tzinfo=None)
            return cls(utc.strftime("%Y-%m-%dT%H:%M:%S") + "+00:00")
        return cls(value.isoformat())

    @classmethod
    def game(cls, ts: str, series_id: str, game_no: int) -> AsOf:
        """The query for a facts game: its series' start, its series, its game number."""
        return cls(str(ts), str(series_id), int(game_no))

    @classmethod
    def parse(cls, key: str) -> AsOf:
        """Invert :attr:`key` (a key never carries a context)."""
        ts, _, rest = key.partition("|")
        if not rest:
            return cls(ts)
        series_id, _, game_no = rest.partition("|")
        return cls(ts, series_id, int(game_no))

    @property
    def key(self) -> str:
        """The cache key: ``ts`` alone for a day, else ``ts|series_id|NN``; sorts like the query."""
        if not self.series_id:
            return self.ts
        return f"{self.ts}|{self.series_id}|{self.game_no:02d}"

    @property
    def facts_game(self) -> bool:
        """Whether the query names a game the facts hold: a position without a scheduled context."""
        return bool(self.series_id) and self.context is None

    @property
    def day(self) -> date:
        """The calendar day of the query, for the day-strict readers."""
        return date.fromisoformat(self.ts[:_ISO_DATE_LEN])

    def with_context(self, context: MatchContext | None) -> AsOf:
        """The same position under an explicit match context."""
        return AsOf(self.ts, self.series_id, self.game_no, context)

    # --- ordering is the facts order (ts, series_id, game_no); context stays out of it ---

    @property
    def position(self) -> tuple[str, str, int]:
        """The sort key in the canonical facts order."""
        return (self.ts, self.series_id, self.game_no)

    def __lt__(self, other: AsOf) -> bool:
        return self.position < other.position

    def __le__(self, other: AsOf) -> bool:
        return self.position <= other.position

    def __gt__(self, other: AsOf) -> bool:
        return self.position > other.position

    def __ge__(self, other: AsOf) -> bool:
        return self.position >= other.position
