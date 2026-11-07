"""THE Impact linear predictor: the region term, the fitted values, and ``game_eta``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

_DEFAULT_ROLE = "mid"
#: The locked lineup width (``impact_corpus`` rejects wider sides): the denominator both
#: the game's region term and the published per-player share divide by, so they always agree.
ROSTER_SIZE = 5


@dataclass(frozen=True, slots=True)
class RegionLevels:
    """The region term: one level per player, priced once per game at team level.

    Attributes:
        level: player -> the log-odds strength of the region their latest domestic league
            normalizes to, or their league tier's anchor when it normalizes to none.
        default_level: the level of a player the resolution never saw — the DEFAULT tier's
            anchor, so an unknown league is never read as an average major region.
    """

    level: dict[str, float]
    default_level: float

    def of(self, player_id: str) -> float:
        """The player's region level in log-odds."""
        return self.level.get(player_id, self.default_level)

    def gap(self, roster_a: Iterable[str], roster_b: Iterable[str]) -> float:
        """The two sides' level difference; a short lineup contributes proportionally less."""
        return self._side_level(roster_a) - self._side_level(roster_b)

    def share(self, player_id: str) -> float:
        """The player's own contribution to their side's level — the published region term."""
        return self.of(player_id) / ROSTER_SIZE

    def _side_level(self, roster: Iterable[str]) -> float:
        """The side's level over the SAME denominator ``share`` uses, never the lineup width."""
        return sum(self.of(player_id) for player_id in roster) / ROSTER_SIZE


@dataclass(frozen=True, slots=True)
class ImpactValues:
    """The fitted linear predictor: everything :func:`game_eta` reads, and nothing else.

    Attributes:
        player_value: ``v_p = gamma * prior_p + delta_p`` per named player.
        replacement_value: ``v`` for a sub-threshold player, by role; empty means 0.0.
        player_role: role of every player the qualification window saw.
        region: the recipe's region term, which derives its gap from the two rosters.
    """

    player_value: dict[str, float]
    replacement_value: dict[str, float]
    player_role: dict[str, str]
    region: RegionLevels
    alpha: float
    beta: float


def _roster_value(values: ImpactValues, roster: Iterable[str]) -> float:
    """Summed ``v_p`` over a roster; an unnamed player takes their role's replacement value."""
    total = 0.0
    for player_id in roster:
        value = values.player_value.get(player_id)
        if value is None:
            # A player the qualification window never saw has no role either, so a post-cutoff
            # debutant is priced at the mid replacement value whatever role they actually play.
            role = values.player_role.get(player_id, _DEFAULT_ROLE)
            value = values.replacement_value.get(role, 0.0)
        total += value
    return total


def game_eta(
    values: ImpactValues,
    roster_a: Iterable[str],
    roster_b: Iterable[str],
    blue_sign: int,
) -> float:
    """THE Impact log-odds of ``roster_a`` beating ``roster_b`` — the one definition.

    Returns:
        ``sum v(a) - sum v(b) + alpha * region.gap + beta * blue_sign``. The fit's alpha
        column is the same ``region.gap`` call, and the gap is derived from the rosters
        rather than passed in, so no caller can price a game differently than the fit did.
    """
    roster_a = tuple(roster_a)
    roster_b = tuple(roster_b)
    return (
        _roster_value(values, roster_a)
        - _roster_value(values, roster_b)
        + values.alpha * values.region.gap(roster_a, roster_b)
        + values.beta * blue_sign
    )
