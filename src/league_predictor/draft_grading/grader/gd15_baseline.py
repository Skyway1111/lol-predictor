"""THE player gd15 baseline the lane head and the dense team-gd15 label both residualize by."""

from __future__ import annotations

import math

from league_predictor.decay import half_life_rate

#: Player trailing-gd15 baseline constants — the team-form class (lane skill and weak-side
#: assignment drift slowly, like team strength; the V3 design doc's D3).
LANE_BASELINE_HALF_LIFE_DAYS = 90.0
LANE_BASELINE_SHRINKAGE_K = 10.0


class PlayerGd15Baseline:
    """Decayed trailing gold diff at 15 per player, read strictly before a game.

    Attributes:
        state: ``player_id -> (last ordinal, decayed gd15 sum, decayed game count)``.
    """

    def __init__(
        self,
        half_life_days: float = LANE_BASELINE_HALF_LIFE_DAYS,
        k: float = LANE_BASELINE_SHRINKAGE_K,
    ):
        self._decay_rate = half_life_rate(half_life_days)
        self._k = k
        self.state: dict[str, tuple[int, float, float]] = {}

    def read(self, player_id: str, ordinal: int) -> float:
        """The player's shrunk decayed gd15 over the lanes folded in so far; 0.0 with none."""
        last_ord, sw_gd, sw_n = self.state.get(player_id, (ordinal, 0.0, 0.0))
        factor = math.exp(-self._decay_rate * (ordinal - last_ord)) if sw_n > 0.0 else 1.0
        return (sw_gd * factor) / (sw_n * factor + self._k)

    def advance(self, player_id: str, ordinal: int, gd15: float) -> None:
        """Fold one played lane in; a caller advances only after reading the whole game."""
        last_ord, sw_gd, sw_n = self.state.get(player_id, (ordinal, 0.0, 0.0))
        if sw_n > 0.0:
            factor = math.exp(-self._decay_rate * (ordinal - last_ord))
            sw_gd, sw_n = sw_gd * factor, sw_n * factor
        self.state[player_id] = (ordinal, sw_gd + gd15, sw_n + 1.0)
