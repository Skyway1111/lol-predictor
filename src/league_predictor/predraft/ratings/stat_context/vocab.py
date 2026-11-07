"""The grid's frozen vocabulary and dials, its two containers, and the weight-span guard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from league_predictor.vocab.roles import ROLE_ORDER

if TYPE_CHECKING:
    import numpy as np

#: Stat vocabulary of the context grid — the union of every role's pillar stats.
STATS = (
    "golddiffat15",
    "csdiffat15",
    "xpdiffat15",
    "cspm",
    "dpm",
    "killparticipation",
    "kda",
    "firstblood",
    "vspm",
    "earnedgoldshare",
)
#: Pillar vocabulary — the union of every role's pillar names.
PILLARS = ("laning", "combat", "vision", "lane", "playmaking", "utility")

#: v6 recipe constants (a change to any is a model change gated by a A/B).
GRID_EPOCH = date(2017, 1, 1)
GRID_STEP_DAYS = 30
GRID_WINDOW_DAYS = 180
PRIOR_HALF_LIFE_DAYS = 60.0
PRIOR_SHRINKAGE_K = 20.0
_Z_WINSOR = 4.0
_MIN_CELL_SUPPORT = 30
_AT10_SCALE = 1.33
#: Numerically-zero floors: a variance or weight sum below _EPS carries no signal, and a
#: per-cell sigma below _MIN_SIGMA would blow up every z it divides.
_EPS = 1e-9
_MIN_SIGMA = 1e-6
#: A half-life short enough to need this many doublings overflows the scaled-weight float64.
_MAX_WEIGHT_DOUBLINGS = 900
#: v7 recipe constant: per-(champion, role, stat) mean-residual offsets shrink by n/(n+K).
#: v8: the lane-BT margin reads ONLY the true lane diffs, never cspm.
_LANE_BT_STATS = {
    "laning": ("golddiffat15", "csdiffat15", "xpdiffat15"),
    "lane": ("golddiffat15", "xpdiffat15"),
}
#: The laning diffs are the only stats carrying opponent / side / blind adjustments.
_ADJUSTED_STATS = ("golddiffat15", "csdiffat15", "xpdiffat15")

_STAT_INDEX = {name: i for i, name in enumerate(STATS)}
_PILLAR_INDEX = {name: i for i, name in enumerate(PILLARS)}
_ROLE_INDEX = {role: i for i, role in enumerate(ROLE_ORDER)}


def game_kda(kills: float, deaths: float, assists: float) -> float:
    """Per-game KDA as ``(k + a) / (d + 1)`` — bounded on zero-death games."""
    return (kills + assists) / (deaths + 1.0)


@dataclass(frozen=True, slots=True)
class ContextZ:
    """Per-appearance context-adjusted z-scores, date-sorted, strictly before ``through``.

    Attributes:
        stat_z: ``[n, len(STATS)]``; NaN marks a stat absent for that appearance. Each z
            is OUTCOME-CONDITIoneD by construction (the game's own result, side, blind,
            champion, and opponent enter its own expectation), so a same-game read of z
            carries the label — any predictive feature must come from the strictly
            pre-game aggregates (``walk_forward_pillar_aggregates``), never from a
            game's own rows.
        pillar_z: ``[n, len(PILLARS)]``; the role's within-pillar weighted mean of present z.
        appearance_games: game index per appearance, for PRE-GAME team-sum feature builds.
        appearance_team1: 1 on the series' canonical ``team_a``, 0 on ``team_b``, -1 neither.
        game_team1_won: per game index, 1/0 for a canonical winner, -1 otherwise.
        champion_offsets / lane_bt: the build's recipe knobs, carried as provenance.
    """

    player_ids: tuple[str, ...]
    appearance_players: np.ndarray
    appearance_dates: np.ndarray
    appearance_roles: np.ndarray
    stat_z: np.ndarray
    pillar_z: np.ndarray
    through: date
    appearance_games: np.ndarray | None = None
    appearance_team1: np.ndarray | None = None
    game_dates: np.ndarray | None = None
    game_team1_won: np.ndarray | None = None
    lane_bt: bool = False


@dataclass(frozen=True, slots=True)
class PlayerPrior:
    """One player's recency-weighted, shrunk aggregate over their context-adjusted games."""

    pillar_z: dict[str, float]
    stat_z: dict[str, float]
    n_games: int


def _assert_scaled_weight_span(dates: np.ndarray, t0: int, half_life_days: float) -> None:
    """Refuse a half-life so short the scaled-weight trick would overflow float64."""
    doublings = (int(dates.max()) - t0) / half_life_days
    if doublings > _MAX_WEIGHT_DOUBLINGS:
        raise ValueError(
            f"half_life_days={half_life_days} spans {doublings:.0f} doublings over this "
            "corpus — the scaled-weight accumulator would overflow; use a longer half-life"
        )
