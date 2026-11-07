"""Engine-simulated per-role power curves: the frozen per-family table."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from league_predictor.draft_grading.champion_attributes.power_store import BREAKPOINTS, CHANNELS
from league_predictor.vocab.roles import ROLE_ORDER, ROLE_RANK

if TYPE_CHECKING:
    from league_predictor.draft_grading.champion_attributes.power_store import PowerGrid
    from league_predictor.draft_grading.grader.embeddings import ChampionIndex

#: THE family-meta key whose truth means the member carries a power table. A fit that
#: freezes one stamps this key; `load_knowledge_tables` reads the member by it.
POWER_MEMBER_FLAG = "power_features"


class PowerTables(NamedTuple):
    """A family's frozen power view: every cell's curve, and which of them are borrowed.

    Attributes:
        table: ``[champion, role, breakpoint, channel]`` float32, the champion index's
            order over ``ROLE_ORDER``, ``BREAKPOINTS`` and ``CHANNELS``.
        borrowed: ``[champion, role]`` bool, the cells run on another role's build.
    """

    table: np.ndarray
    borrowed: np.ndarray


def power_table_shape(n_rows: int) -> tuple[int, int, int, int]:
    """The frozen table's shape over a champion universe of ``n_rows`` (UNKNOWN included)."""
    return (n_rows, len(ROLE_ORDER), len(BREAKPOINTS), len(CHANNELS))


def build_power_tables(index: ChampionIndex, grid: PowerGrid) -> PowerTables:
    """Freeze a ``load_power_grid`` result into float32 ``[champion, role, breakpoint, channel]``.

    Returns:
        The table and the borrowed plane, which freeze together. A champion outside the
        grid, a role it holds no cell in, and UNKNOWN stay zero.
    """
    table = np.zeros(power_table_shape(len(index)), dtype=np.float32)
    borrowed = np.zeros(table.shape[:2], dtype=np.bool_)
    for (role, champion), curve in grid.curves.items():
        own = index.encode(champion)
        if own == index.UNKNOWN or role not in ROLE_RANK:
            continue
        table[own, ROLE_RANK[role]] = curve
        borrowed[own, ROLE_RANK[role]] = (role, champion) in grid.borrowed
    return PowerTables(table, borrowed)
