"""Read the power-curve cache: the fit-time input behind the frozen per-role power table."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping

_REPO_ROOT = Path(__file__).resolve().parents[4]

#: The engine-simulated grid the fit reads; written by
#: scripts/research/scaling/extract_power_curves.py. Serving never opens it.
DEFAULT_POWER_CACHE = _REPO_ROOT / "data" / "power_curves.db"

#: The breakpoints a cell carries, k = 1..4 legendary slots; the grid's row axis.
BREAKPOINTS = (1, 2, 3, 4)

#: The eleven own-perspective power channels, in column order, each read against one body of
#: `data/power_curve_targets.json`.
CHANNELS: tuple[str, ...] = (
    "damage_carry",
    "damage_mage",
    "damage_physical",
    "damage_magic",
    "damage_true",
    "armor_faced",
    "cc",
    "sustain",
    "support",
    "ehp_physical",
    "ehp_magic",
)

#: The five rows a breakpoint needs, keyed ``(target, direction)``. The cache also holds
#: ``("tank", "in")``, plus mr_faced, damage_taken and death_time: provenance, not channels.
REQUIRED_ROWS = (
    ("carry", "out"),
    ("mage", "out"),
    ("tank", "out"),
    ("carry", "in"),
    ("mage", "in"),
)

_SELECT = """
SELECT role, champion, item_slots, target, direction,
       damage, damage_physical, damage_magic, damage_true, armor_faced,
       cc_seconds, healing_received, shield_received, support_value,
       own_armor, own_mr, max_health
FROM power_cells WHERE error IS NULL
"""


def channel_values(rows: Mapping[tuple[str, str], sqlite3.Row]) -> list[float]:
    """One breakpoint's ``CHANNELS``, in column order, from its five engine rows.

    Args:
        rows: the ``REQUIRED_ROWS`` of one (role, champion, breakpoint). An ``out`` row is
            the champion acting on a silent body, an ``in`` row the body acting on a
            silent champion, so a resist reading belongs to whoever takes the damage.

    Returns:
        Output against each body, the tank-body damage split, the armor that kit leaves the
        tank holding, three body-invariant utility means, and the two effective-health pools.
    """
    outs = [rows[key] for key in REQUIRED_ROWS if key[1] == "out"]
    tank, carry_in, mage_in = rows["tank", "out"], rows["carry", "in"], rows["mage", "in"]
    return [
        rows["carry", "out"]["damage"],
        rows["mage", "out"]["damage"],
        tank["damage_physical"],
        tank["damage_magic"],
        tank["damage_true"],
        tank["armor_faced"],
        fmean(row["cc_seconds"] for row in outs),
        fmean(row["healing_received"] + row["shield_received"] for row in outs),
        fmean(row["support_value"] for row in outs),
        carry_in["max_health"] * (1.0 + carry_in["own_armor"] / 100.0),
        mage_in["max_health"] * (1.0 + mage_in["own_mr"] / 100.0),
    ]


class PowerGrid(NamedTuple):
    """The whole fit-time read: every cell's curve, and which of them are extrapolations."""

    curves: dict[tuple[str, str], np.ndarray]
    borrowed: frozenset[tuple[str, str]]


def load_power_grid(cache_path: Path) -> PowerGrid:
    """``(role, champion)`` → ``[len(BREAKPOINTS), len(CHANNELS)]``, row i the ``k=i+1`` curve.

    Args:
        cache_path: the power-curve cache; opened read-only, errored rows dropped.

    Returns:
        Whole cells only, since a partial curve has no shape to occupy and a zero standing
        in for an unrun row would read as a champion that does nothing. ``borrowed`` names
        the cells run on another role's build, which a role with no scraped build reads.
    """
    con = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        points: dict[tuple[str, str, int], dict[tuple[str, str], sqlite3.Row]] = {}
        for row in con.execute(_SELECT):
            key = (row["role"], row["champion"], row["item_slots"])
            points.setdefault(key, {})[row["target"], row["direction"]] = row
        # Tuples, not the connection's Row factory: a Row never compares equal to a key.
        borrowed = frozenset(
            (row["role"], row["champion"])
            for row in con.execute(
                "SELECT DISTINCT role, champion FROM power_cells WHERE borrowed = 1"
            )
        )
    finally:
        con.close()
    whole = {key: rows for key, rows in points.items() if set(rows) >= set(REQUIRED_ROWS)}
    curves = {}
    for role, champion in {(role, champion) for role, champion, _ in whole}:
        keys = [(role, champion, k) for k in BREAKPOINTS]
        if all(key in whole for key in keys):
            curves[(role, champion)] = np.array([channel_values(whole[key]) for key in keys])
    return PowerGrid(curves, borrowed & set(curves))
