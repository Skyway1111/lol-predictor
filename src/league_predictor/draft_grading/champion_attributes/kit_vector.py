"""The dense named kit vector: one scalar per kit family, read from the champion store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.draft_grading.champion_attributes.ability_schema import ABILITY_SLOTS
from league_predictor.draft_grading.champion_attributes.ability_store import z_score_columns

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from league_predictor.draft_grading.champion_attributes.ability_store import (
        AbilityFeatureStore,
    )
    from league_predictor.draft_grading.grader.embeddings import ChampionIndex

    KitRead = Callable[[Mapping[str, float]], float]

#: THE family-meta key whose truth means the member carries a kit matrix. A fit that
#: freezes one stamps this key; `load_knowledge_tables` reads the member by it.
KIT_MEMBER_FLAG = "kit_features"

#: The level the vendored contract prices every slot at (`reviewed_kit.ReviewedKit.level`),
#: and the one level whose growth ramp `(n-1)(0.7025 + 0.0175(n-1))` is exactly 17.
REFERENCE_LEVEL = 18

#: The export's "global" target range, stated by nine champions. It is 4.5x the longest
#: real reach (5500), so it is a flag, never a distance.
GLOBAL_RANGE = 25_000.0


@dataclass(frozen=True)
class KitFamily:
    """One named kit scalar: how a store row reads it, and whether the column is z-scored.

    Attributes:
        standardise: False carries the column as read, for a 0/1 flag and for a family
            so rare that its z-score would be a lone multi-sigma spike.
    """

    read: KitRead
    standardise: bool = True


def _slots(*keys: str) -> KitRead:
    """The named per-slot dims summed over P/Q/W/E/R: one cast of each."""
    return lambda row: sum(row.get(f"{slot}_{key}", 0.0) for slot in ABILITY_SLOTS for key in keys)


def _typed_damage(damage_type: str) -> KitRead:
    """The cast cycle's damage of one class: each slot's base damage under its own one-hot."""
    return lambda row: sum(
        row.get(f"{slot}_base_damage", 0.0) * row.get(f"{slot}_damage_type_{damage_type}", 0.0)
        for slot in ABILITY_SLOTS
    )


def _champion(key: str) -> KitRead:
    """A dim the store already carries once per champion."""
    return lambda row: row.get(key, 0.0)


def _states(key: str) -> KitRead:
    """1.0 where the champion states the dim at all, else 0.0."""
    return lambda row: float(row.get(key, 0.0) > 0.0)


def _grown(base: str, per_level: str) -> KitRead:
    """A base stat at `REFERENCE_LEVEL`."""
    return lambda row: row.get(base, 0.0) + (REFERENCE_LEVEL - 1) * row.get(per_level, 0.0)


def _ranges(row: Mapping[str, float]) -> list[float]:
    """Every slot's stated target range."""
    return [row.get(f"{slot}_target_range", 0.0) for slot in ABILITY_SLOTS]


def _capped_range(row: Mapping[str, float]) -> float:
    """The kit's longest real reach, with the global sentinel left to its own flag."""
    return max((r for r in _ranges(row) if r < GLOBAL_RANGE), default=0.0)


def _has_global_range(row: Mapping[str, float]) -> float:
    """1.0 where a slot reaches the whole map."""
    return float(max(_ranges(row), default=0.0) >= GLOBAL_RANGE)


def _attack_speed(row: Mapping[str, float]) -> float:
    """Attack speed at `REFERENCE_LEVEL`; `as_per_level` is a percent of the base."""
    growth = (REFERENCE_LEVEL - 1) * row.get("as_per_level", 0.0) / 100.0
    return row.get("as_base", 0.0) * (1.0 + growth)


#: The kit's named families aggregate related mechanics from the stored dimensions,
#: one family per mechanic so a bilinear block can pair offence with defence by name.
KIT_FAMILIES: dict[str, KitFamily] = {
    # What one cast of each slot deals, by class and by the scaling it rides.
    "physical_damage": KitFamily(_typed_damage("physical")),
    "magic_damage": KitFamily(_typed_damage("magic")),
    "true_damage": KitFamily(_typed_damage("true")),
    "pct_hp_damage": KitFamily(_slots("scaling_max_hp", "scaling_missing_hp")),
    "ad_ratio": KitFamily(_slots("scaling_ad")),
    "ap_ratio": KitFamily(_slots("scaling_ap")),
    # The realised split LeagueOfGraphs observes: the one reading of the built champion.
    "physical_damage_share": KitFamily(_champion("physical_damage_share")),
    "magic_damage_share": KitFamily(_champion("magic_damage_share")),
    "true_damage_share": KitFamily(_champion("true_damage_share")),
    # Control: the seconds, then the casts each of the three control shapes costs.
    "cc_hard_seconds": KitFamily(_slots("cc_hard_seconds")),
    "cc_slow_seconds": KitFamily(_slots("cc_slow_duration")),
    "cc_slow_pct": KitFamily(_slots("cc_slow_pct")),
    "cc_lockdown_casts": KitFamily(_slots("cc_type_stun", "cc_type_root", "cc_type_suppress")),
    "cc_displacement_casts": KitFamily(
        _slots("cc_type_knockup", "cc_type_knockback", "cc_type_pull")
    ),
    "cc_disable_casts": KitFamily(
        _slots(
            "cc_type_silence",
            "cc_type_charm",
            "cc_type_taunt",
            "cc_type_fear",
            "cc_type_sleep",
            "cc_type_polymorph",
            "cc_type_ground",
            "cc_type_disarm",
        )
    ),
    # Self-repair.
    "heal": KitFamily(_slots("heal_amount")),
    "shield": KitFamily(_slots("shield_amount")),
    "shield_hp_pct": KitFamily(_slots("shield_hp_pct")),
    # One to four kits state each, so a flag carries who and the magnitude stays in the
    # contract's percent. No contract states the flat pair, and zeros are not a family.
    "has_armor_pen": KitFamily(_states("armor_pen_pct"), standardise=False),
    "armor_pen_pct": KitFamily(_champion("armor_pen_pct"), standardise=False),
    "has_magic_pen": KitFamily(_states("magic_pen_pct"), standardise=False),
    "magic_pen_pct": KitFamily(_champion("magic_pen_pct"), standardise=False),
    "has_lethality": KitFamily(_states("lethality"), standardise=False),
    "lethality": KitFamily(_champion("lethality"), standardise=False),
    "armor_shred_pct": KitFamily(_slots("armor_shred_pct")),
    # The resists and mitigation the kit grants itself.
    "armor_gain_flat": KitFamily(_slots("armor_gain_flat")),
    "armor_gain_pct": KitFamily(_slots("armor_gain_pct")),
    "mr_gain_flat": KitFamily(_slots("mr_gain_flat")),
    "mr_gain_pct": KitFamily(_slots("mr_gain_pct")),
    "hp_gain_flat": KitFamily(_slots("hp_gain_flat")),
    "damage_reduction_pct": KitFamily(_slots("damage_reduction_pct")),
    # Mobility and reach.
    "dash_casts": KitFamily(_slots("has_dash")),
    "ms_boost_casts": KitFamily(_slots("has_ms_boost")),
    "move_speed": KitFamily(_champion("ms_base")),
    "attack_range": KitFamily(_champion("attack_range")),
    "ability_range": KitFamily(_capped_range),
    "has_global_range": KitFamily(_has_global_range, standardise=False),
    "is_melee": KitFamily(_champion("is_melee"), standardise=False),
    # Base stats at the reference level: what the penetration and %HP families act on.
    "health": KitFamily(_grown("hp_base", "hp_per_level")),
    "armor": KitFamily(_grown("armor_base", "armor_per_level")),
    "magic_resist": KitFamily(_grown("mr_base", "mr_per_level")),
    "attack_damage": KitFamily(_grown("ad_base", "ad_per_level")),
    "attack_speed": KitFamily(_attack_speed),
}

KIT_FAMILY_NAMES: tuple[str, ...] = tuple(KIT_FAMILIES)

#: The z-scored columns as a mask; the rest are carried as read.
_STANDARDISED = np.array([family.standardise for family in KIT_FAMILIES.values()])


def kit_matrix_shape(n_rows: int) -> tuple[int, int]:
    """The frozen kit table's shape over a champion universe of ``n_rows`` (UNKNOWN included)."""
    return (n_rows, len(KIT_FAMILIES))


def kit_vector(store: AbilityFeatureStore, champion: str) -> np.ndarray:
    """One champion's raw ``KIT_FAMILY_NAMES``-wide reading; `kit_matrix` standardises it."""
    resolved = store.resolve_name(champion)
    if resolved is None:
        raise KeyError(
            f"{champion!r} is not a champion the store can spell, and its all-zero row "
            "would pass a claim rather than fail it"
        )
    row = store.get(resolved)
    return np.array([family.read(row) for family in KIT_FAMILIES.values()], dtype=np.float64)


def kit_matrix(index: ChampionIndex, store: AbilityFeatureStore) -> np.ndarray:
    """The frozen kit table: z-scored over the index but for the flags and the rare magnitudes.

    Returns:
        ``(len(index), K)`` float32. UNKNOWN, and any index name the store cannot
        resolve, read all zeros: the corpus mean on a z-scored family, absent on the rest.
    """
    raw = np.zeros(kit_matrix_shape(len(index)), dtype=np.float64)
    known: list[int] = []
    for i, name in enumerate(index.known_champions):
        if store.resolve_name(name) is not None:
            raw[i] = kit_vector(store, name)
            known.append(i)

    table = np.zeros_like(raw)
    table[known] = z_score_columns(raw[known])[0]
    table[:, ~_STANDARDISED] = raw[:, ~_STANDARDISED]
    return table.astype(np.float32)
