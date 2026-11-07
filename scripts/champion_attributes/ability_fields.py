"""Ability feature values parsed from the wiki's structured fields and leveling tables."""

# Build-time contract: extraction thresholds are fit against these labels' exact
# matching behavior, so editing a rule requires regenerating the artifact.

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable


def parse_numeric_string(value: str | float | None) -> float:
    """Parse '800', '800 / Up to 1340', 'Global' into a float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("varied", "n/a", "none"):
        return 0.0
    if s.lower() == "global":
        return 25000.0
    match = re.search(r"\d+(?:\.\d+)?", s)
    return float(match.group()) if match else 0.0


#: A leveling array this long is a level bracket, never the rank axis, and reads at the
#: reference level 18. The calculator's own `champions/slot_extract._axis_index` rule.
_LEVEL_AXIS_VALUES = 18


def axis_index(values: list[float]) -> int:
    """The index a leveling array is read at: its max rank, or the reference level."""
    return min(_LEVEL_AXIS_VALUES, len(values)) - 1


def max_rank_value(modifiers: list[dict]) -> float:
    """modifiers[0] at its own axis index — max rank, or the reference level."""
    if not modifiers:
        return 0.0
    values = modifiers[0].get("values", [])
    return float(values[axis_index(values)]) if values else 0.0


def normalize_spell_effects(value: str | None) -> str:
    """Collapse spellEffects variants to canonical categories."""
    if not value:
        return "none"
    lowered = value.strip().lower()
    if lowered in ("aoe", "area of effect", "spellaoe"):
        return "aoe"
    if lowered in ("aoedot", "dot", "periodic"):
        return "dot"
    if lowered in ("single", "single target"):
        return "single"
    if lowered in ("proc", "basic", "attack", "default"):
        return "proc"
    return "special"


def extract_leveling_entries(
    effects: Iterable[dict],
) -> list[tuple[str, float, str]]:
    """Parse effects[].leveling[] into (attribute_label, max_rank_value, unit)."""
    entries: list[tuple[str, float, str]] = []
    for effect in effects:
        for leveling in effect.get("leveling") or []:
            attr = str(leveling.get("attribute", ""))
            for mod in leveling.get("modifiers") or []:
                values = mod.get("values", [])
                units = mod.get("units", [])
                if not values:
                    continue
                index = axis_index(values)
                unit = str(units[index]) if index < len(units) else ""
                entries.append((attr, float(values[index]), unit))
    return entries


def extract_scaling_ratios(  # noqa: C901 - the dispatch it enumerates: one branch per unit spelling
    entries: Iterable[tuple[str, float, str]],
) -> dict[str, float]:
    """Offensive scaling ratios from damage-labeled leveling entries only (schema-v4 polarity)."""
    result = {
        "scaling_ad": 0.0,
        "scaling_ap": 0.0,
        "scaling_max_hp": 0.0,
        "scaling_missing_hp": 0.0,
        "scaling_armor": 0.0,
        "scaling_mr": 0.0,
        "scaling_bonus_hp": 0.0,
    }

    def book(key: str, val: float) -> None:
        result[key] = max(result[key], val)

    for attr, val, unit in entries:
        if "damage" not in attr.lower():
            continue
        u = unit.lower()
        if "% ad" in u or "% bonus ad" in u or "% total ad" in u:
            book("scaling_ad", val)
        elif "% ap" in u:
            book("scaling_ap", val)
        elif "target" in u:
            # "% of target's maximum/missing health" — the anti-tank ratios.
            if "missing health" in u:
                book("scaling_missing_hp", val)
            elif "health" in u:
                book("scaling_max_hp", val)
        elif "missing health" in u:
            book("scaling_missing_hp", val)
        elif "armor" in u:
            book("scaling_armor", val)
        elif "magic resist" in u:
            book("scaling_mr", val)
        elif "bonus health" in u:
            book("scaling_bonus_hp", val)
        elif "maximum health" in u or "max health" in u:
            # Bare health unit with no owner named: target implied (v3 reading).
            book("scaling_max_hp", val)
    return result


def extract_defense_grants(  # noqa: C901 - the dispatch it enumerates: one branch per grant label
    entries: Iterable[tuple[str, float, str]],
) -> dict[str, float]:
    """Schema-v4 defensive grant dims from self-buff leveling labels; % units amplify own stats."""
    result = {
        "armor_gain_flat": 0.0,
        "armor_gain_pct": 0.0,
        "mr_gain_flat": 0.0,
        "mr_gain_pct": 0.0,
        "hp_gain_flat": 0.0,
        "damage_reduction_pct": 0.0,
        "shield_hp_pct": 0.0,
        "armor_shred_pct": 0.0,
    }

    def book(key: str, val: float) -> None:
        result[key] = max(result[key], val)

    for attr, val, unit in entries:
        a, u = attr.lower(), unit.lower()
        if "damage reduction" in a:
            # 'Attack Damage Reduction' is an enemy AD debuff (Tryndamere W),
            # not incoming-damage reduction; 'Flat …' is a different mechanic.
            if "flat" not in a and "attack damage" not in a:
                book("damage_reduction_pct", val)
            continue
        if "damage" in a:
            continue  # offensive entry — extract_scaling_ratios' business
        if "shield" in a and "health" in u:
            book("shield_hp_pct", val)
        elif "armor reduction" in a:
            book("armor_shred_pct", val)
        elif "bonus armor" in a:
            book("armor_gain_pct" if "%" in u else "armor_gain_flat", val)
        elif "magic resist" in a:
            book("mr_gain_pct" if "%" in u else "mr_gain_flat", val)
        elif "bonus health" in a and not u:
            book("hp_gain_flat", val)
    return result


_INFLATED_DAMAGE_PREFIXES = (
    "total ",  # multi-hit sums (Viktor R "Total Magic Damage" = 1120)
    "increased ",  # conditional bonuses (Jayce Q "Increased Damage" = 399)
    "maximum ",  # distance-scaled max (Jinx R "Maximum Physical Damage" = 500)
    "minimum ",  # distance-scaled min (Jinx R "Minimum Physical Damage" = 50)
)

# Substrings in the attribute label that indicate PvE-only values.
# These are never valid as champion-relevant base_damage.
_PVE_DAMAGE_SUBSTRINGS = (
    "capped",  # "Capped Monster Damage" (Jayce E = 700)
    "monster",  # "Monster Damage Cap Per Tick" (Ornn W = 100)
    "minion",  # "Minimum/Minion Damage Per Tick" (Ornn W = 56)
)


def _is_inflated_damage_label(attr: str) -> bool:
    """True for total/conditional/extreme labels, which distort base_damage vs the single hit."""
    a = attr.lower()
    return any(a.startswith(p) for p in _INFLATED_DAMAGE_PREFIXES)


def _is_never_use_damage_label(attr: str) -> bool:
    """True for PvE-only labels, never usable as base_damage even as fallback."""
    a = attr.lower()
    # "Non-Minion" and "Non-Champion" are champion-relevant (damage to non-minions).
    if "non-minion" in a or "non-champion" in a:
        return False
    return any(s in a for s in _PVE_DAMAGE_SUBSTRINGS)


def extract_base_damage(entries: Iterable[tuple[str, float, str]]) -> float:
    """Primary flat damage at max rank: direct labels first, inflated labels only as fallback."""
    best_direct = 0.0
    best_inflated = 0.0
    for attr, val, unit in entries:
        if "damage" in attr.lower() and unit == "":
            if _is_never_use_damage_label(attr):
                continue
            if _is_inflated_damage_label(attr):
                best_inflated = max(best_inflated, val)
            else:
                best_direct = max(best_direct, val)
    return best_direct if best_direct > 0.0 else best_inflated


#: Word-bounded heal labels: 'Bonus Health' must never book as a heal ('health'
#: has no ``\bheal\b`` boundary); 'Health Restored' counts via the restore stem.
_HEAL_LABEL_PATTERN = re.compile(r"\bheal(?:s|ing)?\b|\brestor")


def extract_heal_amount(entries: Iterable[tuple[str, float, str]]) -> float:
    """Find flat heal value at max rank ('… Reduction' riders never count)."""
    best = 0.0
    for attr, val, unit in entries:
        a = attr.lower()
        if unit == "" and "reduction" not in a and _HEAL_LABEL_PATTERN.search(a):
            best = max(best, val)
    return best


def extract_shield_amount(entries: Iterable[tuple[str, float, str]]) -> float:
    """Find flat shield value at max rank."""
    best = 0.0
    for attr, val, unit in entries:
        if "shield" in attr.lower() and unit == "":
            best = max(best, val)
    return best
