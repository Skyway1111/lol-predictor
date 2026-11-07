"""The 360-dim per-champion feature vector: its key layout, its assembly and its precedence."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from league_predictor.draft_grading.champion_attributes.ability_schema import (
    ABILITY_FEATURE_KEYS,
    ABILITY_SLOTS,
    CHAMPION_FEATURE_KEYS,
)
from scripts.champion_attributes.ability_fields import (
    extract_base_damage,
    extract_defense_grants,
    extract_heal_amount,
    extract_leveling_entries,
    extract_scaling_ratios,
    extract_shield_amount,
    max_rank_value,
    normalize_spell_effects,
    parse_numeric_string,
)
from scripts.champion_attributes.cc_labels import cc_from_labels
from scripts.champion_attributes.text_mining import (
    ability_effect_texts,
    extract_cc_from_text,
    extract_mechanic_flags,
    extract_slow_duration,
    extract_tank_text_grants,
)

#: The artifact's schema, stamped by `scripts/generate_abilities.py` and asserted at
#: load: a row of the previous schema would read every v6 key as zero.


# Multi-form merge: these keys take the min across forms (a lower cooldown/cost/
# cast time is the best-case form); every other feature key max-merges.
_MIN_MERGE_FEATURES = frozenset({"cooldown", "cost", "cast_time"})
assert set(ABILITY_FEATURE_KEYS) >= _MIN_MERGE_FEATURES


# --- Core extraction ---


def extract_ability_features(ability: Mapping[str, Any]) -> dict[str, float]:
    """Full feature vector for a single ability entry."""
    features: dict[str, float] = dict.fromkeys(ABILITY_FEATURE_KEYS, 0.0)

    # --- Structured fields ---
    targeting = str(ability.get("targeting") or "").lower()
    features["is_passive"] = 1.0 if "passive" in targeting else 0.0
    features["is_skillshot"] = 1.0 if "direction" in targeting else 0.0
    features["is_targeted"] = 1.0 if "unit" in targeting else 0.0

    damage_type = str(ability.get("damageType") or "").upper()
    features["damage_type_physical"] = 1.0 if damage_type == "PHYSICAL_DAMAGE" else 0.0
    features["damage_type_magic"] = 1.0 if damage_type == "MAGIC_DAMAGE" else 0.0
    features["damage_type_true"] = 1.0 if damage_type == "TRUE_DAMAGE" else 0.0

    spell_fx = normalize_spell_effects(ability.get("spellEffects"))
    features["is_aoe"] = 1.0 if spell_fx in ("aoe", "dot") else 0.0
    features["is_dot"] = 1.0 if spell_fx == "dot" else 0.0

    features["is_projectile"] = (
        1.0 if str(ability.get("projectile") or "").upper() == "TRUE" else 0.0
    )

    cd = ability.get("cooldown")
    if isinstance(cd, Mapping) and cd.get("modifiers"):
        features["cooldown"] = max_rank_value(cd["modifiers"])

    cost_obj = ability.get("cost")
    if isinstance(cost_obj, Mapping) and cost_obj.get("modifiers"):
        features["cost"] = max_rank_value(cost_obj["modifiers"])

    features["cast_time"] = parse_numeric_string(ability.get("castTime"))
    features["target_range"] = parse_numeric_string(ability.get("targetRange"))
    features["effect_radius"] = parse_numeric_string(ability.get("effectRadius"))
    features["width"] = parse_numeric_string(ability.get("width"))
    features["speed"] = parse_numeric_string(ability.get("speed"))

    # --- Leveling data ---
    effects = ability.get("effects") or []
    entries = extract_leveling_entries(effects)

    features["base_damage"] = extract_base_damage(entries)
    features["heal_amount"] = extract_heal_amount(entries)
    features["shield_amount"] = extract_shield_amount(entries)
    features.update(extract_scaling_ratios(entries))
    features.update(extract_defense_grants(entries))

    # --- Text mining ---
    texts = ability_effect_texts(ability)
    affects = ability.get("affects")
    features.update(extract_cc_from_text(texts))
    features["cc_slow_duration"] = extract_slow_duration(texts)

    mechanics = extract_mechanic_flags(texts)
    features.update(mechanics)
    for key, val in extract_tank_text_grants(texts).items():
        features[key] = max(features[key], val)
    if features["is_dot"] == 1.0:
        features["has_persistent_aoe"] = 1.0

    # Ally polarity rides the structured affects field, not the key names.
    # (spellshieldable has no target key — subsumed by projectile/skillshot; decided.)
    features["affects_allies"] = (
        1.0 if affects and re.search(r"\ball(?:y|ies)\b", str(affects).lower()) else 0.0
    )

    # Labels last: a control the wiki states outright outranks one read out of prose,
    # and max-merge lets each layer supply what the other cannot see.
    for key, val in cc_from_labels(entries).items():
        features[key] = max(features[key], val)

    return features


def merge_multi_form(forms: Sequence[dict[str, float]]) -> dict[str, float]:
    """Combine multiple ability forms: max damage/range, min cooldown, union flags."""
    if len(forms) == 1:
        return dict(forms[0])
    merged: dict[str, float] = {}
    for key in ABILITY_FEATURE_KEYS:
        values = [f[key] for f in forms]
        # min for cooldown/cost/cast_time (best-case form); max is the default
        # for every other key.
        if key in _MIN_MERGE_FEATURES:
            nonzero = [v for v in values if v > 0]
            merged[key] = min(nonzero) if nonzero else 0.0
        else:
            merged[key] = max(values)
    return merged


def extract_slot_features(abilities: list[dict]) -> dict[str, float]:
    """Extract features for an ability slot, handling multi-form."""
    if not abilities:
        return dict.fromkeys(ABILITY_FEATURE_KEYS, 0.0)
    forms = [extract_ability_features(ability) for ability in abilities]
    return merge_multi_form(forms)


def extract_champion_stats(champion: Mapping[str, Any]) -> dict[str, float]:
    """Level-1 base stats, per-level growth and resource type — machine-derived only."""
    stats = champion.get("stats") or {}

    def stat(name: str, field: str = "flat") -> float:
        return float((stats.get(name) or {}).get(field, 0))

    attack_type = str(champion.get("attackType") or "").upper()
    resource = str(champion.get("resource") or "").upper()
    abilities = champion.get("abilities") or {}
    is_transform = any(len(abilities.get(slot, [])) > 1 for slot in ABILITY_SLOTS)

    return {
        "hp_base": stat("health"),
        "hp_per_level": stat("health", "perLevel"),
        "armor_base": stat("armor"),
        "armor_per_level": stat("armor", "perLevel"),
        "mr_base": stat("magicResistance"),
        "mr_per_level": stat("magicResistance", "perLevel"),
        "ad_base": stat("attackDamage"),
        "ad_per_level": stat("attackDamage", "perLevel"),
        "as_base": stat("attackSpeed"),
        "as_per_level": stat("attackSpeed", "perLevel"),
        "ms_base": stat("movespeed"),
        "attack_range": stat("attackRange"),
        "is_melee": 1.0 if attack_type == "MELEE" else 0.0,
        "is_transform": 1.0 if is_transform else 0.0,
        "resource_mana": 1.0 if resource == "MANA" else 0.0,
        "resource_energy": 1.0 if resource == "ENERGY" else 0.0,
        "resource_none": 1.0 if resource not in ("MANA", "ENERGY") else 0.0,
    }


def extract_all_features(
    champion_key: str,
    champion: Mapping[str, Any],
    edge_cases: dict[str, list[str]] | None = None,
    damage_shares: tuple[float, float, float] | None = None,
) -> dict[str, float]:
    """Full champion feature vector: champion stats, damage shares, P/Q/W/E/R ability features.

    Args:
        damage_shares: (physical, magic, true); ``None`` leaves the three keys at 0.0.
    """
    # Seeded with every champion-level key, so the schema-v6 penetration dims state
    # prose's zero and `assemble` overwrites only where a contract speaks.
    features: dict[str, float] = dict.fromkeys(CHAMPION_FEATURE_KEYS, 0.0)
    features.update(extract_champion_stats(champion))
    physical, magic, true = damage_shares or (0.0, 0.0, 0.0)
    features["physical_damage_share"] = float(physical)
    features["magic_damage_share"] = float(magic)
    features["true_damage_share"] = float(true)

    abilities = champion.get("abilities") or {}
    for slot in ABILITY_SLOTS:
        slot_abilities = abilities.get(slot, [])
        slot_features = extract_slot_features(slot_abilities)
        for key, value in slot_features.items():
            features[f"{slot}_{key}"] = value

    if edge_cases is not None:
        issues: list[str] = [
            f"missing ability slot {slot}" for slot in ABILITY_SLOTS if not abilities.get(slot)
        ]
        for slot in ABILITY_SLOTS:
            n_forms = len(abilities.get(slot, []))
            if n_forms > 2:  # noqa: PLR2004 - a base form plus one transform is normal
                issues.append(f"{slot} has {n_forms} forms")
        all_zero_slots = [
            slot
            for slot in ABILITY_SLOTS
            if all(
                features.get(f"{slot}_{k}", 0.0) == 0.0
                for k in ("base_damage", "cc_hard_seconds", "heal_amount", "shield_amount")
            )
            and not features.get(f"{slot}_is_passive", 0.0)
        ]
        if all_zero_slots:
            issues.append(f"no damage/cc/heal/shield extracted: {','.join(all_zero_slots)}")
        if issues:
            edge_cases[champion_key] = issues

    return features


# --- Precedence: the reviewed contract's reading over prose's ---


#: How far the two readings of a dim may differ and still be one number: both read the
#: same sourced row, so 796 of the 814 match bitwise and the nearest miss is 21% off.
AGREEMENT_RTOL = 1e-6

#: A (champion, dim) both sources state and read apart, and what prose reads instead.
#: The contract prices one named kit row; prose maxes every label of the family.
AGREEMENT_EXCEPTIONS: dict[tuple[str, str], str] = {
    ("Ahri", "P_heal_amount"): "prose maxes the second of the passive's two 'Heal' rows",
    ("Aurora", "P_heal_amount"): "prose maxes the second of the passive's two 'Heal' rows",
    ("Cho'Gath", "R_hp_gain_flat"): "the engine prices the stacked total; prose reads "
    "'Bonus Health Per Stack'",
    ("Diana", "W_shield_amount"): "prose books 'Maximum Shield Strength'",
    ("Graves", "E_armor_gain_flat"): "the engine prices one stack; prose books "
    "'Maximum Bonus Armor'",
    ("Hwei", "W_shield_amount"): "prose books 'Total Maximum Shield'",
    ("Locke", "W_heal_amount"): "prose maxes the second of the slot's two 'Heal' rows",
    ("Lux", "W_shield_amount"): "prose books 'Maximum Shield'",
    ("Master Yi", "W_damage_reduction_pct"): "prose mines the effect text; the kit row is "
    "'Modified Damage Reduction'",
    ("Neeko", "E_cc_hard_seconds"): "prose books 'Empowered Root Duration'",
    ("Renekton", "Q_heal_amount"): "prose books 'Enhanced Healing Cap'; the kit row is "
    "'Champion Healing'",
    ("Rumble", "W_shield_amount"): "prose books 'Enhanced Shield Strength'",
    ("Senna", "R_shield_amount"): "the engine prices the shield at the reference point; "
    "prose reads 'Shield Strength'",
    ("Shyvana", "W_shield_amount"): "the engine prices the shield at the reference point; "
    "prose reads 'Shield Strength'",
    ("Soraka", "R_heal_amount"): "prose books 'Increased Heal'",
    ("Udyr", "W_heal_amount"): "prose books 'Total Healing'; the kit row is 'Heal per Tick'",
    ("Udyr", "W_shield_amount"): "prose books 'Increased Shield Strength'",
    ("Vladimir", "R_heal_amount"): "prose books 'Maximum Total Heal'",
}


class KitDisagreement(ValueError):
    """The contract and prose state one dim and read different numbers."""


def assemble(
    champion: str, prose: Mapping[str, float], contract: Mapping[str, float]
) -> tuple[dict[str, float], dict[str, str]]:
    """One champion's store row: the contract wherever it sources a dim, prose elsewhere.

    Args:
        champion: the store's spelling, which is how `AGREEMENT_EXCEPTIONS` is keyed.
        prose: the `extract_all_features` reading, before precedence.
        contract: this champion's `reviewed_kit.contract_dims`.

    Returns:
        ``(values, sources)``: the row, and ``contract`` or ``exception`` per sourced dim.

    Raises:
        KitDisagreement: both state a dim, read it apart, and no exception declares why.
    """
    values = dict(prose)
    sources: dict[str, str] = {}
    for dim, value in contract.items():
        held = float(prose.get(dim, 0.0))
        agrees = math.isclose(held, value, rel_tol=AGREEMENT_RTOL, abs_tol=AGREEMENT_RTOL)
        if held and not agrees and (champion, dim) not in AGREEMENT_EXCEPTIONS:
            raise KitDisagreement(
                f"{champion} {dim}: the contract reads {value}, prose reads {held}. Fix "
                "the extractor or the bridge, or declare the row in AGREEMENT_EXCEPTIONS."
            )
        values[dim] = value
        sources[dim] = "contract" if held == 0.0 or agrees else "exception"
    return values, sources
