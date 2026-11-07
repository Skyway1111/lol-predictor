"""Vendor the merged-calculator's whole reviewed contract as the champion kit's source."""

# Run directly (the calculator is a sibling checkout, never a dependency):
#   python <this file> --calculator ../merged-calculator

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from league_predictor import clock
from scripts.champion_attributes.calculator_dump import read_cli_dump
from scripts.champion_attributes.reviewed_kit import (
    REVIEWED_KIT_PATH,
)

# Runs inside the calculator's own interpreter, through its own readers, so this
# program transports a reading and never re-derives one. Method: the architecture doc.
_DUMP_PROGRAM = '''
import json, sys
sys.path.insert(0, "src")
from calculator.champions import (
    get_champion_module_contract, registered_champion_names, module_basename,
)
from calculator.champions.contract_vocabulary import REQUIRED_CHAMPION_SLOTS
from calculator.champions.skill_orders import get_ability_rank
from calculator.champions.slot_extract import _axis_index, find_named_leveling
from calculator.data_fetcher import get_champion
from calculator.stats import resolve_pre_combat_stats

LEVEL = 18
TARGET = {
    "target_max_health": 2500.0,
    "target_current_health": 2500.0,
    "target_missing_health": 0.0,
}
RANK_SLOTS = ("Q", "W", "E", "R")

# A cast's self-stat grant, by the key the calculator's stat_buff carries.
BUFF_MECHANIC = {
    "armor_penetration_percent": ("armor_pen", "percent"),
    "armor_penetration_bonus_percent": ("armor_pen", "percent"),
    "magic_penetration_percent": ("magic_pen", "percent"),
    "magic_penetration_flat": ("magic_pen", "flat"),
    "lethality": ("lethality", "flat"),
    "armor": ("armor_gain", "flat"),
    "bonus_armor": ("armor_gain", "flat"),
    "magic_resistance": ("mr_gain", "flat"),
    "bonus_magic_resistance": ("mr_gain", "flat"),
    "health": ("health_gain", "flat"),
    "base_health": ("health_gain", "flat"),
    "bonus_health": ("health_gain", "flat"),
    "bonus_attack_damage": ("attack_damage_gain", "flat"),
    "ability_power": ("ability_power_gain", "flat"),
    "bonus_attack_speed": ("attack_speed_gain", "flat"),
    "critical_strike_chance": ("crit_chance_gain", "flat"),
    "life_steal": ("life_steal_gain", "flat"),
    "move_speed_flat": ("move_speed_gain", "flat"),
    "move_speed_percent": ("move_speed_gain", "percent"),
    "movement_speed_percent": ("move_speed_gain", "percent"),
}

# A cast's resistance shred, by the key the calculator's target_debuff carries.
DEBUFF_MECHANIC = {
    "armor_reduction_percent": ("armor_shred", "percent"),
    "armor_reduction_flat": ("armor_shred", "flat"),
    "mr_reduction_percent": ("mr_shred", "percent"),
    "mr_reduction_flat": ("mr_shred", "flat"),
}
# Qualifiers on a shred row: how long it lasts, how it ramps. Not mechanics.
DEBUFF_QUALIFIERS = ("duration", "stacks", "threshold_hits")

# The cached kit rows a mechanic is stated in, read only where the engine prices
# no payload for it. The calculator ingests and pins these rows against the
# module's packet sha, so reading one is reading the reviewed kit, not prose.
ROW_ATTRIBUTES = {
    "armor_pen": ("Armor Penetration",),
    "magic_pen": ("Magic Penetration",),
    "lethality": ("Lethality",),
    "armor_shred": (
        "Armor Reduction", "Resistances Reduction", "Resistances Reduction Per Stack",
    ),
    "mr_shred": (
        "Magic Resistance Reduction", "Enhanced Magic Resistance Reduction",
        "Resistances Reduction", "Resistances Reduction Per Stack",
    ),
    "armor_gain": (
        "Bonus Armor", "Increased Bonus Armor", "Maximum Bonus Armor",
        "Self Bonus Armor", "Bonus Resistances",
    ),
    "mr_gain": (
        "Bonus Magic Resistance", "Self Bonus Magic Resistance", "Bonus Resistances",
    ),
    "health_gain": ("Bonus Health",),
    "damage_reduction": (
        "Damage Reduction", "Damage reduction", "Flat Damage Reduction",
        "Magic Damage Reduction", "Physical Damage Reduction",
        "Modified Damage Reduction",
    ),
    "heal": (
        "Heal", "Total Heal", "Maximum Heal", "Minimum Heal", "Heal per Tick",
        "Heal Per Tick", "Champion Healing", "Total Healing",
    ),
    "shield": (
        "Shield Strength", "Shield", "Magic Shield Strength",
        "Maximum Shield Strength", "Initial Shield Strength", "Total Shield",
        "Initial Shield",
    ),
}

# A parsed row whose name carries no slot letter: a summon acting on its own, or a
# passive's second row. Any other such name raises rather than reaching the artifact.
STANDALONE_ROWS = ("passive_double_shot", "tibbers_attacks", "voidling_attacks")

# The only two units a row's number is typed by with no further context. Any
# other unit ("% of target armor", "% armor") prices against a stat this
# reference point fixes none of, so the row is recorded unevaluated with its
# unit rather than typed as one of these.
SCALE_FOR_UNIT = {"": "flat", "%": "percent"}


def refuse(what, champion, slot, key):
    raise SystemExit(
        "the calculator's " + what + " for " + champion + " " + slot + " carries "
        + repr(key) + ", which this bridge has no mechanic for. Add it to "
        "build_reviewed_kit's map (or name it a qualifier) rather than letting "
        "an axis leave the artifact."
    )


def engine_rows(entry, champion, slot):
    """Every mechanic the engine prices for this cast, in the entry's own keys."""
    rows = []
    for key, value in (entry.get("stat_buff") or {}).items():
        pair = BUFF_MECHANIC.get(key) or refuse("stat_buff", champion, slot, key)
        rows.append({
            "mechanic": pair[0], "scale": pair[1], "value": float(value),
            "source": "engine", "key": key,
            "innate": bool(entry.get("innate_grant")),
        })
    debuff = entry.get("target_debuff") or {}
    for key, value in debuff.items():
        if key in DEBUFF_QUALIFIERS:
            continue
        pair = DEBUFF_MECHANIC.get(key) or refuse("target_debuff", champion, slot, key)
        row = {
            "mechanic": pair[0], "scale": pair[1], "value": float(value),
            "source": "engine", "key": key,
        }
        row.update({q: float(debuff[q]) for q in DEBUFF_QUALIFIERS if q in debuff})
        rows.append(row)
    for event in entry.get("self_shield_events") or []:
        rows.append({
            "mechanic": "shield", "scale": "flat", "source": "engine",
            "value": float(event.get("amount", 0.0)),
            "duration": float(event.get("duration", 0.0)),
        })
    return rows


def row_value(leveling, rank):
    """One cached leveling row's (value, unit) at the reference point, nothing resolved.

    The calculator's own `_axis_index` picks the axis, so a level bracket reads at
    the reference level and a rank array at the rank; re-deriving that rule here
    would drift from the engine reading the same row. A rank of None is a row with
    no rank axis (a passive, a recast) and reads its last rank value; a rank of 0
    is a slot the kit never ranks (Jayce R) and clamps to the first.
    """
    for modifier in leveling.get("modifiers", []):
        values = modifier.get("values", [])
        if values:
            units = modifier.get("units", [])
            index = _axis_index(values, len(values) if rank is None else max(rank, 1), LEVEL)
            return float(values[index]), units[index] if index < len(units) else ""
    return None


def stated_row(abilities, mechanic, rank):
    """The kit's own statement of one mechanic for this slot, or None."""
    for attribute in ROW_ATTRIBUTES[mechanic]:
        for ability in abilities:
            leveling = find_named_leveling(ability, attribute)
            pair = row_value(leveling, rank) if leveling is not None else None
            if pair is None:
                continue
            value, unit = pair
            scale = SCALE_FOR_UNIT.get(unit)
            if scale is None:
                return {
                    "mechanic": mechanic, "attribute": attribute, "unit": unit,
                    "value": value,
                    "why": "the unit prices against a stat this reference fixes none of",
                }
            return {
                "mechanic": mechanic, "scale": scale, "value": value,
                "source": "kit_row", "attribute": attribute,
            }
    return None


def damage_rows(entry):
    """The slot's typed damage parts at the reference rank."""
    rows, unevaluated = [], []
    for part in entry.get("parts") or ():
        row = {
            "type": part.damage_type,
            "amount": round(float(part.amount), 6),
            "count": int(part.count),
        }
        if part.hp_scaled_damage is not None:
            try:
                # Both endpoints of the ramp: a target at full health, and one at none.
                at_full = float(part.hp_scaled_damage(0.0))
                at_empty = float(part.hp_scaled_damage(1.0))
            except Exception as exc:
                unevaluated.append({
                    "mechanic": "hp_scaled_damage", "why": repr(exc)[:120],
                })
            else:
                row["hp_scaled_amount"] = round(at_full, 6)
                row["hp_scaled_amount_max"] = round(at_empty, 6)
                row["pct_max_health"] = round(
                    100.0 * at_full / TARGET["target_max_health"], 6
                )
                row["pct_max_health_max"] = round(
                    100.0 * at_empty / TARGET["target_max_health"], 6
                )
        if part.cc_kind and part.cc_kind != "none":
            row["cc_kind"] = part.cc_kind
        if part.cc_duration:
            row["cc_duration"] = float(part.cc_duration)
        rows.append(row)
    return rows, unevaluated


def control(entry, declared):
    """The slot's control: the declared kind, the longest interval and the strength.

    Over both authoring routes — a control riding a damage part, and a
    control-only ``ControlEvent`` — because a kit uses either.
    """
    events = entry.get("control_events") or []
    return {
        "kind": declared,
        "seconds": max(
            [float(p.cc_duration) for p in (entry.get("parts") or ()) if p.cc_duration]
            + [float(e.duration) for e in events],
            default=0.0,
        ),
        "magnitude": max([float(e.magnitude) for e in events], default=0.0),
    }


def slot_dump(champion, slot, entry, abilities, rank, declared, coverage, channels):
    """One slot: what the contract prices, what its kit states, what stayed declared."""
    damage, unevaluated = damage_rows(entry)
    mechanics = engine_rows(entry, champion, slot)
    priced = {row["mechanic"] for row in mechanics}
    for mechanic in ROW_ATTRIBUTES:
        if mechanic in priced:
            continue
        stated = stated_row(abilities, mechanic, rank)
        if stated is None:
            continue
        (mechanics if "source" in stated else unevaluated).append(stated)
    cc = control(entry, declared)
    if declared not in ("none", "per_part") and cc["seconds"] <= 0.0:
        unevaluated.append({
            "mechanic": "cc_seconds", "declared": declared,
            "why": "the module declares the kind and prices no interval here",
        })
    unevaluated.extend({
        "mechanic": channel,
        "why": "a coverage channel the fight engine resolves, not this parse",
    } for channel in channels)
    return {
        "name": entry.get("name", ""),
        "rank": rank,
        "coverage": coverage,
        "damage": damage,
        "cc": cc,
        "mechanics": mechanics,
        "unevaluated": unevaluated,
    }


out = {}
for champion in registered_champion_names():
    contract = get_champion_module_contract(champion)
    data = get_champion(champion)
    stats = resolve_pre_combat_stats(
        data, LEVEL, [], item_options=None, role="", role_quest_complete=False,
        rune_page=None, external_stat_bonuses=None,
    )
    ranks = {s: get_ability_rank(s, LEVEL, champion) for s in RANK_SLOTS}
    parsed = contract.parse_abilities(
        data, LEVEL, stats.get("ability_power", 0.0), ability_ranks=dict(ranks),
        champion_stats=dict(stats), target_stats=dict(TARGET),
    )
    abilities = data.get("abilities") or {}
    channels = contract.coverage_channels or {}
    # A slot the parse emits no entry for still reaches the fight through a
    # coverage channel (Blitzcrank's passive shield, Zac's revive), so the union
    # of the five contract slots and the parse's own keys is the walk.
    entries = {("P" if s == "passive" else s): e for s, e in parsed.items()}
    for row_name in entries:
        if row_name[:1] not in REQUIRED_CHAMPION_SLOTS and row_name not in STANDALONE_ROWS:
            raise SystemExit(
                "the calculator parses " + champion + " row " + repr(row_name)
                + ", which names no slot. Declare it in STANDALONE_ROWS here and in "
                "reviewed_kit.EXTRA_SLOTS rather than letting an unread row into the "
                "artifact."
            )
    slots = {
        slot: slot_dump(
            champion, slot, entries.get(slot) or {}, abilities.get(slot) or [],
            ranks.get(slot[:1]),
            (contract.cc_kinds or {}).get(slot, "none"),
            (contract.coverage or {}).get(slot, "modeled"),
            tuple(channels.get(slot, ())),
        )
        for slot in sorted(set(REQUIRED_CHAMPION_SLOTS) | set(entries) | set(channels))
    }
    conversion = contract.stat_conversion
    out[champion] = {
        "module": module_basename(contract.module) + ".py",
        "ranks": ranks,
        "base_stats": {k: round(float(v), 6) for k, v in sorted(stats.items())},
        "stat_conversion": (
            None if conversion is None
            else {"source": conversion.source,
                  "attack_damage_ratio": float(conversion.attack_damage_ratio)}
        ),
        "revive": any("starting_revive_defense" in rows for rows in channels.values()),
        # The module owns a self-heal resolver. What it pays is a function of the
        # fight's own damage events, so this parse states that it exists and not
        # how much: a declared mechanic, unevaluated at the reference point.
        "self_healing_rule": getattr(contract.module, "SELF_HEALING_RULE", None) is not None,
        "slots": slots,
    }

json.dump({"level": LEVEL, "target": TARGET, "champions": out}, sys.stdout)
'''


def build_artifact(payload: Mapping, calculator: Path) -> dict:
    """Shape the dump into the vendored artifact: the kit plus a per-module sha256."""
    modules = calculator / "src" / "calculator" / "champions"
    champions: dict[str, dict] = {}
    receipts: dict[str, str] = {}
    for name, entry in sorted(payload["champions"].items()):
        module = entry.pop("module")
        champions[name] = entry
        receipts[name] = hashlib.sha256((modules / module).read_bytes()).hexdigest()
    return {
        "source": "merged-calculator reviewed champion module contracts, via its own registry",
        "generated": clock.utc_today().isoformat(),
        "reference": {
            "level": payload["level"],
            "ranks": "each slot's sourced rank at the reference level (skill_orders)",
            "items": "none",
            "target": payload["target"],
        },
        "module_sha256": receipts,
        "champions": champions,
    }


def report(artifact: Mapping) -> str:
    """The counts the build prints: what was read, and what stayed declared."""
    champions = artifact["champions"]
    slots = [slot for entry in champions.values() for slot in entry["slots"].values()]
    control = [s for s in slots if s["cc"]["kind"] not in ("none", "per_part")]
    priced = sum(1 for s in control if s["cc"]["seconds"] > 0)
    return "\n".join(
        (
            f"{len(champions):5d}  champions, level {artifact['reference']['level']}, no items",
            f"{len(slots):5d}  slots",
            f"{sum(len(s['damage']) for s in slots):5d}  typed damage parts",
            f"{sum(len(s['mechanics']) for s in slots):5d}  mechanic rows",
            f"{priced:5d}  of {len(control)} declared control slots price their seconds",
            f"{sum(len(s['unevaluated']) for s in slots):5d}  declared-but-unevaluated entries",
        )
    )


def main() -> None:
    payload, calculator = read_cli_dump(_DUMP_PROGRAM)
    artifact = build_artifact(payload, calculator)
    REVIEWED_KIT_PATH.write_text(
        json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report(artifact))
    print(f"Wrote {REVIEWED_KIT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
