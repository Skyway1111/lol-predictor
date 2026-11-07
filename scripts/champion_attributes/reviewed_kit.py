"""Read the calculator's vendored reviewed kit, and audit the prose reading against it."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor.draft_grading.champion_attributes.ability_schema import ABILITY_SLOTS
from league_predictor.draft_grading.champion_attributes.normalize import (
    normalize_champion_name,
    to_canonical,
)
from scripts.champion_attributes.paths import DATA_DIR
from scripts.champion_attributes.reviewed_cc import (
    UNREPRESENTED_KINDS,
    VARIES_PER_PART,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

REVIEWED_KIT_PATH = DATA_DIR / "reviewed_kit.json"

#: Each kit mechanic's store dim, by the scale its value is stated on; the ONE
#: place the two vocabularies meet. Empty: UNREPRESENTED_MECHANICS says why.
STORE_DIMS: dict[str, dict[str, str]] = {
    "armor_pen": {"flat": "armor_pen_flat", "percent": "armor_pen_pct"},
    "magic_pen": {"flat": "magic_pen_flat", "percent": "magic_pen_pct"},
    "lethality": {"flat": "lethality"},
    "armor_shred": {"percent": "armor_shred_pct"},
    "armor_gain": {"flat": "armor_gain_flat", "percent": "armor_gain_pct"},
    "mr_gain": {"flat": "mr_gain_flat", "percent": "mr_gain_pct"},
    "health_gain": {"flat": "hp_gain_flat"},
    "damage_reduction": {"percent": "damage_reduction_pct"},
    "heal": {"flat": "heal_amount"},
    "shield": {"flat": "shield_amount", "percent": "shield_hp_pct"},
    "mr_shred": {},
    "attack_damage_gain": {},
    "ability_power_gain": {},
    "attack_speed_gain": {},
    "crit_chance_gain": {},
    "life_steal_gain": {},
    "move_speed_gain": {},
}

#: A mechanic the contract sources that the store carries no dim for, with the reason.
UNREPRESENTED_MECHANICS: dict[str, str] = {
    "mr_shred": "the schema shreds armor only; armor_shred_pct has no MR twin",
    "attack_damage_gain": "an offensive steroid; the schema reads damage, not the stat",
    "ability_power_gain": "an offensive steroid; the schema reads damage, not the stat",
    "attack_speed_gain": "an offensive steroid; the schema reads damage, not the stat",
    "crit_chance_gain": "an offensive steroid; the schema reads damage, not the stat",
    "life_steal_gain": "sustain through damage; heal_amount reads a stated heal only",
    "move_speed_gain": "a frontier axis: has_ms_boost is the prose flag, and stays prose",
}

#: Store dims a contract states per slot and the store carries once per champion: a
#: kit's penetration is the build's, whichever cast grants it. Folded by max over slots.
CHAMPION_LEVEL_DIMS: frozenset[str] = frozenset(
    {"armor_pen_flat", "armor_pen_pct", "magic_pen_flat", "magic_pen_pct", "lethality"}
)

#: The calculator's whole control vocabulary in the store's spelling, or None
#: where this schema has no dim. Total: an unknown kind raises, never reads zero.
CC_TYPE_DIMS: dict[str, str | None] = {
    "stun": "cc_type_stun",
    "root": "cc_type_root",
    "snare": "cc_type_root",
    "suppression": "cc_type_suppress",
    "silence": "cc_type_silence",
    "charm": "cc_type_charm",
    "taunt": "cc_type_taunt",
    "fear": "cc_type_fear",
    "flee": "cc_type_fear",
    "sleep": "cc_type_sleep",
    "knockup": "cc_type_knockup",
    "airborne": "cc_type_knockup",
    "knockback": "cc_type_knockback",
    "pull": "cc_type_pull",
    "polymorph": "cc_type_polymorph",
    "slow": None,
    "immobilize": None,
    "berserk": None,
    "cripple": None,
    "stasis": None,
    "blind": None,
    "none": None,
    VARIES_PER_PART: None,
}

#: Why a control kind reaches no type dim. `blind`'s reason has one home, the CC
#: gate's own table; the rest are this map's.
UNREPRESENTED_CC_KINDS: dict[str, str] = {
    **UNREPRESENTED_KINDS,
    "slow": "not a type flag here: cc_slow_duration and cc_slow_pct carry a slow",
    "immobilize": "a category over root/stun/knockup, not one of them",
    "berserk": "no dim expresses a forced-attack state",
    "cripple": "an attack-speed slow; the schema reads no attack-speed debuff",
    "stasis": "has_stasis is the self-denial flag; no dim applies stasis to an enemy",
    "none": "the reviewed no-control answer",
    VARIES_PER_PART: "no control is asserted on any particular part",
}

#: A parsed row outside the five slots, folded onto its slot, or None where it
#: has none (a summon). Total: a new row name raises, never becomes a sixth slot.
EXTRA_SLOTS: dict[str, str | None] = {
    "P2": "P",
    "Q2": "Q",
    "Q_passive": "Q",
    "W_frenzy": "W",
    "W_vigor": "W",
    "E_opener": "E",
    "E_passive": "E",
    "R_buff": "R",
    "R_onhit": "R",
    "passive_double_shot": "P",
    "tibbers_attacks": None,
    "voidling_attacks": None,
}

#: The damage reading both sources mean alike, in the order a tie breaks on. The
#: amount is not one: a cast priced at level 18 is not the leveling row's base.
_DAMAGE_TYPE_DIMS: dict[str, str] = {
    "physical": "damage_type_physical",
    "magic": "damage_type_magic",
    "true": "damage_type_true",
}

#: Every family the maps above can source once per slot; the champion-level five are
#: theirs alone.
SLOT_LEVEL_DIMS: frozenset[str] = (
    frozenset(_DAMAGE_TYPE_DIMS.values())
    | {"cc_hard_seconds", "cc_slow_duration", "cc_slow_pct"}
    | {dim for dim in CC_TYPE_DIMS.values() if dim is not None}
    | {dim for dims in STORE_DIMS.values() for dim in dims.values()}
) - CHAMPION_LEVEL_DIMS

#: Every dim the maps above can source. The audit reads all of them for every champion,
#: so the prose-only direction fires where a contract is silent.
SOURCEABLE_DIMS: frozenset[str] = (
    frozenset(f"{slot}_{dim}" for slot in ABILITY_SLOTS for dim in SLOT_LEVEL_DIMS)
    | CHAMPION_LEVEL_DIMS
)


class ReviewedKitUnavailable(RuntimeError):
    """The calculator checkout, its interpreter, or the artifact is absent."""


@dataclass(frozen=True)
class ReviewedKit:
    """The vendored contract reading: its reference point, its receipts, its kits."""

    reference: dict
    module_sha256: dict[str, str]
    champions: dict[str, dict]

    @property
    def level(self) -> int:
        """The champion level every slot in this artifact was evaluated at."""
        return int(self.reference["level"])


def load_reviewed_kit(path: Path = REVIEWED_KIT_PATH) -> ReviewedKit:
    """The vendored reviewed kit (scripts/build_reviewed_kit.py writes it).

    Raises:
        ReviewedKitUnavailable: the artifact has not been vendored.
    """
    if not path.is_file():
        raise ReviewedKitUnavailable(
            f"{path} does not exist — run champion_attributes/scripts/"
            "build_reviewed_kit.py --calculator <path to the sibling checkout>."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReviewedKit(
        reference=payload["reference"],
        module_sha256=payload["module_sha256"],
        champions=payload["champions"],
    )


def _cc_dims(cc: Mapping) -> dict[str, float]:
    """The dims one slot's reviewed kind, priced seconds and strength source.

    Raises:
        KeyError: a control kind outside the calculator's vocabulary as mapped.
    """
    seconds, magnitude = float(cc["seconds"]), float(cc["magnitude"])
    if cc["kind"] == "slow":
        return {
            dim: value
            for dim, value in (("cc_slow_duration", seconds), ("cc_slow_pct", magnitude))
            if value > 0
        }
    dim = CC_TYPE_DIMS[cc["kind"]]
    if dim is None:
        return {}
    return {dim: 1.0} | ({"cc_hard_seconds": seconds} if seconds > 0 else {})


def _damage_dim(parts: list[dict]) -> str | None:
    """The slot's one damage class: the largest part, ties by _DAMAGE_TYPE_DIMS order."""
    order = list(_DAMAGE_TYPE_DIMS)
    ranked = sorted(
        parts,
        key=lambda p: (-p.get("hp_scaled_amount", p["amount"]), order.index(p["type"])),
    )
    return _DAMAGE_TYPE_DIMS[ranked[0]["type"]] if ranked else None


def contract_dims(champion: str, kit: ReviewedKit | None = None) -> dict[str, float]:
    """The store dims this champion's reviewed contract sources, with their values.

    Returns:
        Store-dim name to value, slot-prefixed but for `CHAMPION_LEVEL_DIMS`. Damage
        is one-hot per slot, the class of the largest part at the reference rank,
        because the store one-hots the wiki's single `damageType`; a slot dealing two
        classes states its dominant one.

    Raises:
        KeyError: no reviewed contract for this champion, or a row this map
            has no rule for.
    """
    kit = kit or load_reviewed_kit()
    dims: dict[str, float] = {}

    def state(slot: str, dim: str, value: float) -> None:
        """A recast row folds onto its slot, and a champion-level dim onto the champion."""
        key = dim if dim in CHAMPION_LEVEL_DIMS else f"{slot}_{dim}"
        dims[key] = max(dims.get(key, 0.0), value)

    parts: dict[str, list[dict]] = {}
    for row_name, slot_kit in kit.champions[champion]["slots"].items():
        slot = row_name if row_name in ABILITY_SLOTS else EXTRA_SLOTS[row_name]
        if slot is None:
            continue
        parts.setdefault(slot, []).extend(slot_kit["damage"])
        for dim, value in _cc_dims(slot_kit["cc"]).items():
            state(slot, dim, value)
        for row in slot_kit["mechanics"]:
            dim = STORE_DIMS[row["mechanic"]].get(row["scale"])
            if dim is not None:
                state(slot, dim, float(row["value"]))
    for slot, slot_parts in parts.items():
        damage = _damage_dim(slot_parts)
        if damage is not None:
            state(slot, damage, 1.0)
    return dims


def _resolve(champion: str, lookup: Mapping[str, str]) -> str | None:
    """The reading's own spelling of one champion, by normalized then canonical name."""
    return lookup.get(normalize_champion_name(champion)) or lookup.get(
        normalize_champion_name(to_canonical(champion))
    )


def resolved_contracts(
    names: Iterable[str], kit: ReviewedKit | None = None
) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
    """Every reviewed contract's store dims under the caller's own spelling of the name.

    Returns:
        ``(by_name, unresolved)``: the dims each resolved name sources, and the
        reviewed champions no name in `names` spells.
    """
    kit = kit or load_reviewed_kit()
    lookup = {normalize_champion_name(name): name for name in names}
    by_name: dict[str, dict[str, float]] = {}
    unresolved: list[str] = []
    for champion in sorted(kit.champions):
        resolved = _resolve(champion, lookup)
        if resolved is None:
            unresolved.append(champion)
        else:
            by_name[resolved] = contract_dims(champion, kit)
    return by_name, tuple(unresolved)
