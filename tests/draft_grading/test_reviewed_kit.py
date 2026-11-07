"""The reviewed-kit bridge: what the calculator's contract sources, and what it may not."""

import re

import pytest

from league_predictor.draft_grading.champion_attributes.ability_schema import (
    ABILITY_SLOTS,
    ALL_FEATURE_KEYS,
)
from scripts.champion_attributes.calculator_dump import dump_from_calculator
from scripts.champion_attributes.reviewed_kit import (
    CC_TYPE_DIMS,
    CHAMPION_LEVEL_DIMS,
    EXTRA_SLOTS,
    SLOT_LEVEL_DIMS,
    SOURCEABLE_DIMS,
    STORE_DIMS,
    UNREPRESENTED_CC_KINDS,
    UNREPRESENTED_MECHANICS,
    ReviewedKitUnavailable,
    contract_dims,
    load_reviewed_kit,
    resolved_contracts,
)
from tests.draft_grading._knowledge_audit import audit_prose, load_reviewed_cc

#: Every store-dim family a reviewed contract may source, whatever today's roster
#: happens to state. A dim outside it is prose's to own, whatever the calculator knows.
CONTRACT_SOURCED_DIMS: frozenset[str] = frozenset(
    {
        "damage_type_physical",
        "damage_type_magic",
        "damage_type_true",
        "cc_hard_seconds",
        "cc_slow_duration",
        "cc_slow_pct",
        "cc_type_stun",
        "cc_type_root",
        "cc_type_suppress",
        "cc_type_silence",
        "cc_type_charm",
        "cc_type_taunt",
        "cc_type_fear",
        "cc_type_sleep",
        "cc_type_knockup",
        "cc_type_knockback",
        "cc_type_pull",
        "cc_type_polymorph",
        "heal_amount",
        "shield_amount",
        "shield_hp_pct",
        "armor_gain_flat",
        "armor_gain_pct",
        "mr_gain_flat",
        "mr_gain_pct",
        "hp_gain_flat",
        "damage_reduction_pct",
        "armor_shred_pct",
        "armor_pen_flat",
        "armor_pen_pct",
        "magic_pen_flat",
        "magic_pen_pct",
        "lethality",
    }
)

#: The three the store one-hots from the wiki's single `damageType`.
DAMAGE_TYPE_DIMS: tuple[str, ...] = (
    "damage_type_physical",
    "damage_type_magic",
    "damage_type_true",
)

#: Families most contracts state nothing about; the audit reads them for every champion
#: even so, which is what makes the prose-only direction more than a rounding error.
BLIND_SPOT_FAMILIES: tuple[str, ...] = (
    "heal_amount",
    "shield_hp_pct",
    "armor_shred_pct",
    "damage_reduction_pct",
)

#: Axes the campaign plan leaves to prose. None may reach `contract_dims`.
FRONTIER_DIMS: tuple[str, ...] = (
    "has_stealth",
    "has_terrain",
    "has_dash",
    "has_ms_boost",
    "has_untargetable",
    "has_invulnerable",
    "has_stasis",
    "has_spellshield",
    "has_projectile_block",
    "has_unstoppable",
    "has_revive",
    "is_projectile",
    "is_skillshot",
    "affects_allies",
)


def family(dim: str) -> str:
    """The dim's family: a slot dim drops its slot, a champion-level dim is its own."""
    return dim if dim in CHAMPION_LEVEL_DIMS else dim.partition("_")[2]


@pytest.fixture(scope="module")
def kit():
    return load_reviewed_kit()


def test_the_artifact_vendors_every_reviewed_champion(kit):
    """A champion the bridge dropped would read as an empty kit, never as a miss."""
    assert kit.champions


def test_every_champion_carries_a_source_module_sha(kit):
    """The receipt is what makes a vendored reading auditable against its source."""
    assert set(kit.module_sha256) == set(kit.champions)
    assert all(re.fullmatch(r"[0-9a-f]{64}", sha) for sha in kit.module_sha256.values())


def test_the_header_names_the_reference_the_numbers_were_read_at(kit):
    """A damage number with no level and no build is not a number anyone can use."""
    assert kit.level == 18
    assert kit.reference["items"] == "none"
    assert kit.reference["target"]["target_max_health"] > 0


def test_every_champion_carries_the_five_slots(kit):
    """A recast row is extra; the five contract slots are never missing."""
    for champion, entry in kit.champions.items():
        assert set(ABILITY_SLOTS) <= set(entry["slots"]), champion


def test_a_known_slot_reads_its_known_damage_part(kit):
    """Pantheon W is one physical part that stuns for the sourced second."""
    (part,) = kit.champions["Pantheon"]["slots"]["W"]["damage"]
    assert part["type"] == "physical"
    assert part["count"] == 1
    assert kit.champions["Pantheon"]["slots"]["W"]["cc"] == {
        "kind": "stun",
        "seconds": 1.0,
        "magnitude": 0.0,
    }


def test_a_repeated_part_carries_its_hit_count(kit):
    """Renekton R is 30 sourced 0.5-second ticks, not one lump of magic damage."""
    (part,) = kit.champions["Renekton"]["slots"]["R"]["damage"]
    assert (part["type"], part["count"]) == ("magic", 30)


def test_the_penetration_the_store_cannot_see_is_in_the_artifact(kit):
    """The campaign's opening finding: three kits the prose reading reads as zero."""
    assert contract_dims("Pantheon", kit)["armor_pen_pct"] == 30.0
    assert contract_dims("Zaahen", kit)["armor_pen_pct"] == 30.0
    assert contract_dims("Renekton", kit)["E_armor_shred_pct"] == 35.0


def test_a_mechanic_the_engine_prices_is_read_from_the_engine(kit):
    """Darius E's armor penetration is an engine payload, not a cached row."""
    (row,) = [
        row
        for row in kit.champions["Darius"]["slots"]["E"]["mechanics"]
        if row["mechanic"] == "armor_pen"
    ]
    assert (row["source"], row["value"]) == ("engine", 40.0)


def test_a_row_with_no_rank_axis_reads_its_last_value(kit):
    """Aphelios P brackets lethality by level, 4.5 to 27; rank 1 would read the floor."""
    passive = kit.champions["Aphelios"]["slots"]["P"]
    assert passive["rank"] is None
    (row,) = [r for r in passive["mechanics"] if r["mechanic"] == "lethality"]
    assert row["value"] == 27.0


def test_base_stats_are_the_reference_level_with_no_items(kit):
    """Every kit number is read against the same stat block, so two kits compare."""
    stats = kit.champions["Pantheon"]["base_stats"]
    assert stats["ability_power"] == 0.0
    assert stats["health"] > 1000


# --- the precedence contract package 2b consumes ---


def test_the_declared_reach_is_what_the_maps_can_source():
    """The reach is the maps', not today's roster: a dim no champion states is still ruled."""
    assert SLOT_LEVEL_DIMS | CHAMPION_LEVEL_DIMS == CONTRACT_SOURCED_DIMS
    assert {
        f"{slot}_{name}" for slot in ABILITY_SLOTS for name in SLOT_LEVEL_DIMS
    } | CHAMPION_LEVEL_DIMS == SOURCEABLE_DIMS


def test_contract_dims_sources_only_the_declared_families(kit):
    """The dims the contract owns are a closed set; everything else stays prose's."""
    reached = {family(dim) for c in kit.champions for dim in contract_dims(c, kit)}
    assert reached <= CONTRACT_SOURCED_DIMS


def test_a_slot_states_one_damage_class(kit):
    """The store one-hots the wiki's single damageType, so the contract states one too."""
    for champion in kit.champions:
        dims = contract_dims(champion, kit)
        for slot in ABILITY_SLOTS:
            stated = [d for d in DAMAGE_TYPE_DIMS if dims.get(f"{slot}_{d}")]
            assert len(stated) <= 1, f"{champion} {slot} {stated}"


def test_a_two_class_slot_states_its_largest_part(kit):
    """Fizz Q is 70 magic and 109 physical; the physical half is the class it reads as."""
    parts = {p["type"]: p["amount"] for p in kit.champions["Fizz"]["slots"]["Q"]["damage"]}
    assert parts["physical"] > parts["magic"]
    dims = contract_dims("Fizz", kit)
    assert dims["Q_damage_type_physical"] == 1.0
    assert "Q_damage_type_magic" not in dims


def test_a_tied_two_class_slot_breaks_toward_physical_then_magic(kit):
    """Yone W is 175 physical and 175 magic, Ahri Q 135 magic and 135 true."""
    assert contract_dims("Yone", kit)["W_damage_type_physical"] == 1.0
    assert contract_dims("Ahri", kit)["Q_damage_type_magic"] == 1.0


def test_contract_dims_never_reaches_a_prose_only_axis(kit):
    """A frontier axis the calculator does not model must never read as contract-sourced."""
    reached = {family(dim) for c in kit.champions for dim in contract_dims(c, kit)}
    assert reached.isdisjoint(FRONTIER_DIMS)


def test_every_dim_the_contract_sources_is_a_store_key(kit):
    """A dim name outside the schema is a typo that would audit against nothing."""
    for champion in kit.champions:
        for dim in contract_dims(champion, kit):
            assert dim in ALL_FEATURE_KEYS, f"{champion} {dim}"


def test_every_dim_the_contract_sources_names_a_real_slot_or_the_champion(kit):
    """A recast row folds onto its slot and penetration onto the champion; nothing else."""
    for champion in kit.champions:
        for dim in contract_dims(champion, kit):
            named = dim in CHAMPION_LEVEL_DIMS or dim.partition("_")[0] in ABILITY_SLOTS
            assert named, f"{champion} {dim}"


def test_every_mechanic_the_artifact_states_has_a_store_rule(kit):
    """A widened dump must move this map, never leave a mechanic sourcing nothing."""
    stated = {
        row["mechanic"]
        for entry in kit.champions.values()
        for slot in entry["slots"].values()
        for row in slot["mechanics"]
    }
    assert stated <= set(STORE_DIMS)


def test_every_control_kind_the_calculator_can_state_has_a_store_rule():
    """The whole vocabulary, not today's artifact: an unused kind must still be ruled."""
    assert set(load_reviewed_cc()["vocabulary"]) <= set(CC_TYPE_DIMS)


def test_every_extra_row_the_artifact_states_is_folded_or_dropped(kit):
    """A new recast spelling must be ruled, not silently become a slot of its own."""
    extra = {
        name
        for entry in kit.champions.values()
        for name in entry["slots"]
        if name not in ABILITY_SLOTS
    }
    assert extra <= set(EXTRA_SLOTS)


def test_a_mechanic_with_no_dim_carries_its_reason():
    """An axis the schema drops is a decision with a receipt, never an omission."""
    assert {m for m, dims in STORE_DIMS.items() if not dims} == set(UNREPRESENTED_MECHANICS)
    assert {k for k, dim in CC_TYPE_DIMS.items() if dim is None} == set(UNREPRESENTED_CC_KINDS)


# --- the audit, and the refusals ---


def test_every_reviewed_champion_resolves_in_the_prose_reading(prose_reading, kit):
    """A champion the reading has no row for would audit as an all-zero vector."""
    assert audit_prose(prose_reading, kit).unresolved == ()


def test_the_audit_reads_a_dim_the_contract_is_silent_on(prose_reading, kit):
    """The prose-only direction fires where a contract says nothing, not only where it speaks."""
    audit = audit_prose(prose_reading, kit)
    contracts, _ = resolved_contracts(prose_reading, kit)
    silent = {
        (champion, dim)
        for champion, dims in contracts.items()
        for dim in SOURCEABLE_DIMS & set(ALL_FEATURE_KEYS)
        if dim not in dims
    }
    reported = {(champion, dim) for champion, dim, _ in audit.prose_only}
    assert reported <= silent
    assert set(BLIND_SPOT_FAMILIES) <= {family(dim) for _, dim in reported}


def test_the_contract_states_a_shield_wherever_prose_reads_one(prose_reading, kit):
    """Both read the cached Shield Strength row, so shield_amount has no one-sided read."""
    audit = audit_prose(prose_reading, kit)
    assert [dim for _, dim, _ in audit.prose_only if dim.endswith("shield_amount")] == []


def test_the_audit_compares_both_directions(prose_reading, kit):
    """The report the store build works from: agreement and both one-sided reads."""
    audit = audit_prose(prose_reading, kit)
    contracts, _ = resolved_contracts(prose_reading, kit)
    assert audit.counts["agree"] > 0
    assert audit.counts["dims compared"] == sum(len(dims) for dims in contracts.values())
    assert audit.prose_zero
    assert audit.prose_only
    assert audit.table()


def test_a_missing_artifact_names_the_build_that_writes_it(tmp_path):
    """An absent vendored file is a build to run, not a champion store of zeros."""
    with pytest.raises(ReviewedKitUnavailable, match="build_reviewed_kit"):
        load_reviewed_kit(tmp_path / "nothing.json")


def test_a_bogus_checkout_names_what_is_missing(tmp_path):
    """The bridge refuses rather than vendoring an empty kit from nowhere."""
    with pytest.raises(RuntimeError, match="not a merged-calculator checkout"):
        dump_from_calculator(tmp_path, tmp_path / "python.exe", "")


def test_a_bogus_interpreter_names_what_is_missing(tmp_path):
    """The calculator's own interpreter runs the dump; ours cannot answer for it."""
    (tmp_path / "src" / "calculator").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="does not exist"):
        dump_from_calculator(tmp_path, tmp_path / "nowhere" / "python.exe", "")
