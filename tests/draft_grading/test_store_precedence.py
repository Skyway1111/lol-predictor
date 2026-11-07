"""Schema v6 precedence: what the contract sources, what stays prose's, and the meta."""

import json

import pytest

from league_predictor.draft_grading.champion_attributes.ability_schema import (
    ALL_FEATURE_KEYS,
    SCHEMA_VERSION,
)
from league_predictor.draft_grading.champion_attributes.ability_store import (
    AbilityFeatureStore,
)
from scripts.champion_attributes.ability_features import (
    AGREEMENT_EXCEPTIONS,
    KitDisagreement,
    assemble,
    extract_all_features,
)
from scripts.champion_attributes.damage_shares import (
    load_damage_profile_cache,
)
from scripts.champion_attributes.reviewed_kit import (
    CHAMPION_LEVEL_DIMS,
    load_reviewed_kit,
    resolved_contracts,
)
from tests.draft_grading._knowledge_audit import audit_prose

#: The five keys schema v6 appends, in their `CHAMPION_FEATURE_KEYS` order.
PENETRATION_KEYS: tuple[str, ...] = (
    "armor_pen_flat",
    "armor_pen_pct",
    "magic_pen_flat",
    "magic_pen_pct",
    "lethality",
)

#: The kits that state penetration today, the campaign's opening finding made a roster.
PENETRATION_ROSTERS: dict[str, dict[str, float]] = {
    "armor_pen_pct": {"Ambessa": 30.0, "Darius": 40.0, "Pantheon": 30.0, "Zaahen": 30.0},
    "magic_pen_pct": {"Annie": 20.0, "Mordekaiser": 15.0},
    "lethality": {"Aphelios": 27.0},
}

#: A dim whose contract value differs from what prose reads, and the store's answer.
#: One per precedence route: a declared exception, and a dim prose reads as zero.
PRECEDENCE_CASES: tuple[tuple[str, str, float, float], ...] = (
    ("Lux", "W_shield_amount", 100.0, 200.0),
    ("Zilean", "E_cc_slow_duration", 2.5, 0.0),
)

#: An axis the calculator's coverage frontier leaves out, so prose owns it outright.
FRONTIER_CASES: tuple[tuple[str, str], ...] = (
    ("Ezreal", "E_has_dash"),
    ("Akali", "W_has_stealth"),
    ("Anivia", "W_has_terrain"),
)


@pytest.fixture(scope="module")
def kit():
    return load_reviewed_kit()


@pytest.fixture(scope="module")
def artifact(store: AbilityFeatureStore) -> dict:
    """The checked-in store as written, header and source meta included."""
    return json.loads(store._path.read_text(encoding="utf-8"))


# --- precedence ---


def test_a_contract_dim_wins_over_the_prose_reading(store, prose_reading):
    """The campaign's rule: where both speak, the reviewed contract is the store."""
    for champion, dim, contract_value, prose_value in PRECEDENCE_CASES:
        assert prose_reading[champion][dim] == pytest.approx(prose_value), dim
        assert store.get(champion)[dim] == pytest.approx(contract_value), dim


def test_a_frontier_axis_comes_from_prose(store, prose_reading, artifact):
    """An axis the engine does not model is prose's, and the meta says so."""
    for champion, dim in FRONTIER_CASES:
        assert store.get(champion)[dim] == pytest.approx(prose_reading[champion][dim])
        assert store.get(champion)[dim] > 0.0, dim
        assert dim not in artifact["sources"].get(champion, {})


def test_a_disagreement_outside_tolerance_names_the_champion_and_the_dim():
    """An undeclared disagreement stops the build rather than picking a side quietly."""
    with pytest.raises(KitDisagreement, match=r"Nemesis W_heal_amount"):
        assemble("Nemesis", {"W_heal_amount": 40.0}, {"W_heal_amount": 90.0})


def test_agreement_inside_tolerance_reads_as_the_contract():
    """Both sides read the same sourced number, so the pair is one value, not two."""
    values, sources = assemble("Nemesis", {"W_heal_amount": 90.0}, {"W_heal_amount": 90.0})
    assert values["W_heal_amount"] == pytest.approx(90.0)
    assert sources == {"W_heal_amount": "contract"}


def test_every_declared_exception_is_a_live_disagreement(prose_reading, kit):
    """A row that stops disagreeing is stale, and a new one may not pass quietly."""
    audit = audit_prose(prose_reading, kit)
    assert {(champion, dim) for champion, dim, _, _ in audit.disagree} == set(AGREEMENT_EXCEPTIONS)
    assert all(reason for reason in AGREEMENT_EXCEPTIONS.values())


# --- the penetration keys ---


def test_the_penetration_keys_are_the_last_five_dims():
    """Appended, never inserted: every earlier key keeps its name and its neighbours."""
    assert ALL_FEATURE_KEYS[20:25] == PENETRATION_KEYS


def test_the_penetration_keys_carry_exactly_what_the_contract_declares(store, kit):
    """A key nonzero anywhere else would be a prose reading of an axis prose cannot see."""
    contracts, _ = resolved_contracts(store.champions, kit)
    for key in PENETRATION_KEYS:
        declared = {name for name, dims in contracts.items() if dims.get(key)}
        assert {name for name in store.champions if store.get(name)[key]} == declared, key


def test_the_penetration_rosters_are_the_kits_the_campaign_named(store):
    """The three the fight engine prices none of reach the model through this store."""
    for key, roster in PENETRATION_ROSTERS.items():
        assert {name for name in store.champions if store.get(name)[key]} == set(roster)
        for name, value in roster.items():
            assert store.get(name)[key] == pytest.approx(value)


# --- the source meta ---


def test_the_artifact_stamps_its_schema(artifact):
    """A row of the previous schema would read every v6 key as zero."""
    assert artifact["schema_version"] == SCHEMA_VERSION
    assert artifact["source_default"] == "prose"


def test_a_store_of_the_previous_schema_is_refused(tmp_path):
    """The store names the build to run rather than serving five silent zeros."""
    path = tmp_path / "ability_features.json"
    path.write_text(json.dumps({"Lux": {}}), encoding="utf-8")
    with pytest.raises(ValueError, match="generate_abilities"):
        AbilityFeatureStore(path=path)


def test_every_dim_of_every_champion_resolves_to_one_source(artifact):
    """The meta is total: a dim outside the three would have no provenance at all."""
    for champion, row in artifact["champions"].items():
        sources = artifact["sources"].get(champion, {})
        assert set(row) == set(ALL_FEATURE_KEYS), champion
        assert set(sources) <= set(row), champion
        assert set(sources.values()) <= {"contract", "exception"}, champion


def test_the_meta_names_exactly_the_dims_the_contract_sources(artifact, kit):
    """A dim sourced without a receipt, or a receipt without a source, is a broken read."""
    contracts, unresolved = resolved_contracts(artifact["champions"], kit)
    assert unresolved == ()
    for champion, dims in contracts.items():
        assert set(artifact["sources"].get(champion, {})) == set(dims), champion


def test_an_exception_is_stamped_as_one(artifact):
    """The three sources are distinguishable, so a declared disagreement is auditable."""
    stamped = {
        (champion, dim)
        for champion, row in artifact["sources"].items()
        for dim, source in row.items()
        if source == "exception"
    }
    assert stamped == set(AGREEMENT_EXCEPTIONS)


@pytest.mark.slow
def test_the_checked_in_store_is_exactly_what_the_build_writes(artifact):
    """No store value is hand-edited: a rebuild in memory matches it cell for cell."""
    from league_predictor.draft_grading.champion_attributes.wiki_export import (
        DEFAULT_WIKI_EXPORT_PATH,
        load_wiki_champion_export,
    )
    from scripts.champion_attributes.generate_abilities import (
        apply_contract,
        resolve_damage_shares,
    )

    wiki = load_wiki_champion_export(DEFAULT_WIKI_EXPORT_PATH)
    cache = load_damage_profile_cache()
    prose = {}
    for key in sorted(wiki):
        champion = wiki[key]
        name = str(champion.get("name") or key)
        shares = resolve_damage_shares(cache, name, champion)
        prose[name] = extract_all_features(key, champion, damage_shares=shares)
    rebuilt, sources = apply_contract(prose)
    assert rebuilt == artifact["champions"]
    assert sources == artifact["sources"]


def test_a_champion_level_dim_is_never_slot_prefixed(artifact):
    """Penetration is the build's, not a cast's: one dim per champion, not five."""
    for row in artifact["sources"].values():
        assert not any(dim[2:] in CHAMPION_LEVEL_DIMS for dim in row)
