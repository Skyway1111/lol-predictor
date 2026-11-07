"""THE home for "this champion should read like X": the claim table over the kit vector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from league_predictor.draft_grading.champion_attributes.ability_store import (
    AbilityFeatureStore,
)
from league_predictor.draft_grading.champion_attributes.kit_vector import (
    GLOBAL_RANGE,
    KIT_FAMILIES,
    KIT_FAMILY_NAMES,
    REFERENCE_LEVEL,
    kit_matrix,
    kit_vector,
)
from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from scripts.champion_attributes.reviewed_kit import load_reviewed_kit

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


def _has_armor_pen(kit: Mapping[str, float]) -> bool:
    return kit["armor_pen_pct"] > 0.0


#: (champion, the family it is about, a predicate over that champion's named kit reading,
#: why). The magnitudes stay pinned in test_store_precedence; a row here claims the carry.
CLAIMS: tuple[tuple[str, str, Callable[[Mapping[str, float]], bool], str], ...] = (
    (
        "Zaahen",
        "armor_pen_pct",
        _has_armor_pen,
        "R declares 30% armor penetration in the module's kit row and the engine grants "
        "no stat for it, so the vendored row is the number's only home",
    ),
    (
        "Pantheon",
        "armor_pen_pct",
        _has_armor_pen,
        "R's armor penetration sits in the module's ASSUMPTIONS with no stat grant, so "
        "the engine prices none and the kit row carries it",
    ),
    (
        "Darius",
        "armor_pen_pct",
        _has_armor_pen,
        "E's innate 40% armor penetration IS a stat grant, so the engine prices it",
    ),
    (
        "Ambessa",
        "armor_pen_pct",
        _has_armor_pen,
        "R grants 30% armor penetration as a stat the engine prices",
    ),
    (
        "Annie",
        "magic_pen_pct",
        lambda kit: kit["magic_pen_pct"] > 0.0,
        "R grants 20% magic penetration as a stat the engine prices",
    ),
    (
        "Mordekaiser",
        "magic_pen_pct",
        lambda kit: kit["magic_pen_pct"] > 0.0,
        "E's kit row declares 15% magic penetration and the engine grants no stat",
    ),
    (
        "Aphelios",
        "lethality",
        lambda kit: kit["lethality"] > 0.0,
        "the passive brackets lethality by level, so the reference level reads 27 where "
        "rank-1 indexing read 4.5",
    ),
    (
        "Renekton",
        "armor_shred_pct",
        lambda kit: kit["armor_shred_pct"] > 0.0,
        "E shreds 35% armor, a kit row the fight engine prices no payload for",
    ),
    (
        "Corki",
        "magic_damage",
        lambda kit: (
            min(kit["physical_damage"], kit["magic_damage"])
            > 0.3 * (kit["physical_damage"] + kit["magic_damage"] + kit["true_damage"])
        ),
        "the hybrid ADC: Q and W are magic, E and R physical, so neither class carries "
        "under 30% of the cast cycle's typed damage (330 magic against 517.5 physical). "
        "The observed LeagueOfGraphs split is 0.651 physical to 0.208 magic, so the "
        "cast cycle, not the built champion, is what an even split can be claimed of",
    ),
)

#: (champion A, champion B, the family, why A reads above B on it).
ORDINALS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Lee Sin",
        "Malphite",
        "dash_casts",
        "Q's second cast and W both dash; Malphite's only dash is R",
    ),
    (
        "Leona",
        "Ezreal",
        "cc_hard_seconds",
        "a lockdown support against a poke ADC whose kit states no hard control",
    ),
    (
        "Thresh",
        "Jinx",
        "cc_displacement_casts",
        "the hook and the flay displace; Jinx's E roots in place instead",
    ),
    (
        "Caitlyn",
        "Darius",
        "attack_range",
        "the longest-ranged ADC against a melee juggernaut",
    ),
    (
        "Soraka",
        "Jinx",
        "heal",
        "the healer against an ADC whose kit heals nothing",
    ),
    (
        "Malphite",
        "Jinx",
        "armor",
        "a rock's armor at the reference level against a squishy ADC's",
    ),
    (
        "Annie",
        "Jinx",
        "magic_damage_share",
        "the observed split: a burst mage against a physical ADC",
    ),
)


def reading(store: AbilityFeatureStore, champion: str) -> dict[str, float]:
    """One champion's kit vector under its family names, in the families' own units."""
    return dict(zip(KIT_FAMILY_NAMES, kit_vector(store, champion), strict=True))


@pytest.mark.parametrize(
    ("champion", "key", "predicate", "why"), CLAIMS, ids=[f"{c}-{k}" for c, k, _, _ in CLAIMS]
)
def test_claim_holds_on_the_store(store, champion, key, predicate, why) -> None:
    """Every claim row reads the shipped store through the kit vector."""
    kit = reading(store, champion)
    assert predicate(kit), f"{champion} {key} reads {kit[key]}, and {why}"


@pytest.mark.parametrize(
    ("above", "below", "key", "why"), ORDINALS, ids=[f"{a}-over-{b}-{k}" for a, b, k, _ in ORDINALS]
)
def test_ordinal_holds_on_the_store(store, above, below, key, why) -> None:
    """Every ordinal row: champion A outreads champion B on one named family."""
    high, low = reading(store, above)[key], reading(store, below)[key]
    assert high > low, f"{above} reads {high} and {below} reads {low} on {key}, but {why}"


def test_the_reference_level_is_the_contracts() -> None:
    """The level the families read base stats at is the level the vendored kit prices."""
    assert load_reviewed_kit().level == REFERENCE_LEVEL


def test_a_family_sums_its_dim_over_the_slots(store) -> None:
    """The per-slot families are one cast of each slot, not a max or a single slot."""
    row = store.get("Leona")
    slot_sum = sum(row[f"{slot}_cc_hard_seconds"] for slot in ("P", "Q", "W", "E", "R"))
    assert reading(store, "Leona")["cc_hard_seconds"] == pytest.approx(slot_sum)


class TestKitMatrix:
    def test_unknown_row_is_exact_zeros(self, index, store) -> None:
        """The member's UNKNOWN convention: the corpus mean, which z-scoring puts at zero."""
        table = kit_matrix(index, store)
        assert table.shape == (len(index), len(KIT_FAMILIES))
        np.testing.assert_array_equal(table[index.UNKNOWN], 0.0)

    def test_an_unresolved_name_reads_the_corpus_mean(self, store) -> None:
        """A champion in the index the store cannot spell is absent, not an outlier."""
        index = ChampionIndex.from_names([*store.champions, "Nosuchchampion"])
        row = index.encode("Nosuchchampion")
        assert row != index.UNKNOWN
        np.testing.assert_array_equal(kit_matrix(index, store)[row], 0.0)

    def test_continuous_families_are_z_scored(self, index, store) -> None:
        """Every continuous family the corpus varies on reads mean 0, sd 1 over the known rows."""
        known = kit_matrix(index, store)[: index.UNKNOWN]
        standardised = np.array([f.standardise for f in KIT_FAMILIES.values()])
        varies = standardised & (known.std(axis=0) > 0.0)
        np.testing.assert_allclose(known[:, varies].mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(known[:, varies].std(axis=0), 1.0, atol=1e-4)

    def test_a_rare_family_keeps_its_own_units(self, index, store) -> None:
        """Four kits state armor penetration, so the column is the percent, not a z-score."""
        table = kit_matrix(index, store)
        column = KIT_FAMILY_NAMES.index("armor_pen_pct")
        assert table[index.encode("Darius"), column] == pytest.approx(40.0)
        assert table[index.encode("Caitlyn"), column] == 0.0

    def test_binary_flags_are_not_standardised(self, index, store) -> None:
        """A flag family stays 0 or 1: melee, global reach, and the rare-family indicators."""
        table = kit_matrix(index, store)[: index.UNKNOWN]
        for name in ("is_melee", "has_global_range", "has_armor_pen"):
            flags = table[:, KIT_FAMILY_NAMES.index(name)]
            assert set(np.unique(flags)) <= {0.0, 1.0}
        melee = table[:, KIT_FAMILY_NAMES.index("is_melee")]
        assert melee[index.encode("Malphite")] == 1.0
        assert melee[index.encode("Caitlyn")] == 0.0

    def test_a_global_ultimate_is_a_flag_not_a_distance(self, store) -> None:
        """Jinx R reads the 25000 sentinel, which would swamp every real reach axis."""
        kit = reading(store, "Jinx")
        assert kit["has_global_range"] == 1.0
        assert kit["ability_range"] < GLOBAL_RANGE
        assert reading(store, "Malphite")["has_global_range"] == 0.0

    def test_an_unspellable_champion_raises(self, store) -> None:
        """A misspelled name must fail a claim, never pass it on the store's zero row."""
        with pytest.raises(KeyError, match="Nosuchchampion"):
            kit_vector(store, "Nosuchchampion")
