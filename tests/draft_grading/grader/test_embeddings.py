"""Tests for grader/embeddings.py — champion index and ability-initialized embeddings."""

from __future__ import annotations

from league_predictor.draft_grading.champion_attributes.ability_store import AbilityFeatureStore
from league_predictor.draft_grading.grader.embeddings import ChampionIndex


class TestChampionIndex:
    def test_roundtrip(self, index: ChampionIndex, store: AbilityFeatureStore):
        """Index -> name -> index is identity for all known champions."""
        for name in store.champions:
            idx = index.encode(name)
            assert idx != index.UNKNOWN
            assert index.decode(idx) == name

    def test_resolves_aliases(self, index: ChampionIndex):
        """Common aliases resolve to valid indices, not UNKNOWN."""
        aliases = ["Nunu", "Ksante", "LeBlanc", "Kaisa", "Belveth"]
        for alias in aliases:
            idx = index.encode(alias)
            assert idx != index.UNKNOWN, f"alias {alias!r} resolved to UNKNOWN"

    def test_unknown_champion(self, index: ChampionIndex):
        """Unknown champion returns UNKNOWN sentinel."""
        assert index.encode("NotAChampion_XYZ") == index.UNKNOWN

    def test_len_includes_unknown(self, index: ChampionIndex, store: AbilityFeatureStore):
        """Length includes all champions + UNKNOWN sentinel."""
        assert len(index) == len(store.champions) + 1
