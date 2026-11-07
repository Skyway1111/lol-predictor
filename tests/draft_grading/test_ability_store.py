"""Tests for AbilityFeatureStore."""

import numpy as np

from league_predictor.draft_grading.champion_attributes.ability_schema import ALL_FEATURE_KEYS


def test_store_loads(store):
    # The exact count is not a contract (the roster grows on regen); a substantial pool is.
    assert len(store) > 150, "expected a substantial champion pool"


def test_store_champions_sorted(store):
    names = store.champions
    assert names == sorted(names)


def test_store_get_vector_shape(store):
    vec = store.get_vector("Ornn")
    assert vec.shape == (len(ALL_FEATURE_KEYS),)
    assert vec.dtype == np.float32


def test_store_feature_names_match_vector(store):
    assert len(store.feature_names) == len(ALL_FEATURE_KEYS)
    vec = store.get_vector("Ornn")
    assert len(store.feature_names) == len(vec)


def test_store_unknown_champion_returns_defaults(store):
    vec = store.get_vector("NotAChampion")
    assert np.all(vec == 0.0)


def test_store_name_resolution(store):
    assert store.resolve_name("Nunu & Willump") is not None
    nunu_direct = store.get_vector("Nunu & Willump")
    nunu_alias = store.get_vector("nunu")
    np.testing.assert_array_equal(nunu_direct, nunu_alias)


def test_store_get_returns_all_keys(store):
    features = store.get("Ornn")
    assert set(features.keys()) == set(ALL_FEATURE_KEYS)
