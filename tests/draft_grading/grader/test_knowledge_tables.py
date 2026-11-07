"""Tests for grader/knowledge_tables.py — the frozen per-family champion-knowledge member."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np
import pytest

from league_predictor.draft_grading.champion_attributes.ability_schema import SCHEMA_VERSION
from league_predictor.draft_grading.champion_attributes.ability_store import (
    AbilityFeatureStore,
    load_ability_vectors,
)
from league_predictor.draft_grading.champion_attributes.kit_vector import kit_matrix
from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from league_predictor.draft_grading.grader.knowledge_tables import (
    KNOWLEDGE_MEMBER_SUFFIX,
    build_knowledge_tables,
    knowledge_member_path,
    load_knowledge_tables,
    save_knowledge_tables,
)
from league_predictor.draft_grading.grader.power_curve_features import (
    PowerTables,
    power_table_shape,
)

CHAMPS = ["Ahri", "Jinx", "Thresh", "Zed"]


@pytest.fixture
def ability_store(tmp_path) -> AbilityFeatureStore:
    """A tiny ability store with distinct nonzero features per champion."""
    data = {
        "Ahri": {"attack_range": 550.0, "P_cc_hard_seconds": 1.0, "Q_base_damage": 300.0},
        "Jinx": {"attack_range": 525.0, "hp_base": 610.0, "R_base_damage": 500.0},
        "Thresh": {"attack_range": 450.0, "Q_cc_hard_seconds": 1.5, "E_shield_amount": 200.0},
        "Zed": {"attack_range": 125.0, "is_melee": 1.0, "W_has_dash": 1.0},
    }
    path = tmp_path / "abilities.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source_default": "prose",
                "champions": data,
                "sources": {},
            }
        ),
        encoding="utf-8",
    )
    return AbilityFeatureStore(path=path)


@pytest.fixture
def index() -> ChampionIndex:
    return ChampionIndex.from_names(CHAMPS)


def _meta_for(tables, sha: str, **flags) -> dict:
    """The family-meta subset load_knowledge_tables verifies against."""
    meta = {
        "champions": list(tables.champions),
        "knowledge_sha": sha,
        "knowledge_dim": len(tables.feature_names),
        "embed_dim": 8,
    }
    meta.update(flags)
    return meta


def _power_tables(index) -> PowerTables:
    """A well-formed frozen power view: one populated mid cell, borrowed."""
    table = np.zeros(power_table_shape(len(index)), dtype=np.float32)
    table[0, 2] = np.arange(table[0, 2].size, dtype=np.float32).reshape(table.shape[2:])
    borrowed = np.zeros((len(index), table.shape[1]), dtype=np.bool_)
    borrowed[0, 2] = True
    return PowerTables(table, borrowed)


def _drop_member_key(stem, key: str, meta: dict) -> None:
    """Rewrite the saved member without ``key`` and restamp the meta, so only that check fires."""
    path = knowledge_member_path(stem)
    with np.load(path, allow_pickle=False) as npz:
        arrays = {name: npz[name] for name in npz.files if name != key}
    with path.open("wb") as handle:
        np.savez(handle, **arrays)
    meta["knowledge_sha"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _with(table: np.ndarray, cell: tuple[int, ...], value: float) -> np.ndarray:
    """``table`` with one cell overwritten — the corruption each integrity check must catch."""
    broken = table.copy()
    broken[cell] = value
    return broken


def _saved(tmp_path, tables, **flags):
    """Save ``tables`` under a family stem and return (stem, verifying meta)."""
    stem = tmp_path / "draft"
    sha = save_knowledge_tables(stem, tables)
    return stem, _meta_for(tables, sha, **flags)


class TestBuild:
    def test_z_matrix_rows_are_store_z_vectors(self, index, ability_store):
        """Row i is champion i's per-feature z-scored vector; UNKNOWN row is exact zeros."""
        tables = build_knowledge_tables(index, ability_store)
        zvecs, _, _ = load_ability_vectors(ability_store)
        assert tables.champions == tuple(CHAMPS)
        assert tables.feature_names == tuple(ability_store.feature_names)
        assert tables.z_matrix.shape == (len(CHAMPS) + 1, len(ability_store.feature_names))
        assert tables.z_matrix.dtype == np.float32
        for i, name in enumerate(CHAMPS):
            np.testing.assert_array_equal(tables.z_matrix[i], zvecs[name])
        np.testing.assert_array_equal(tables.z_matrix[index.UNKNOWN], 0.0)

    def test_kit_matrix_is_the_named_kit_vector(self, index, ability_store):
        """kit=True freezes the dense named kit vector, a derived table like the tool matrix."""
        tables = build_knowledge_tables(index, ability_store, kit=True)
        np.testing.assert_array_equal(tables.kit, kit_matrix(index, ability_store))

    def test_z_vectors_view(self, index, ability_store):
        """z_vectors() is the name→row dict the axis/lane-head consumers eat."""
        tables = build_knowledge_tables(index, ability_store)
        view = tables.z_vectors()
        assert set(view) == set(CHAMPS)
        np.testing.assert_array_equal(view["Thresh"], tables.z_matrix[CHAMPS.index("Thresh")])


class TestRoundtrip:
    def test_save_load_roundtrip(self, tmp_path, index, ability_store):
        original = build_knowledge_tables(index, ability_store)
        stem, meta = _saved(
            tmp_path, original, lane_pair_features=False, attribute_embeddings=False
        )
        assert knowledge_member_path(stem).name == f"draft{KNOWLEDGE_MEMBER_SUFFIX}"
        loaded = load_knowledge_tables(stem, meta)
        assert loaded.champions == original.champions
        assert loaded.feature_names == original.feature_names
        np.testing.assert_array_equal(loaded.z_matrix, original.z_matrix)

    def test_dotted_stem_member_is_name_append(self, tmp_path, index, ability_store):
        """A dotted side stem (draft.tools) keeps its full name — never with_suffix."""
        tables = build_knowledge_tables(index, ability_store)
        stem = tmp_path / "draft.tools"
        save_knowledge_tables(stem, tables)
        assert (tmp_path / f"draft.tools{KNOWLEDGE_MEMBER_SUFFIX}").is_file()


class TestIntegrity:
    def test_missing_member_names_the_backfill_tool(self, tmp_path, index, ability_store):
        tables = build_knowledge_tables(index, ability_store)
        stem, meta = _saved(tmp_path, tables)
        knowledge_member_path(stem).unlink()
        with pytest.raises(RuntimeError, match="retrain draft"):
            load_knowledge_tables(stem, meta)

    def test_bit_flip_fails_the_sha(self, tmp_path, index, ability_store):
        tables = build_knowledge_tables(index, ability_store)
        stem, meta = _saved(tmp_path, tables)
        path = knowledge_member_path(stem)
        raw = bytearray(path.read_bytes())
        raw[len(raw) // 2] ^= 0xFF
        path.write_bytes(bytes(raw))
        with pytest.raises(RuntimeError, match="sha"):
            load_knowledge_tables(stem, meta)

    def test_cross_family_member_is_refused(self, tmp_path, index, ability_store):
        """Family A's member under family B's meta (different universe) raises."""
        tables_a = build_knowledge_tables(index, ability_store)
        stem, meta = _saved(tmp_path, tables_a)
        meta["champions"] = ["Jinx", "Ahri", "Zed", "Thresh"]  # same set, other order
        with pytest.raises(RuntimeError, match="champion"):
            load_knowledge_tables(stem, meta)

    def test_nonzero_unknown_row_is_refused(self, tmp_path, index, ability_store):
        tables = build_knowledge_tables(index, ability_store)
        z = tables.z_matrix.copy()
        z[-1] = 1.0
        stem, meta = _saved(tmp_path, dataclasses.replace(tables, z_matrix=z))
        with pytest.raises(RuntimeError, match="UNKNOWN"):
            load_knowledge_tables(stem, meta)

    def test_non_finite_z_is_refused(self, tmp_path, index, ability_store):
        tables = build_knowledge_tables(index, ability_store)
        z = tables.z_matrix.copy()
        z[0, 0] = np.nan
        stem, meta = _saved(tmp_path, dataclasses.replace(tables, z_matrix=z))
        with pytest.raises(RuntimeError, match="finite"):
            load_knowledge_tables(stem, meta)

    def test_knowledge_dim_mismatch_is_refused(self, tmp_path, index, ability_store):
        tables = build_knowledge_tables(index, ability_store)
        stem, meta = _saved(tmp_path, tables)
        meta["knowledge_dim"] = 999
        with pytest.raises(RuntimeError, match="knowledge_dim"):
            load_knowledge_tables(stem, meta)

    def test_power_tables_roundtrip_and_their_presence_must_match_the_flag(
        self, tmp_path, index, ability_store
    ):
        power = _power_tables(index)
        with_power = build_knowledge_tables(index, ability_store, power=power)
        stem, meta = _saved(tmp_path, with_power, power_features=True)
        loaded = load_knowledge_tables(stem, meta)
        assert loaded.power is not None
        np.testing.assert_array_equal(loaded.power.table, power.table)
        np.testing.assert_array_equal(loaded.power.borrowed, power.borrowed)

        meta.pop("power_features")
        with pytest.raises(RuntimeError, match="power_table presence"):
            load_knowledge_tables(stem, meta)

    def test_a_table_without_its_borrowed_plane_is_refused(self, tmp_path, index, ability_store):
        tables = build_knowledge_tables(index, ability_store, power=_power_tables(index))
        stem, meta = _saved(tmp_path, tables, power_features=True)
        _drop_member_key(stem, "power_borrowed", meta)
        with pytest.raises(RuntimeError, match="power_table pairing"):
            load_knowledge_tables(stem, meta)

    def test_corrupt_power_members_are_refused(self, tmp_path, index, ability_store):
        power = _power_tables(index)
        cases = {
            "power_table shape": PowerTables(power.table[:, :-1], power.borrowed),
            "power_table finite": PowerTables(
                _with(power.table, (0, 2, 0, 0), np.nan), power.borrowed
            ),
            "power_table UNKNOWN": PowerTables(
                _with(power.table, (index.UNKNOWN, 0, 0, 0), 1.0), power.borrowed
            ),
            "power_borrowed shape": PowerTables(power.table, power.borrowed[:, :-1]),
            "power_borrowed UNKNOWN": PowerTables(
                power.table, _with(power.borrowed, (index.UNKNOWN, 0), value=True)
            ),
        }
        for i, (check, broken) in enumerate(cases.items()):
            tables = build_knowledge_tables(index, ability_store, power=broken)
            stem, meta = _saved(tmp_path / f"power{i}", tables, power_features=True)
            with pytest.raises(RuntimeError, match=check):
                load_knowledge_tables(stem, meta)

    def test_kit_matrix_roundtrips_and_its_presence_must_match_the_flag(
        self, tmp_path, index, ability_store
    ):
        with_kit = build_knowledge_tables(index, ability_store, kit=True)
        stem, meta = _saved(tmp_path, with_kit, kit_features=True)
        np.testing.assert_array_equal(load_knowledge_tables(stem, meta).kit, with_kit.kit)

        meta.pop("kit_features")
        with pytest.raises(RuntimeError, match="kit_matrix presence"):
            load_knowledge_tables(stem, meta)

    def test_corrupt_kit_members_are_refused(self, tmp_path, index, ability_store):
        built = build_knowledge_tables(index, ability_store, kit=True)
        assert built.kit is not None
        cases = {
            "kit_matrix shape": built.kit[:, :-1].copy(),
            "kit_matrix finite": _with(built.kit, (0, 0), np.nan),
            "kit_matrix UNKNOWN": _with(built.kit, (index.UNKNOWN, 0), 1.0),
        }
        for i, (check, broken) in enumerate(cases.items()):
            tables = dataclasses.replace(built, kit=broken)
            stem, meta = _saved(tmp_path / f"kit{i}", tables, kit_features=True)
            with pytest.raises(RuntimeError, match=check):
                load_knowledge_tables(stem, meta)

    def test_unstamped_meta_with_member_is_refused(self, tmp_path, index, ability_store):
        tables = build_knowledge_tables(index, ability_store)
        stem, meta = _saved(tmp_path, tables)
        meta.pop("knowledge_sha")
        with pytest.raises(RuntimeError, match="knowledge_sha"):
            load_knowledge_tables(stem, meta)
