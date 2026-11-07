"""The format-4 family on disk: the three members, the round trip, the id, the dispatch."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from league_predictor import artifacts, config
from league_predictor.draft_grading.grader.artifact import family_meta_path
from league_predictor.draft_grading.grader.set_artifact import (
    SET_FORMAT,
    SET_WEIGHT_KEYS,
    checked_weights,
    load_set_meta,
    set_family_files,
)
from league_predictor.draft_grading.live import load_grader
from tests.draft_grading._set_helpers import (
    RECIPE,
    THRESHOLDS,
    StubStats,
    patch_wiring,
    stub_draft,
    stub_family,
    stub_index,
    stub_weights,
    write_set_family,
)

STATS = StubStats()


@pytest.fixture(autouse=True)
def _stub_wiring(monkeypatch):
    """Every load in this file rebuilds the stub encoder, never 4b's production pair."""
    patch_wiring(monkeypatch)


def _meta(root):
    return json.loads(family_meta_path(root).read_text(encoding="utf-8"))


# --- The three members and the top meta ----------------------------------------------


def test_save_writes_exactly_the_three_members(tmp_path):
    root = write_set_family(tmp_path / "draft")
    assert sorted(p.name for p in tmp_path.glob("draft.*")) == [
        "draft.encoder.pt",
        "draft.json",
        "draft.knowledge.npz",
    ]
    meta = _meta(root)
    assert meta["format"] == SET_FORMAT
    assert meta["recipe"] == RECIPE
    assert meta["grade_semantics"] == "edge"
    assert meta["thresholds"] == list(THRESHOLDS)
    assert (meta["seeds"], meta["d_in"]) == (3, 4)
    assert meta["knowledge_dim"] == 2
    assert meta["edge_cuts_pp"] == [8.0, 4.0]


def test_save_sweeps_an_undeclared_stray(tmp_path):
    stray = tmp_path / "draft.stale.bin"
    tmp_path.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"junk")
    write_set_family(tmp_path / "draft")
    assert not stray.exists()  # a stale sibling would corrupt the content id


def test_the_weights_member_reloads_under_weights_only(tmp_path):
    root = write_set_family(tmp_path / "draft")
    state = torch.load(root.with_name("draft.encoder.pt"), weights_only=True)
    assert sorted(state) == sorted(SET_WEIGHT_KEYS)
    assert len(state["states"]) == 3


def test_a_payload_that_is_not_weights_only_is_refused_at_save(tmp_path):
    """A numpy array in the payload loads today and fails in production; refuse it here."""
    family = stub_family()
    family._weights["history"] = {"loss": np.zeros(3)}
    with pytest.raises(ValueError, match="weights_only"):
        family.save(tmp_path / "draft")


def test_a_weights_payload_missing_its_column_vocabulary_is_refused():
    state = stub_weights()
    state["input_spec"] = {"columns": ["kit_0"]}
    with pytest.raises(ValueError, match="input_spec"):
        checked_weights(state)
    del state["states"]
    with pytest.raises(ValueError, match="missing"):
        checked_weights(state)


# --- The round trip -------------------------------------------------------------------


def test_the_family_round_trips_to_the_same_served_numbers(tmp_path):
    """On one device the round trip is bitwise; across devices float32 moves it by ~1e-8."""
    index = stub_index()
    root = write_set_family(tmp_path / "draft", index)
    before = stub_family(index).predict(stub_draft(), stats_cache=STATS)
    loaded = load_grader(None, artifact=root, device="cpu")
    after = loaded.predict(stub_draft(), stats_cache=STATS)
    assert after.prob_blue_wins == before.prob_blue_wins
    assert (after.grade_blue, after.edge_pp_blue) == (before.grade_blue, before.edge_pp_blue)
    assert loaded.artifact_stem == "draft"
    assert loaded.temperature == 1.3
    assert loaded.k == RECIPE["k"]
    assert loaded.index.known_champions == index.known_champions


def test_a_tampered_knowledge_member_is_refused(tmp_path):
    root = write_set_family(tmp_path / "draft")
    member = root.with_name("draft.knowledge.npz")
    member.write_bytes(member.read_bytes() + b"tampered")
    with pytest.raises(RuntimeError, match="sha256"):
        load_grader(None, artifact=root)


def test_a_meta_missing_a_served_field_is_refused(tmp_path):
    root = write_set_family(tmp_path / "draft")
    meta = _meta(root)
    del meta["temperature"]
    family_meta_path(root).write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing required metadata"):
        load_set_meta(root)


# --- The content id over the three files ----------------------------------------------


def test_the_minted_id_covers_every_member(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    root = write_set_family(tmp_path / "draft" / "draft")
    assert [p.name for p in artifacts.family_paths("draft")] == [
        p.name for p in set_family_files(root)
    ]
    before, _sha = artifacts.mint("draft")
    for member in set_family_files(root):
        original = member.read_bytes()
        # The meta stays parseable (the enumerator reads it) and still moves the id.
        moved = json.dumps({**_meta(root), "note": "moved"}) if member.suffix == ".json" else b"m"
        member.write_bytes(moved.encode() if isinstance(moved, str) else original + moved)
        assert artifacts.mint("draft")[0] != before, member.name
        member.write_bytes(original)
    assert artifacts.mint("draft")[0] == before


def test_a_stray_sibling_never_enters_the_id(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    root = write_set_family(tmp_path / "draft" / "draft")
    clean = artifacts.mint("draft")
    root.with_name("draft.json.4242.swap.staged").write_bytes(b"leftover temp")
    assert artifacts.mint("draft") == clean
