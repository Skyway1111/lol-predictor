"""Served artifact contracts for the one fixed recipe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from league_predictor.predraft.pipeline.chosen_model import (
    SERVED_TEMPERATURE,
    ServedModel,
    chosen_model,
)
from league_predictor.predraft.ratings.skill_filter.artifact import RatingDependency

_REPO = Path(__file__).resolve().parents[3]


def _dependency(tag: str = "a") -> RatingDependency:
    return RatingDependency(tag * 64, 1, "b" * 64)


def _model(*, learned: bool = False) -> ServedModel:
    return ServedModel(
        _bag(["rating_logit_diff"] if learned else ["recency_winrate20_diff"]),
        families=["learned_rating"] if learned else ["recency"],
        feature_names=["rating_logit_diff"] if learned else ["recency_winrate20_diff"],
        rating_dependency=_dependency() if learned else None,
        rating_blend=0.7 if learned else 0.0,
        version="served_20260101_fixed",
    )


def test_save_writes_only_the_fixed_formula_contract(tmp_path):
    path = tmp_path / "served.json"
    _model(learned=True).save(path)

    meta = json.loads(path.read_text())
    assert meta["format"] == 3
    recipe = meta["recipe"]
    assert recipe["version"] == chosen_model().version
    assert recipe["temperature"] == SERVED_TEMPERATURE
    assert recipe["lgb_params"] == chosen_model().lgb_params
    assert recipe["families"] == ["learned_rating"]
    assert meta["rating_dependency"] == _dependency().to_meta()
    for legacy_key in ("training_config", "temperature", "families", "feature_names"):
        assert legacy_key not in meta
    assert len(list(tmp_path.glob("served.member*.lgb"))) == 1


def test_production_fit_stamp_equals_the_source_recipe(tmp_path):
    """A model that follows chosen_model() persists the recipe dict verbatim."""
    current = chosen_model()
    path = tmp_path / "served.json"
    ServedModel(
        _bag(current.features),
        families=current.families,
        feature_names=current.features,
        rating_dependency=_dependency(),
        rating_blend=current.rating_blend,
        series_lead=current.series_lead,
    ).save(path)

    assert json.loads(path.read_text())["recipe"] == current.to_dict()


def test_load_round_trips_fixed_formula_metadata(tmp_path):
    path = tmp_path / "served.json"
    _model(learned=True).save(path)

    loaded = ServedModel.load(path, actual_rating_dependency=_dependency())

    assert loaded.temperature == SERVED_TEMPERATURE
    assert loaded.rating_dependency == _dependency()
    assert loaded.rating_blend == 0.7
    assert loaded.families == ["learned_rating"]


def test_load_rejects_feature_schema_mismatch(tmp_path):
    path = tmp_path / "served.json"
    _model().save(path)
    meta = json.loads(path.read_text())
    meta["feature_schema_fingerprint"] = "deadbeefdeadbeef"
    path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match=r"[Rr]etrain served"):
        ServedModel.load(path)


def test_load_binds_the_persisted_artifact_temperature(tmp_path):
    path = tmp_path / "served.json"
    _model().save(path)
    meta = json.loads(path.read_text())
    meta["recipe"]["temperature"] = 1.0
    path.write_text(json.dumps(meta))

    loaded = ServedModel.load(path)

    assert loaded.temperature == 1.0


def test_load_rejects_a_recipeless_meta(tmp_path):
    """A format-3 meta without a ``recipe`` key never serves."""
    path = tmp_path / "served.json"
    _model().save(path)
    meta = json.loads(path.read_text())
    del meta["recipe"]
    path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match="no recipe"):
        ServedModel.load(path)


def test_load_rejects_rating_dependency_mismatch(tmp_path):
    path = tmp_path / "served.json"
    _model(learned=True).save(path)

    with pytest.raises(RuntimeError, match="rating dependency mismatch"):
        ServedModel.load(path, actual_rating_dependency=_dependency("c"))


def test_save_rejects_missing_dependency_for_learned_rating(tmp_path):
    model = ServedModel(
        _bag(["rating_logit_diff"]),
        families=["learned_rating"],
        feature_names=["rating_logit_diff"],
    )

    with pytest.raises(RuntimeError, match="rating_dependency"):
        model.save(tmp_path / "served.json")


def test_save_removes_unrecognized_family_siblings(tmp_path):
    stray = tmp_path / "served.unknown"
    stray.write_bytes(b"stale")
    _model().save(tmp_path / "served.json")
    assert not stray.exists()


def test_load_rejects_unsupported_format_meta(tmp_path):
    path = tmp_path / "served.json"
    _model().save(path)
    meta = json.loads(path.read_text())
    meta["format"] = 1
    path.write_text(json.dumps(meta))

    with pytest.raises(RuntimeError, match=r"(?s)format.*retrain served"):
        ServedModel.load(path)


@pytest.mark.slow
def test_live_production_artifact_loads_under_current_code():
    """The deployed family must load with its exact formula and rating dependency."""
    from league_predictor.predraft.pipeline.chosen_skill_filter import chosen_rating_dependency

    artifact = _REPO / "models" / "served" / "served.json"
    model = ServedModel.load(artifact, actual_rating_dependency=chosen_rating_dependency())
    assert model.temperature == pytest.approx(1.0381057133080291)
    assert model.rating_dependency == chosen_rating_dependency()
    assert len(model.bag._members) == 7
    assert tuple(model.feature_names) == chosen_model().features


def _bag(features):
    from league_predictor.predraft.pipeline.lightgbm_bag import LightGBMBagBackend, LightGBMParams

    return LightGBMBagBackend(LightGBMParams(n_estimators=1)).fit(
        [{name: float(i) for name in features} for i in range(4)],
        [0, 1, 0, 1],
        sample_weight=None,
        feature_names=features,
        categorical_names=[],
    )
