"""Public served-model operation ports."""

from datetime import date
from types import SimpleNamespace

import pytest


def test_fit_current_model_owns_locked_recipe_context(monkeypatch):
    from league_predictor.predraft.pipeline import operations

    context = object()
    model = object()
    calls = []

    monkeypatch.setattr(
        operations,
        "build_context",
        lambda _conn, *, through, months: context,  # noqa: ARG005 - mirrors build_context's keywords
    )

    def fake_fit(conn, *, through, ctx):
        calls.append((conn, through, ctx))
        return model

    monkeypatch.setattr(operations, "fit_served_model", fake_fit)

    conn = object()
    through = date(2026, 7, 13)
    result = operations.fit_current_model(conn, through=through)

    assert result is model
    assert calls == [(conn, through, context)]


# --- the pushed-down lifecycle surface: coverage guard + evaluation binding -------------


def _score(game_id: str, probability: float = 0.6):
    return SimpleNamespace(game_id=game_id, p_a=probability, asof_date="2026-07-01")


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.1, 1.1, True])
def test_complete_coverage_rejects_invalid_probabilities(probability):
    from league_predictor.predraft.pipeline import operations

    with pytest.raises(RuntimeError, match="invalid probability"):
        operations.require_complete_coverage("oos", [_score("g1", probability)], {"g1"})


def test_complete_coverage_rejects_duplicates():
    from league_predictor.predraft.pipeline import operations

    with pytest.raises(RuntimeError, match="duplicate"):
        operations.require_complete_coverage("oos", [_score("g1"), _score("g1")], {"g1"})


def test_complete_coverage_names_missing_and_unexpected_ids():
    from league_predictor.predraft.pipeline import operations

    with pytest.raises(RuntimeError, match=r"missing 1.*unexpected 1"):
        operations.require_complete_coverage("oos", [_score("g2")], {"g1"})


@pytest.mark.parametrize("temperature", [0.75, 1.25])
def test_cached_backfill_uses_the_artifact_formula(temperature):
    import numpy as np

    from league_predictor.asof import AsOf
    from league_predictor.predraft.pipeline import operations
    from league_predictor.predraft.pipeline.chosen_model import ServedModel
    from league_predictor.predraft.pipeline.serve import predict
    from league_predictor.predraft.pipeline.walk_forward import FeaturizedCorpus

    day = date(2026, 1, 1)
    as_of = AsOf.of(day)
    bag = SimpleNamespace(
        predict_proba=lambda rows: [0.8 if row["rating_logit_diff"] > 0 else 0.2 for row in rows]
    )
    model = ServedModel(
        bag, families=["rating"], feature_names=["rating_logit_diff"], rating_blend=0.7
    )
    model.temperature = temperature
    indexes = {
        "rating": SimpleNamespace(
            features=lambda a, b, _when: {"rating_logit_diff": 1.0 if a < b else -1.0}
        )
    }
    corpus = FeaturizedCorpus._from_parts(
        matrix=np.array([[1.0], [-1.0]]),
        y=[1, 0],
        meta=[{"match_id": "g", "date": day}] * 2,
        feature_names=model.feature_names,
        drop_features=(),
        start=day,
        end=date(2026, 1, 2),
    )
    actual = operations._score_games_from_corpus(
        model,
        indexes,
        [{"game_id": "g", "team_a": "A", "team_b": "B"}],
        matchups=[("A", "B", as_of)],
        corpus=corpus,
    )
    assert actual == [predict(model, indexes, "A", "B", as_of=as_of)]
