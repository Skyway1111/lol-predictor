"""The Served model exposes one immutable production recipe."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from league_predictor.predraft.pipeline import chosen_model as served


def test_fit_served_model_wires_the_fixed_recipe_and_dependency(monkeypatch):
    dependency = object()
    family = SimpleNamespace(name="learned_rating")
    index = SimpleNamespace(rating_dependency=dependency, game_inputs=True)
    bag = object()
    league = object()
    captured = {}

    context = SimpleNamespace(
        fams=[family], indexes={"learned_rating": index}, drop=frozenset(), corpus=object()
    )
    monkeypatch.setattr(served, "build_fit_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(served, "bag_backend", lambda: "backend")
    table = SimpleNamespace(with_betas=lambda *, cross_tier, within_tier1: league)  # noqa: ARG005 - mirrors RegionRatingTable.with_betas
    monkeypatch.setattr(served, "fit_region_ratings", lambda _conn, *, through: table)  # noqa: ARG005 - mirrors fit_region_ratings's keywords

    def fit_window(backend, **kwargs):
        captured.update(
            backend=backend,
            **kwargs,
        )
        return bag

    monkeypatch.setattr(served, "fit_window", fit_window)

    conn = object()
    through = date(2026, 7, 15)
    model = served.fit_served_model(conn, through=through)

    assert model.bag is bag
    assert model.league is league
    assert model.rating_dependency is dependency
    assert model.temperature == served.SERVED_TEMPERATURE
    assert model.rating_blend == served.chosen_model().rating_blend
    assert model.feature_names == list(served.chosen_model().features)
    assert captured["backend"] == "backend"
    assert captured["window"] == served.chosen_model().window


def test_an_undamped_recipe_refuses_a_rating_that_reads_the_last_record(monkeypatch):
    """Undamped facts games need the rating's game inputs; a day-path rating must not fit."""
    index = SimpleNamespace(rating_dependency=object(), game_inputs=False)
    context = SimpleNamespace(
        fams=[SimpleNamespace(name="learned_rating")],
        indexes={"learned_rating": index},
        drop=frozenset(),
        corpus=object(),
    )
    monkeypatch.setattr(served, "build_fit_context", lambda *_args, **_kwargs: context)
    monkeypatch.setattr(served, "bag_backend", lambda: "backend")
    monkeypatch.setattr(served, "fit_window", lambda *_args, **_kwargs: object())
    table = SimpleNamespace(with_betas=lambda *, cross_tier, within_tier1: None)  # noqa: ARG005 - mirrors RegionRatingTable.with_betas
    monkeypatch.setattr(served, "fit_region_ratings", lambda _conn, *, through: table)  # noqa: ARG005 - mirrors fit_region_ratings's keywords
    with pytest.raises(RuntimeError, match="retrain rating"):
        served.fit_served_model(object(), through=date(2026, 7, 15))
