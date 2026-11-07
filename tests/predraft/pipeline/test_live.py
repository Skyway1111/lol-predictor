"""Live served-model loading through the public predraft port."""

from pathlib import Path


def test_load_live_oracle_uses_explicit_artifact_and_builds_indexes(monkeypatch):
    from league_predictor.predraft.pipeline import live

    artifact = Path("models/served/served.json")
    model = object()
    indexes = object()
    conn = object()
    dependency = object()
    loaded = []

    monkeypatch.setattr(live, "chosen_rating_dependency", lambda: dependency)
    monkeypatch.setattr(
        live.ServedModel,
        "load",
        lambda path, *, actual_rating_dependency: (
            loaded.append((path, actual_rating_dependency)) or model
        ),
    )
    monkeypatch.setattr(live, "build_live_indexes", lambda *_: indexes)

    assert live.load_live_oracle(conn, artifact) == (model, indexes)
    assert loaded == [(artifact, dependency)]
