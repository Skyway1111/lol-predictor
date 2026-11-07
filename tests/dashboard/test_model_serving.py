"""Warm serving handles reload when committed facts change under the same artifact."""

from datetime import date

from league_predictor import db
from league_predictor.dashboard import model_serving


def test_draft_handle_reloads_after_committed_fact_update(tmp_path, monkeypatch):
    path = tmp_path / "facts.db"
    writer = db.get_connection(path)
    db.create_schema(writer)
    artifact = tmp_path / "draft.pt"
    artifact.touch()
    monkeypatch.setattr(model_serving.artifacts, "primary_path", lambda *_: artifact)
    loaded = []

    def load(*_args, **_kwargs):
        engine = object()
        loaded.append(engine)
        return engine

    monkeypatch.setattr(model_serving.DraftSuggestEngine, "load", load)
    scorer = model_serving.DraftScorer(path)
    first = scorer._loaded_engine()
    assert scorer._loaded_engine() is first
    writer.execute("INSERT INTO teams (team_id, name) VALUES ('A', 'Alpha')")
    writer.commit()
    assert scorer._loaded_engine() is not first
    assert len(loaded) == 2
    scorer._conn.close()
    writer.close()


def test_served_handle_rearms_after_committed_fact_update(tmp_path, monkeypatch):
    from league_predictor.predraft.pipeline import live, serve

    path = tmp_path / "facts.db"
    writer = db.get_connection(path)
    db.create_schema(writer)
    artifact = tmp_path / "served.pt"
    artifact.touch()
    monkeypatch.setattr(model_serving.artifacts, "primary_path", lambda *_: artifact)
    monkeypatch.setattr(live, "load_live_oracle", lambda *_: (object(), {}))
    monkeypatch.setattr(serve, "predict", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(model_serving.threading.Thread, "start", lambda _self: None)
    oracle = model_serving.ServedOracle(path)
    oracle._build()
    assert oracle.predict_series("A", "B", date(2026, 1, 1)) is not None
    writer.execute("INSERT INTO teams (team_id, name) VALUES ('A', 'Alpha')")
    writer.commit()
    assert oracle.predict_series("A", "B", date(2026, 1, 1)) is None
    assert oracle.status == "warming"
    oracle._conn.close()
    writer.close()
