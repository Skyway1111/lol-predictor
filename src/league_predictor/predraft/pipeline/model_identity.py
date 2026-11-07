"""The registry row of the live rating artifact."""

from __future__ import annotations

from league_predictor import artifacts
from league_predictor.db.rows import ModelRow


def current_rating_model_id() -> str:
    """``rating:<sha12>`` of the live rating artifact."""
    return model_row("rating").model_id


def model_row(kind: str) -> ModelRow:
    """The ``models`` row for ``kind``'s live artifact."""
    model_id, sha = artifacts.mint(kind)
    return ModelRow(model_id=model_id, kind=kind, artifact_sha=sha)
