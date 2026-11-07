"""Shared-leaf artifact tests use temporary model slots."""

import pytest

from league_predictor import config


@pytest.fixture(autouse=True)
def _isolated_models_dir(request, tmp_path, monkeypatch):
    if request.node.get_closest_marker("slow"):
        return
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
