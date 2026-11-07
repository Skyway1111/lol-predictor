"""Configuration resolves environment paths at import and remains patchable."""

import importlib.util
from pathlib import Path

from league_predictor import config


def test_environment_overrides_config_paths(monkeypatch, tmp_path):
    monkeypatch.setenv("LEAGUE_PREDICTOR_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEAGUE_PREDICTOR_DB_PATH", str(tmp_path / "custom.db"))
    monkeypatch.delenv("LEAGUE_PREDICTOR_RAW_DIR", raising=False)
    spec = importlib.util.spec_from_file_location("config_probe", config.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert tmp_path / "custom.db" == module.DB_PATH
    assert tmp_path / "raw" == module.RAW_DIR
    assert isinstance(module.MODELS_DIR, Path)
