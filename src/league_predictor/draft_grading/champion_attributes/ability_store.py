"""Serve ability-level champion feature vectors from a pre-computed artifact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from league_predictor.draft_grading.champion_attributes.ability_schema import (
    ALL_FEATURE_KEYS,
    SCHEMA_VERSION,
)
from league_predictor.draft_grading.champion_attributes.normalize import (
    _CHAMPION_NAME_ALIASES,
    normalize_champion_name,
)

_MODULE_DIR = Path(__file__).resolve().parent
_DEFAULT_PATH = _MODULE_DIR / "data" / "ability_features.json"

#: A feature this flat is constant across champions; dividing by it would amplify noise.
_DEGENERATE_STD = 1e-8


def z_score_columns(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """THE per-column standardisation: ``(z, mean, std)``; a flat column divides by 1, staying 0."""
    mean, std = rows.mean(axis=0), rows.std(axis=0)
    std[std < _DEGENERATE_STD] = 1.0
    return (rows - mean) / std, mean, std


class AbilityFeatureStore:
    """Load and serve ability-level champion feature vectors."""

    def __init__(self, path: Path | None = None):
        self._path = path or _DEFAULT_PATH
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{self._path} is schema {payload.get('schema_version')!r}, not "
                f"{SCHEMA_VERSION}. Regenerate it with champion_attributes/scripts/"
                "generate_abilities.py."
            )
        self._data: dict[str, dict[str, float]] = payload["champions"]
        self._champion_names = sorted(self._data.keys())
        self._lookup: dict[str, str] = {}
        for name in self._champion_names:
            self._lookup[normalize_champion_name(name)] = name
        for alias, target in _CHAMPION_NAME_ALIASES.items():
            if target in self._data:
                self._lookup[normalize_champion_name(alias)] = target
        self._defaults: dict[str, float] = dict.fromkeys(ALL_FEATURE_KEYS, 0.0)

        self._default_vector = np.zeros(len(ALL_FEATURE_KEYS), dtype=np.float32)
        self._vector_cache: dict[str, np.ndarray] = {}
        ordered_keys = list(ALL_FEATURE_KEYS)
        for name, features in self._data.items():
            vec = np.array(
                [float(features.get(k, 0.0)) for k in ordered_keys],
                dtype=np.float32,
            )
            vec.flags.writeable = False
            self._vector_cache[name] = vec

    def resolve_name(self, champion: str) -> str | None:
        if champion in self._data:
            return champion
        return self._lookup.get(normalize_champion_name(champion))

    @property
    def champions(self) -> list[str]:
        return list(self._champion_names)

    @property
    def feature_names(self) -> list[str]:
        return list(ALL_FEATURE_KEYS)

    def get(self, champion: str) -> dict[str, float]:
        resolved = self.resolve_name(champion)
        if resolved is None:
            return dict(self._defaults)
        return dict(self._data[resolved])

    def get_vector(self, champion: str) -> np.ndarray:
        """Return the pre-built ``ALL_FEATURE_KEYS``-wide feature vector (read-only)."""
        resolved = self.resolve_name(champion)
        if resolved is None:
            return self._default_vector.copy()
        return self._vector_cache[resolved]

    def __len__(self) -> int:
        return len(self._champion_names)


def load_ability_vectors(
    store: AbilityFeatureStore,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """``(vectors, mean, std)`` per-feature z-scored; a missing champion reads as the mean."""
    raw = {name: store.get_vector(name) for name in store.champions}
    z, mean, std = z_score_columns(np.stack(list(raw.values())))
    return dict(zip(raw, z, strict=True)), mean, std
