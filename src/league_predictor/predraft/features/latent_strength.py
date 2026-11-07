"""Registers learned_rating; rating_* is computed by skill_filter/serving_index.py::SkillFilterIndex."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING, ClassVar

from league_predictor.predraft.ratings.skill_filter import (
    BaggedSkillFilterBackend,
    derived_feature_names,
)

if TYPE_CHECKING:
    from league_predictor.asof import AsOf


class _LatentStrengthFamilyIndex:
    """Latent-strength features per (a, b, as_of): end-of-day states read at the query's day.

    A facts game reads its own fielded five and context; any other query the teams' last records.
    """

    def __init__(self, strength_index) -> None:
        self._strength_index = strength_index

    @property
    def rating_dependency(self):
        """The exact ordered rating bag consumed by this feature index."""
        return self._strength_index.rating_dependency

    @property
    def game_inputs(self) -> bool:
        """Whether the rating reads a facts game's own five and context."""
        return self._strength_index.game_inputs

    def features(self, team_a: str, team_b: str, as_of: AsOf) -> dict[str, float]:
        return self._strength_index.derived_features(team_a, team_b, as_of)

    def features_batch(
        self,
        queries: Sequence[tuple[str, str, AsOf]],
    ) -> list[dict[str, float]]:
        """Batched fast path: one wide torch forward, float-equivalent to per-row ``features``."""
        return self._strength_index.derived_features_batch(list(queries))


class LatentStrengthFamily:
    """Expose the skill filter's latent structure as matchup columns.

    Args:
        config: duck-typed on ``.d`` / ``.d_org``; fixes the emitted column set up front and
            must match the artifacts the backend or an externally supplied index carries.
        backend: needed ONLY to ``build`` a standalone index; a caller that supplies its own
            leak-safe index may leave it ``None``.
    """

    name = "learned_rating"
    categorical_names: ClassVar[list[str]] = []

    def __init__(self, config, backend: BaggedSkillFilterBackend | None = None) -> None:
        self._backend = backend
        self.feature_names = derived_feature_names(config.d, config.d_org)

    def build(
        self, conn: sqlite3.Connection, *, through: date, query_dates: Sequence[date]
    ) -> _LatentStrengthFamilyIndex:
        if self._backend is None:
            raise ValueError(
                "LatentStrengthFamily has no backend to build an index from; supply a leak-safe "
                "index externally, or construct it with a backend"
            )
        return _LatentStrengthFamilyIndex(
            self._backend.build_index(conn, through=through, query_dates=query_dates)
        )
