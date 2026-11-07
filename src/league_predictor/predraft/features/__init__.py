"""Define feature-family contracts and assemble leak-safe matchup features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from league_predictor.asof import AsOf

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Mapping, Sequence
    from datetime import date

    import numpy as np


class FamilyIndex(Protocol):
    """Immutable result of one family's replay; answers any matchup at a query point."""

    def features(self, team_a: str, team_b: str, as_of: AsOf) -> dict[str, float | str]: ...


class FeatureFamily(Protocol):
    """One signal source: a name, the columns it emits, and a builder for its index.

    Args:
        query_dates: the as-of dates the index must answer; a family with a per-as-of refit
            cost uses it to bound its work, cheap families ignore it.

    Attributes:
        categorical_names: the subset of ``feature_names`` a native-categorical backend treats
            as categories; empty for a pure numeric/diff family.
    """

    name: str
    feature_names: list[str]
    categorical_names: list[str]

    def build(
        self, conn: sqlite3.Connection, *, through: date, query_dates: Sequence[date]
    ) -> FamilyIndex: ...


def _queries(queries: Sequence[tuple[str, str, AsOf | date]]) -> list[tuple[str, str, AsOf]]:
    """The public batch shape normalized: a bare date is the start of that day."""
    return [(a, b, AsOf.of(d)) for a, b, d in queries]


# sightline-ok: 14 - the matchup's home is build_features_batch's (team_a, team_b, as_of) tuple
def build_features(
    team_a: str,
    team_b: str,
    as_of: AsOf | date,
    *,
    indexes: Mapping[str, FamilyIndex],
    families: Sequence[str],
) -> dict[str, float | str]:
    """Union enabled feature families for one matchup; bulk callers use the batch twin."""
    as_of = AsOf.of(as_of)
    out: dict[str, float | str] = {}
    for name in families:
        out.update(indexes[name].features(team_a, team_b, as_of))

    return out


def build_features_batch(
    queries: Sequence[tuple[str, str, AsOf | date]],
    *,
    indexes: Mapping[str, FamilyIndex],
    families: Sequence[str],
) -> list[dict[str, float | str]]:
    """The batched twin of ``build_features``: one fresh dict per input matchup query.

    Args:
        indexes: each family yields its per-query dicts through its ``features_batch`` fast
            path when it has one, otherwise through a per-row ``features`` loop.

    Returns:
        Rows float-equivalent to per-query ``build_features``; the caller owns each dict, so
        none aliases a family's internal cache.
    """
    queries = _queries(queries)
    per_family: list[list[dict[str, float | str]]] = []
    for name in families:
        index = indexes[name]
        batch = getattr(index, "features_batch", None)
        per_family.append(
            batch(queries)
            if batch is not None
            else [index.features(a, b, d) for a, b, d in queries]
        )
    out: list[dict[str, float | str]] = []
    for i in range(len(queries)):
        row: dict[str, float | str] = {}
        for family_rows in per_family:
            row.update(family_rows[i])
        out.append(row)
    return out


def build_feature_matrix_batch(
    queries: Sequence[tuple[str, str, AsOf | date]],
    *,
    indexes: Mapping[str, FamilyIndex],
    families: Sequence[FeatureFamily],
    drop_features: frozenset[str] = frozenset(),
) -> np.ndarray:
    """Build the all-numeric feature matrix without materializing merged row dictionaries.

    Args:
        indexes: a family may expose ``features_matrix(queries, kept_names)`` to write from its
            native storage; otherwise its batch/per-row dict contract is converted.

    Returns:
        A float64 matrix in family/feature order.

    Raises:
        ValueError: a fast path returned a block of the wrong shape (never a silent zero/NaN).
    """
    import numpy as np

    queries = _queries(queries)
    kept_by_family = [
        [name for name in family.feature_names if name not in drop_features] for family in families
    ]
    width = sum(len(names) for names in kept_by_family)
    matrix = np.empty((len(queries), width), dtype=np.float64)
    offset = 0
    for family, names in zip(families, kept_by_family, strict=True):
        if not names:
            continue
        index = indexes[family.name]
        matrix_builder = getattr(index, "features_matrix", None)
        if matrix_builder is not None:
            block = np.asarray(matrix_builder(queries, names), dtype=np.float64)
        else:
            batch_builder = getattr(index, "features_batch", None)
            rows = (
                batch_builder(queries)
                if batch_builder is not None
                else [index.features(a, b, d) for a, b, d in queries]
            )
            block = np.fromiter(
                (float(row[name]) for row in rows for name in names),
                dtype=np.float64,
                count=len(queries) * len(names),
            ).reshape(len(queries), len(names))
        expected = (len(queries), len(names))
        if block.shape != expected:
            raise ValueError(
                f"{family.name} features_matrix returned shape {block.shape}, expected {expected}"
            )
        matrix[:, offset : offset + len(names)] = block
        offset += len(names)
    return matrix


def lazy_family(module: str, cls: str) -> Callable[[], FeatureFamily]:
    """Zero-arg factory for family class ``cls`` in ``module``, deferring that family's deps."""

    def make() -> FeatureFamily:
        from importlib import import_module

        return getattr(import_module(module), cls)()

    return make


def _served(module: str, cls: str) -> Callable[[], FeatureFamily]:
    return lazy_family(f"league_predictor.predraft.features.families.{module}", cls)


def _latent_strength() -> FeatureFamily:
    """Build the served skill-filter family: the SHA-pinned chosen bag as derived columns."""
    from types import SimpleNamespace

    # sightline-ok: 35 - deferred: a torch-free registry; the pinned artifact loads on build
    from league_predictor.predraft.features.latent_strength import LatentStrengthFamily
    from league_predictor.predraft.pipeline.chosen_skill_filter import chosen_skill_filter_backend
    from league_predictor.predraft.ratings.skill_filter.recipe import (
        LATENT_STRENGTH_D,
        LATENT_STRENGTH_D_ORG,
    )

    # LatentStrengthFamily/BaggedSkillFilterBackend duck-type the FeatureFamily/backend contracts;
    # latent_strength.py's annotations (backend, dict[str, float] index) are narrower than runtime.
    return LatentStrengthFamily(  # pyright: ignore[reportReturnType]
        SimpleNamespace(d=LATENT_STRENGTH_D, d_org=LATENT_STRENGTH_D_ORG),
        chosen_skill_filter_backend(),  # pyright: ignore[reportArgumentType]
    )


# name -> lazy zero-arg factory. Exactly the families the served model draws from
# (``chosen_model().families``); keys equal each family's ``.name``.
FEATURE_FAMILIES: dict[str, Callable[[], FeatureFamily]] = {
    # THE predraft rating -- the skill filter wrapped as its rich derived family
    # (20 cols at d=2/d_org=4).
    "learned_rating": _latent_strength,
    # The full pre-pruning team-strength feature set (features/snapshots, prune=False).
    # See features/families/original.py.
    "original": _served("original", "OriginalFeaturesFamily"),
    # Recent-form diffs a slow rating system lags behind.
    "recency": _served("recency", "RecencyFamily"),
    # Expected-fielded-lineup strength/continuity -- part of the served set.
    # See features/lineup.py::LineupFamily.
    "lineup": lazy_family("league_predictor.predraft.features.lineup", "LineupFamily"),
}
