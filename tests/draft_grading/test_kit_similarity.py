"""The kit-similarity floor: the store's kNN(3) same-class rate, chance, and a block pin."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pytest

from league_predictor.draft_grading.champion_attributes import classes
from league_predictor.draft_grading.champion_attributes.ability_store import (
    AbilityFeatureStore,
    load_ability_vectors,
)
from league_predictor.draft_grading.champion_attributes.classes import (
    effective_class,
)
from league_predictor.draft_grading.champion_attributes.kit_vector import kit_matrix
from league_predictor.draft_grading.grader.embeddings import ChampionIndex

#: The campaign plan's reading of the same two statistics, to two significant figures.
#: Its store rate is the prose-only store's, which `KIT_SIMILARITY_FLOOR` carries.
_PLAN_STORE_RATE = 0.118
_PLAN_CHANCE_RATE = 0.034
#: Both plan numbers are rounded to three decimals, so a half-count at n=510 is the floor
#: of any honest disagreement; 0.002 is that, doubled.
_PLAN_TOL = 0.002
#: The kit rate's regression allowance on a regenerated store. Its 510 draws are three
#: per champion off one matrix, so the true SE tops the binomial 0.014: one is noise.
_KIT_REGRESSION_TOL = 2 / 510


@pytest.fixture(scope="module")
def store_matrix(store: AbilityFeatureStore) -> np.ndarray:
    """The store in the form every consumer reads it: per-dimension z-scored rows."""
    vectors, _, _ = load_ability_vectors(store)
    return np.stack([vectors[name] for name in store.champions])


def test_store_rate_is_the_pinned_reading(
    store: AbilityFeatureStore, store_matrix: np.ndarray
) -> None:
    """The constant is definitionally the checked-in store's own reading, so it tracks it."""
    assert knn_class_rate(store.champions, store_matrix) == pytest.approx(STORE_KNN_CLASS_RATE)


def test_the_kit_vector_clears_the_floor(store: AbilityFeatureStore) -> None:
    """The dense named kit reads the classes better than the 360-dim store it is built from."""
    index = ChampionIndex(store)
    rate = knn_class_rate(store.champions, kit_matrix(index, store)[: index.UNKNOWN])
    assert rate > KIT_SIMILARITY_FLOOR
    assert rate >= KIT_KNN_CLASS_RATE - _KIT_REGRESSION_TOL
    assert rate > STORE_KNN_CLASS_RATE


def test_the_floor_is_the_plans_reading_of_the_prose_store() -> None:
    """The bar phase 2's kit vector must clear is the prose store's rate, not today's."""
    assert pytest.approx(_PLAN_STORE_RATE, abs=_PLAN_TOL) == KIT_SIMILARITY_FLOOR
    assert STORE_KNN_CLASS_RATE < KIT_SIMILARITY_FLOOR


def test_chance_rate_reproduces_the_plan(store: AbilityFeatureStore) -> None:
    """Under the class marginals the same statistic reads 0.033: the store beats it 3.1x."""
    chance = chance_class_rate(store.champions)
    assert chance == pytest.approx(_PLAN_CHANCE_RATE, abs=_PLAN_TOL)
    assert 3 * chance < STORE_KNN_CLASS_RATE


def test_standardisation_is_what_lifts_the_rate(store: AbilityFeatureStore) -> None:
    """In raw units the same read collapses toward chance: the z-scoring is load-bearing."""
    raw = np.stack([store.get_vector(n) for n in store.champions]).astype(np.float64)
    assert knn_class_rate(store.champions, raw) < 0.07


def test_block_structured_matrix_reads_one(store: AbilityFeatureStore) -> None:
    """A matrix whose classes are orthogonal blocks scores 1.0: every neighbour is in-class."""
    blocks: dict[str, list[str]] = defaultdict(list)
    for name in store.champions:
        role = primary_role(name)
        if role is not None:
            blocks[effective_class(name, role)].append(name)
    wide = sorted((label, names) for label, names in blocks.items() if len(names) > 3)
    names = [name for _, block in wide for name in block]
    matrix = np.zeros((len(names), len(wide)))
    row = 0
    for column, (_, block) in enumerate(wide):
        matrix[row : row + len(block), column] = 1.0
        row += len(block)
    assert knn_class_rate(names, matrix) == pytest.approx(1.0)


#: The checked-in store's own rate through `load_ability_vectors`: 53 in-class neighbours
#: over the 170 clustered champions' 3 each.
STORE_KNN_CLASS_RATE = 53 / 510

#: The prose-only store's rate, the bar phase 2's computed kit vector must exceed.
KIT_SIMILARITY_FLOOR = 60 / 510

#: The dense named kit vector's own rate over the same store and class table
#: (`kit_vector.kit_matrix`): 62 in-class neighbours, two clear of the floor's 60.
KIT_KNN_CLASS_RATE = 62 / 510


def primary_role(display_name: str) -> str | None:
    classes.ensure_loaded()
    return classes._PRIMARY_CLUSTERED_ROLE.get(classes.resolve_form(display_name))


def _same_class(names: Sequence[str]) -> tuple[list[int], np.ndarray]:
    """Query rows and the `(query, candidate)` same-class mask, read in each query's role."""
    same = np.zeros((len(names), len(names)), dtype=bool)
    queries: list[int] = []
    for i, name in enumerate(names):
        role = primary_role(name)
        if role is None:
            continue
        own = effective_class(name, role)
        same[i] = [effective_class(other, role) == own for other in names]
        queries.append(i)
    if not queries:
        raise ValueError("no champion in `names` is clustered by the class table")
    np.fill_diagonal(same, val=False)
    return queries, same


def knn_class_rate(names: Sequence[str], matrix: np.ndarray, k: int = 3) -> float:
    """Rate at which a champion's `k` nearest rows share its class in its primary role."""
    queries, same = _same_class(names)
    rows = matrix.astype(np.float64)
    centred = rows - rows.mean(axis=0)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    unit = centred / norms
    sims = unit @ unit.T
    np.fill_diagonal(sims, -np.inf)
    hits = sum(int(same[i, np.argsort(-sims[i], kind="stable")[:k]].sum()) for i in queries)
    return hits / (len(queries) * k)


def chance_class_rate(names: Sequence[str]) -> float:
    """`knn_class_rate` with the neighbour drawn uniformly: the class marginals, in closed form."""
    queries, same = _same_class(names)
    return int(same[queries].sum()) / (len(queries) * (len(names) - 1))
