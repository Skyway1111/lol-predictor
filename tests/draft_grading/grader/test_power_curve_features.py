"""Tests for the power-curve block (draft_grading/grader/power_curve_features.py)."""

from __future__ import annotations

import numpy as np
import pytest

from league_predictor.draft_grading.champion_attributes.power_store import (
    BREAKPOINTS,
    CHANNELS,
    DEFAULT_POWER_CACHE,
    PowerGrid,
)
from league_predictor.draft_grading.grader import power_curve_features as pcf
from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from league_predictor.vocab.roles import ROLE_ORDER, ROLE_RANK

CHAMPS = ["Anivia", "Renekton", "Elise", "Nautilus"]
WIDTH = len(BREAKPOINTS) * len(CHANNELS)


def _index() -> ChampionIndex:
    return ChampionIndex.from_names(CHAMPS)


def _curve(start: float) -> np.ndarray:
    """A ``[4, 11]`` curve of consecutive integers, exact in float32."""
    return np.arange(start, start + WIDTH, dtype=np.float64).reshape(len(BREAKPOINTS), -1)


def _grid(borrowed=(), **cells: np.ndarray) -> PowerGrid:
    """A power grid from ``role__champion=curve`` keywords."""
    return PowerGrid(
        {tuple(key.split("__")): curve for key, curve in cells.items()}, frozenset(borrowed)
    )


# --- Layout ---


# --- The frozen table ---


def test_table_shape_order_and_the_zeros_around_one_cell():
    index = _index()
    tables = pcf.build_power_tables(index, _grid(mid__Anivia=_curve(1)))
    assert tables.table.shape == (len(index), len(ROLE_ORDER), len(BREAKPOINTS), len(CHANNELS))
    assert tables.table.dtype == np.float32
    np.testing.assert_array_equal(tables.table[index.encode("Anivia"), ROLE_RANK["mid"]], _curve(1))
    # Every other champion, every other role, and UNKNOWN read zero.
    filled = tables.table.copy()
    filled[index.encode("Anivia"), ROLE_RANK["mid"]] = 0.0
    np.testing.assert_array_equal(filled, 0.0)


def test_a_champion_outside_the_grid_and_a_grid_cell_outside_the_index_read_zeros():
    index = _index()
    tables = pcf.build_power_tables(
        index, _grid(top__Renekton=_curve(1), top__Sion=_curve(99), arena__Elise=_curve(50))
    )
    # Elise (no cell), UNKNOWN, and the out-of-universe Sion never reach the table.
    np.testing.assert_array_equal(tables.table[index.encode("Elise")], 0.0)
    np.testing.assert_array_equal(tables.table[index.UNKNOWN], 0.0)
    assert index.encode("Sion") == index.UNKNOWN
    assert float(tables.table.sum()) == float(_curve(1).sum())


def test_the_borrowed_plane_stamps_exactly_the_borrowed_cells():
    index = _index()
    tables = pcf.build_power_tables(
        index,
        _grid(borrowed=[("mid", "Anivia")], mid__Anivia=_curve(1), top__Renekton=_curve(2)),
    )
    assert tables.borrowed.shape == (len(index), len(ROLE_ORDER))
    assert tables.borrowed.dtype == np.bool_
    assert tables.borrowed[index.encode("Anivia"), ROLE_RANK["mid"]]
    assert not tables.borrowed[index.encode("Renekton"), ROLE_RANK["top"]]
    assert int(tables.borrowed.sum()) == 1


# --- The block each side reads ---


# --- Mirror: champion half-swap == block half-swap, bit for bit ---


@pytest.mark.slow
def test_the_prod_grid_freezes_one_table_of_the_pinned_size():
    """The production index's frozen table: (174, 5, 4, 11) float32, under 150 KiB."""
    from league_predictor.draft_grading.champion_attributes.ability_store import (
        AbilityFeatureStore,
    )
    from league_predictor.draft_grading.champion_attributes.power_store import load_power_grid

    if not DEFAULT_POWER_CACHE.is_file():
        pytest.skip("the extracted power cache is the fixture")
    index = ChampionIndex(AbilityFeatureStore())
    tables = pcf.build_power_tables(index, load_power_grid(DEFAULT_POWER_CACHE))
    assert tables.table.shape == (174, 5, 4, 11)
    assert tables.table.dtype == np.float32
    assert tables.table.nbytes < 150 * 1024
    # The grid's own count: 389 cells wear their build, 471 borrow one.
    assert int(tables.borrowed.sum()) == 471
