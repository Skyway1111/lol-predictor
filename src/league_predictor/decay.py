"""Dependency-leaf exponential decay weights; half-lives are caller data."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    import numpy as np


@overload
def half_life_weight(age: float, half_life: float) -> float: ...


@overload
def half_life_weight(age: np.ndarray, half_life: float) -> np.ndarray: ...


def half_life_weight(age: float | np.ndarray, half_life: float) -> float | np.ndarray:
    """Return ``0.5 ** (age / half_life)``, elementwise for an array ``age``."""
    return 0.5 ** (age / half_life)


def half_life_rate(half_life: float) -> float:
    """Return ``log(2) / half_life`` — the rate of the equivalent ``exp(-rate * age)`` form."""
    return math.log(2) / half_life
