"""The fitted-model protocol plus the side-agnostic prediction glue."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol


class FittedModel(Protocol):
    """A trained model: rows in (feature dicts), P(team_a wins) out."""

    def predict_proba(self, X: Sequence[Mapping[str, float | str]]) -> list[float]: ...


def predict_symmetric(
    fitted: FittedModel,
    X_ab: Sequence[Mapping[str, float | str]],
    X_ba: Sequence[Mapping[str, float | str]],
) -> list[float]:
    """Side-agnostic per-row P(team_a wins): ``0.5 * (P(a beats b) + (1 - P(b beats a)))``.

    Args:
        X_ab, X_ba: the two orientations of the same matchups, in the same row order.
    """
    ab = list(X_ab)
    both = fitted.predict_proba(ab + list(X_ba))
    p_ab = both[: len(ab)]
    p_ba = both[len(ab) :]
    return [0.5 * (a + (1.0 - b)) for a, b in zip(p_ab, p_ba, strict=True)]
