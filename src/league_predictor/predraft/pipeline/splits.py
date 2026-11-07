"""The canonical walk-forward train/val/test cutoffs, half-open and single-sourced."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class Split:
    """Temporal boundaries; each is the exclusive upper bound of its segment."""

    train_end: date
    val_end: date
    test_end: date


def default_split() -> Split:
    """The canonical walk-forward split: fit span, early-stop holdout, final test window."""
    return Split(
        train_end=date(2026, 4, 15),
        val_end=date(2026, 5, 9),
        test_end=date(2026, 5, 31),
    )
