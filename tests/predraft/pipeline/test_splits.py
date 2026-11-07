"""Tests for the single temporal train/val/test boundary set."""

from __future__ import annotations

from league_predictor.predraft.pipeline.splits import default_split


def test_default_split_is_strictly_monotone():
    """train_end < val_end < test_end -- the walk-forward ordering every stage reads."""
    s = default_split()
    assert s.train_end < s.val_end < s.test_end
