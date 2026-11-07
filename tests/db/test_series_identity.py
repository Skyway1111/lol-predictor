"""Tests for deterministic database series identities."""

from __future__ import annotations

from league_predictor.db.series_identity import mint_series_id


def test_mint_is_deterministic_and_prefixed():
    a = mint_series_id("2026-07-04", "oe:t1", "oe:t2", 0)
    b = mint_series_id("2026-07-04", "oe:t1", "oe:t2", 0)
    assert a == b
    assert a.startswith("ser_")
    assert len(a) == len("ser_") + 16


def test_day_pair_and_seq_each_change_the_id():
    base = mint_series_id("2026-07-04", "oe:t1", "oe:t2", 0)
    assert mint_series_id("2026-07-05", "oe:t1", "oe:t2", 0) != base  # day
    assert mint_series_id("2026-07-04", "oe:t1", "oe:t3", 0) != base  # pair
    assert mint_series_id("2026-07-04", "oe:t1", "oe:t2", 1) != base  # seq


def test_orientation_is_carried_by_the_caller():
    """The mint takes team_a/team_b positionally: swapping them yields a different id."""
    assert mint_series_id("2026-07-04", "oe:t1", "oe:t2", 0) != (
        mint_series_id("2026-07-04", "oe:t2", "oe:t1", 0)
    )


def test_mint_formula_is_pinned():
    """'ser_' + sha1('day|team_a|team_b|seq')[:16] is pinned: stored ids depend on it."""
    import hashlib

    key = "2026-07-04|oe:t1|oe:t2|0"
    expected = "ser_" + hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]
    assert mint_series_id("2026-07-04", "oe:t1", "oe:t2", 0) == expected
