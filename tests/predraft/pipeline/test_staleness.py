"""Tests for the staleness dampener: pulls a prediction toward 0.5 on stale day-gaps."""

from __future__ import annotations

import math

from league_predictor.predraft.pipeline.staleness import (
    STALE_AFTER_DAYS,
    STALE_LOGIT_SCALE,
    dampen_if_stale,
)


def test_stale_team_halves_the_logit() -> None:
    """A team idle past the threshold pulls the prob toward 0.5 by scaling its logit."""
    p = 0.92
    expected = 1 / (1 + math.exp(-math.log(p / (1 - p)) * STALE_LOGIT_SCALE))
    assert dampen_if_stale(p, days_since_a=42.0, days_since_b=3.0) == expected
    assert 0.5 < dampen_if_stale(p, 42.0, 3.0) < p  # still favours the team, just humbler


def test_both_teams_current_is_identity() -> None:
    """Neither team idle past the threshold -> the prob is returned byte-identically."""
    for p in (0.92, 0.5, 0.13):
        assert dampen_if_stale(p, days_since_a=7.0, days_since_b=STALE_AFTER_DAYS) == p


def test_fires_on_the_more_stale_team() -> None:
    """Only one team need be stale: the guard keys off the MAX day-gap."""
    p = 0.80
    only_b_stale = dampen_if_stale(p, days_since_a=2.0, days_since_b=99.0)
    assert only_b_stale < p
    assert only_b_stale == dampen_if_stale(p, days_since_a=99.0, days_since_b=2.0)


def test_underdog_is_pulled_up_toward_half() -> None:
    """A stale underdog (<0.5) is dampened UP toward 0.5, staying an underdog."""
    pulled = dampen_if_stale(0.08, days_since_a=60.0, days_since_b=1.0)
    assert 0.08 < pulled < 0.5


def test_half_is_a_fixed_point() -> None:
    """A coin-flip prediction has no confidence to shed -- stale or not, it stays 0.5."""
    assert dampen_if_stale(0.5, days_since_a=365.0, days_since_b=365.0) == 0.5
