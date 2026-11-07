"""Tier mapping for model-uncertainty pooling + rating-pool rename aliases."""

from __future__ import annotations

import pytest

from league_predictor.predraft.league_tier import (
    INTERNATIONAL_EVENTS,
    LEAGUE_TIER_MAP,
    LEAGUES,
    RATING_POOL_ALIAS,
    league_tier,
    resolve_domestic_league,
)


@pytest.mark.parametrize(
    ("region", "expected"),
    [
        # Tier 1: major pro leagues and internationals.
        ("LCK", "tier1"),
        ("LPL", "tier1"),
        ("LEC", "tier1"),
        ("LCS", "tier1"),
        ("LTA N", "tier1"),
        ("LTA S", "tier1"),
        ("LTA", "tier1"),
        ("CBLOL", "tier1"),
        ("MSI", "tier1"),
        ("Worlds", "tier1"),
        # Tier 2: regional / domestic leagues a rung below the majors.
        ("LFL", "tier2"),
        ("PRM", "tier2"),
        ("Prime League 1st Division", "tier2"),
        ("LIT", "tier2"),
        ("LCP", "tier2"),
        ("PCS", "tier2"),
        ("VCS", "tier2"),
        # Tier 3: academy / challenger / unknown (conservative default).
        ("LCK CL", "tier3"),
        ("LCK Challengers League", "tier3"),
        ("NACL", "tier3"),
        ("LPL Academy", "tier3"),
        ("", "tier3"),
        (None, "tier3"),
        ("Unknown Region", "tier3"),
    ],
)
def test_league_tier_mapping(region: str | None, expected: str) -> None:
    assert league_tier(region) == expected


def test_rating_pool_alias_targets_are_current_major_codes():
    """Every rename-pool target must be tier-1, else ``region_rating`` splits its pool."""
    for old, current in RATING_POOL_ALIAS.items():
        assert league_tier(current) == "tier1", (old, current)


@pytest.mark.parametrize(
    ("league_id", "expected"),
    [
        ("LCK", "LCK"),
        ("LCKC", "LCKC"),
        ("NACL", "NACL"),
        ("LCSA", "LCSA"),
        ("LJLCS", "LJLCS"),
        ("LPLOL", "LPLOL"),
        ("DCup", "DCup"),
        ("IC", "IC"),
        ("NA LCS", "NA LCS"),
        ("EU LCS", "EU LCS"),
        ("LTA N", "LTA N"),
        ("LTA S", "LTA S"),
        ("Worlds", None),
        (None, None),
    ],
)
def test_domestic_league_resolution_is_exact(league_id, expected):
    assert resolve_domestic_league(league_id) == expected


def test_every_international_event_is_tier1_and_outside_the_training_population():
    for name in INTERNATIONAL_EVENTS:
        assert league_tier(name) == "tier1", name
        assert LEAGUE_TIER_MAP.get(name) is None, name
    assert (
        frozenset(name for name, lg in LEAGUES.items() if lg.international) == INTERNATIONAL_EVENTS
    )
