"""The fitted region-strength table: what every consumer of a region rating reads."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from league_predictor.predraft.league_tier import (
    LEAGUES,
    RATING_POOL_ALIAS,
    domestic_league_tier,
    league_tier,
)
from league_predictor.probability import ELO_ORIGIN, ELO_PER_LOGODDS, logit, sigmoid

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date

_EPS = 1e-6  # logit saturation: served probs live well inside [1e-6, 1-1e-6]

_TIER_OF_TIER_NAME = {"tier1": 1, "tier2": 2, "tier3": 3}


def _pool_tier(pool: str) -> int:
    """A pool's integer tier; unknown spellings price as tier 3."""
    if pool in LEAGUES:
        return domestic_league_tier(pool)
    return _TIER_OF_TIER_NAME[league_tier(pool)]


@dataclass(frozen=True)
class RegionRatingTable:
    """The fitted region-strength table plus the served residual's coefficients.

    Attributes:
        r: pool -> log-odds population level (fitted pool term + five times the
            pool's appearance-weighted mean player effect), tier-1 mean at 0. The
            player term is what lets a pool with zero cross-pool games move off
            its tier level on migration evidence.
        tier_effect: tier (1/2/3) -> log-odds level pricing pools without data.
        home: ``"team_id|year"`` -> pool, from the fit's tier-inclusive home map.
        beta_cross_tier: residual weight on cross-tier matchups; 0 disables.
        beta_within_tier1: residual weight on tier1-vs-tier1 cross-pool matchups.
    """

    r: dict[str, float]
    tier_effect: dict[int, float]
    home: dict[str, str]
    beta_cross_tier: float = 0.0
    beta_within_tier1: float = 0.0

    def rating(self, league_id: str | None) -> float:
        """A league's log-odds rating: its pool's fit, else its tier effect."""
        if league_id is None:
            return self.tier_effect[3]
        pool = RATING_POOL_ALIAS.get(league_id, league_id)
        if pool in self.r:
            return self.r[pool]
        return self.tier_effect[_pool_tier(pool)]

    def has_pool(self, league_id: str | None) -> bool:
        """Whether ``rating`` resolves this league to a fitted pool row (not a tier fallback)."""
        if league_id is None:
            return False
        return RATING_POOL_ALIAS.get(league_id, league_id) in self.r

    def elo(self, league_id: str | None) -> float:
        """The display-scale rating (tier-1 mean sits at ``ELO_ORIGIN``)."""
        return ELO_ORIGIN + self.rating(league_id) * ELO_PER_LOGODDS

    def league_of(self, team_id: str, year: int) -> str | None:
        """The team's home pool for ``year``, or ``None`` when unmapped."""
        return self.home.get(f"{team_id}|{year}")

    def adjust(self, served: float, team_a: str, team_b: str, as_of: date) -> float:
        """Cross-pool residual over a served probability.

        Returns:
            ``served`` byte-identically on same-pool, unmapped, or same-tier
            (below tier 1) matchups; otherwise the served logit shifted by the
            matchup class's beta times the rating gap.
        """
        pa, pb = self.league_of(team_a, as_of.year), self.league_of(team_b, as_of.year)
        if pa is None or pb is None or pa == pb:
            return served
        ta, tb = _pool_tier(pa), _pool_tier(pb)
        beta = (self.beta_within_tier1 if ta == 1 else 0.0) if ta == tb else self.beta_cross_tier
        if beta == 0.0:
            return served
        return sigmoid(logit(served, eps=_EPS) + beta * (self.rating(pa) - self.rating(pb)))

    def to_meta(self) -> dict[str, object]:
        """Serialize to the JSON-safe dict persisted in the served artifact meta."""
        return {
            "r": dict(self.r),
            "tier_effect": {str(t): v for t, v in self.tier_effect.items()},
            "home": dict(self.home),
            "beta_cross_tier": float(self.beta_cross_tier),
            "beta_within_tier1": float(self.beta_within_tier1),
        }

    @staticmethod
    def from_meta(data: Mapping) -> RegionRatingTable:
        """Reconstruct :func:`to_meta`'s table exactly."""
        return RegionRatingTable(
            r=dict(data["r"]),
            tier_effect={int(t): v for t, v in data["tier_effect"].items()},
            home=dict(data["home"]),
            beta_cross_tier=float(data["beta_cross_tier"]),
            beta_within_tier1=float(data["beta_within_tier1"]),
        )

    def with_betas(self, *, cross_tier: float, within_tier1: float) -> RegionRatingTable:
        """The same table with the served residual coefficients stamped."""
        return replace(self, beta_cross_tier=cross_tier, beta_within_tier1=within_tier1)
