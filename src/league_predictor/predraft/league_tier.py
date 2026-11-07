"""The one home of league-name knowledge: one row per spelling, every tier view derives."""

from __future__ import annotations

from dataclasses import dataclass
from types import EllipsisType


@dataclass(frozen=True)
class League:
    """One league spelling's row in ``LEAGUES``.

    Attributes:
        uncertainty: model-uncertainty tier ("tier1"/"tier2" exact; None falls
            through the prefix rules, then tier3).
        training: TRAINING-POPULATION tier selecting leagues for player-pillar
            ratings and regional Elo. None = international; ``...`` = outside the
            training population (readers apply ``LEAGUE_TIER_DEFAULT``).
        international: cross-regional event (the only games connecting regional
            pools; the skill-filter corpus reads this as a context channel).
        region: normalized region for Impact's EXACT league→region map; None
            stays out of ``REGION_NORMALIZATION_MAP`` (misses price by tier).
    """

    uncertainty: str | None = None
    training: int | None | EllipsisType = ...
    international: bool = False
    region: str | None = None


LEAGUES: dict[str, League] = {
    "LCK": League(uncertainty="tier1", training=1, region="LCK"),
    "LPL": League(uncertainty="tier1", training=1, region="LPL"),
    "LEC": League(uncertainty="tier1", training=1, region="LEC"),
    "LCS": League(uncertainty="tier1", training=1, region="NA"),
    "EU LCS": League(uncertainty="tier1"),
    "NA LCS": League(uncertainty="tier1"),
    "LTA N": League(uncertainty="tier1", training=1, region="NA"),
    "LTA S": League(uncertainty="tier1", training=1, region="LATAM"),
    "LTA": League(uncertainty="tier1", training=1, region="LATAM"),
    "CBLOL": League(uncertainty="tier1", training=2, region="CBLOL"),
    "MSI": League(uncertainty="tier1", training=None, international=True),
    "Worlds": League(uncertainty="tier1", international=True),
    "WLDs": League(uncertainty="tier1", training=None, international=True),
    "First Stand": League(uncertainty="tier1", international=True),
    "FST": League(uncertainty="tier1", training=None, international=True),
    "EWC": League(uncertainty="tier1", training=None, international=True),
    "Esports World Cup": League(uncertainty="tier1", international=True),
    "LFL": League(uncertainty="tier2", training=2),
    "PRM": League(uncertainty="tier2", training=2),
    "LIT": League(uncertainty="tier2"),
    "LCP": League(uncertainty="tier2", training=2),
    "PCS": League(uncertainty="tier2", training=2, region="PCS"),
    "VCS": League(uncertainty="tier2", training=2, region="VCS"),
    "LJL": League(uncertainty="tier2", training=2, region="LJL"),
    "LCO": League(uncertainty="tier2", training=2, region="LCO"),
    "TCL": League(uncertainty="tier2", training=2, region="TCL"),
    "LLA": League(uncertainty="tier2", training=2, region="LATAM"),
    "ROL": League(uncertainty="tier2"),
    "RL": League(uncertainty="tier2"),
    "EM": League(uncertainty="tier2", training=2),
    "LES": League(uncertainty="tier2"),
    "CD": League(uncertainty="tier2"),
    "NLC": League(uncertainty="tier2", training=2),
    "Ultraliga": League(uncertainty="tier2"),
    "Honor League": League(uncertainty="tier2"),
    "SuperLiga": League(uncertainty="tier2"),
    "LVP SL": League(uncertainty="tier2", training=2),
    "Arabian League": League(uncertainty="tier2"),
    # Academy/challenger circuits: uncertainty falls through the prefix rules.
    "NACL": League(training=2),
    "LDL": League(training=2),
    "LCKC": League(training=2),
    "LFL2": League(training=2),
    "CBLOLA": League(training=2),
    "AL": League(training=2),
    "HM": League(training=2),
    "LAS": League(training=2),
    "EBL": League(training=2),
    # Observed domestic cups/development circuits outside the model-training map.
    "LCSA": League(),
    "LJLCS": League(),
    "LPLOL": League(),
    "DCup": League(),
    "IC": League(),
}

# ``ratings.region_rating`` folds these codes into the current league's pool so a rename
# does not split rating history — rating-pool identity only, not present-day league identity.
RATING_POOL_ALIAS: dict[str, str] = {
    "LTA N": "LCS",
    "LTA S": "CBLOL",
    "EU LCS": "LEC",
    "NA LCS": "LCS",
}


def resolve_domestic_league(league_id: str | None) -> str | None:
    """Return an exact known domestic league id; unknown and international ids miss."""
    if league_id is None:
        return None
    league = LEAGUES.get(league_id)
    if league is None or league.international:
        return None
    return league_id


INTERNATIONAL_EVENTS: frozenset[str] = frozenset(
    name for name, lg in LEAGUES.items() if lg.international
)

# The ``league_id`` spellings of the international events (OE short codes, matched against
# DB columns). International rows with ``training`` unset are tournament-name spellings only.
INTERNATIONAL_LEAGUES: tuple[str, ...] = tuple(
    name for name, lg in LEAGUES.items() if lg.international and lg.training is None
)

#: The domestic top flights (tier-1 model uncertainty, not international).
MAJOR_LEAGUES: tuple[str, ...] = tuple(
    name for name, lg in LEAGUES.items() if lg.uncertainty == "tier1" and not lg.international
)


#: Impact's EXACT league→normalized-region map; a league without a row/region misses
#: and prices by tier (never as an average major region).
REGION_NORMALIZATION_MAP: dict[str, str] = {
    name: lg.region for name, lg in LEAGUES.items() if lg.region is not None
}

#: The four premier regions, in the normalized region space of ``REGION_NORMALIZATION_MAP``;
#: the player-Impact reference population every published Impact number is read against.
PREMIER_REGIONS: frozenset[str] = frozenset({"LCK", "LPL", "LEC", "NA"})

LEAGUE_TIER_MAP: dict[str, int | None] = {
    name: lg.training for name, lg in LEAGUES.items() if not isinstance(lg.training, EllipsisType)
}
LEAGUE_TIER_DEFAULT: int = 3


def domestic_league_tier(league_id: str) -> int:
    """Return the team-display/regional-Elo tier without changing training membership."""
    training_tier = LEAGUE_TIER_MAP.get(league_id)
    if training_tier is not None:
        return training_tier
    uncertainty = LEAGUES[league_id].uncertainty
    if uncertainty == "tier1":
        return 1
    if uncertainty == "tier2":
        return 2
    return LEAGUE_TIER_DEFAULT


_TIER1: frozenset[str] = frozenset(
    name for name, lg in LEAGUES.items() if lg.uncertainty == "tier1"
)
_TIER2_EXACT: frozenset[str] = frozenset(
    name for name, lg in LEAGUES.items() if lg.uncertainty == "tier2"
)

# Tier 2 long-form tournament names that don't prefix-match a short code.
_TIER2_PREFIXES: tuple[str, ...] = ("Prime League",)

_TIER3_PREFIXES: tuple[str, ...] = (
    "LCK CL",
    "LCK Challengers",
    "LPL Academy",
    "LEC Div",
    "LTA N Div",
    "NACL",
)


def league_tier(region: str | None) -> str:
    """Return ``"tier1" | "tier2" | "tier3"``; unknown or empty regions fall through to tier3."""
    if not region:
        return "tier3"
    if region in _TIER1:
        return "tier1"
    if region in _TIER2_EXACT:
        return "tier2"
    for prefix in _TIER3_PREFIXES:
        if region.startswith(prefix):
            return "tier3"
    for prefix in _TIER2_PREFIXES:
        if region.startswith(prefix):
            return "tier2"
    return "tier3"
