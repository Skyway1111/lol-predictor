"""Player performance profiles with recency-weighted exponential decay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime

#: ``YYYY-MM-DD`` prefix length: anything past it is the time part of an ISO stamp.
_ISO_DATE_LEN = 10

OPTIONAL_STATS = (
    "wpm",
    "wcpm",
    "wardsplaced",
    "wardskilled",
    "damagetakenperminute",
    "damagemitigatedperminute",
    "firstbloodkill",
    "firstbloodassist",
    "killsat15",
    "assistsat15",
    "deathsat15",
)


@dataclass(frozen=True, slots=True)
class PerformanceAggregate:
    """Sufficient statistics from which every public profile field is derived."""

    total_weight: float
    laning_weight: float
    sums: dict[str, float]
    laning_sums: dict[str, float]
    optional_sums: dict[str, float]
    optional_weights: dict[str, float]
    champions: dict[str, tuple[int, int]]
    games: int
    full_laning_games: int
    fallback_laning_games: int


def assemble_performance_profile(aggregate: PerformanceAggregate) -> dict[str, object]:
    """Build the canonical profile shape from reader-independent sufficient statistics."""
    if aggregate.total_weight <= 0:
        raise ValueError("performance aggregate has no weight")

    def avg(stat: str) -> float:
        return aggregate.sums[stat] / aggregate.total_weight

    def laning_avg(stat: str) -> float:
        if aggregate.laning_weight <= 0:
            return 0.0
        return aggregate.laning_sums[stat] / aggregate.laning_weight

    def optional_avg(stat: str) -> float:
        weight = aggregate.optional_weights[stat]
        return aggregate.optional_sums[stat] / weight if weight > 0 else 0.0

    teamfighting = {
        "dpm": avg("dpm"),
        "damageshare": avg("damageshare"),
        "killparticipation": avg("killparticipation"),
        "kda": avg("kda_numerator") / (avg("deaths") + 1e-3),
        "earnedgoldshare": avg("earnedgoldshare"),
        "firstbloodkill": optional_avg("firstbloodkill"),
        "firstbloodassist": optional_avg("firstbloodassist"),
    }
    ranked_champions = sorted(
        aggregate.champions,
        key=lambda champion: aggregate.champions[champion],
        reverse=True,
    )
    if aggregate.full_laning_games:
        quality = "full"
    elif aggregate.fallback_laning_games:
        quality = "partial"
    else:
        quality = "imputed"
    return {
        "laning": {
            "golddiffat15": laning_avg("golddiffat15"),
            "csdiffat15": laning_avg("csdiffat15"),
            "xpdiffat15": laning_avg("xpdiffat15"),
            "cspm": avg("cspm"),
        },
        "teamfighting": teamfighting,
        "vision": {
            "vspm": avg("vspm"),
            "controlwardsbought": avg("controlwardsbought"),
            "wpm": optional_avg("wpm"),
            "wcpm": optional_avg("wcpm"),
            "wardsplaced": optional_avg("wardsplaced"),
            "wardskilled": optional_avg("wardskilled"),
        },
        "durability": {
            "damagetakenperminute": optional_avg("damagetakenperminute"),
            "damagemitigatedperminute": optional_avg("damagemitigatedperminute"),
        },
        "box15": {
            "killsat15": optional_avg("killsat15"),
            "assistsat15": optional_avg("assistsat15"),
            "deathsat15": optional_avg("deathsat15"),
        },
        "objective": {
            "firstblood_rate": avg("firstblood"),
            "firsttower_rate": avg("firsttower"),
            "firstdragon_rate": avg("firstdragon"),
            "firstbaron_rate": avg("firstbaron"),
        },
        "champion_pool": {
            "champion_diversity": len(aggregate.champions),
            "top_champions": ranked_champions[:5],
            "games_played": aggregate.games,
        },
        "data_quality": quality,
        "present": {stat: aggregate.optional_weights[stat] > 0 for stat in OPTIONAL_STATS},
    }


def _parse_match_datetime(match_date_str: str) -> datetime:
    """Parse ``series.scheduled_ts`` (``+00:00`` offset) to NAIVE UTC, no strptime overhead.

    Returns:
        The same instant with tzinfo dropped, so the recency math's subtraction from
        naive as-of datetimes yields correct deltas.
    """
    if len(match_date_str) > _ISO_DATE_LEN and match_date_str[_ISO_DATE_LEN] in {" ", "T"}:
        dt = datetime.fromisoformat(match_date_str.replace(" ", "T", 1))
        if dt.tzinfo is not None:
            dt = dt.astimezone(UTC).replace(tzinfo=None)
        return dt
    day = date.fromisoformat(match_date_str[:_ISO_DATE_LEN])
    return datetime.combine(day, datetime.min.time())


def _as_of_datetime(as_of_date: date | datetime) -> datetime:
    """Datetime cutoffs pass through; date cutoffs become that day's midnight."""
    if isinstance(as_of_date, datetime):
        return as_of_date
    return datetime.combine(as_of_date, datetime.min.time())
