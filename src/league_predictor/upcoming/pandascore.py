"""PandaScore API client for the LoL match feeds — the one module that calls it."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import requests

from league_predictor import clock

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

UPCOMING_URL = "https://api.pandascore.co/lol/matches/upcoming"
RUNNING_URL = "https://api.pandascore.co/lol/matches/running"
UPCOMING_HORIZON_DAYS = 14  # covers the dashboard's week view with margin
_PAGE_SIZE = 100
_MAX_PAGES = 10  # request-count guard; 1000 scheduled pro matches never happens


@dataclass(frozen=True)
class PandaTeam:
    """One filled opponent slot of a PandaScore match."""

    ps_team_id: int
    name: str


@dataclass(frozen=True)
class PandaMatch:
    """One PandaScore match, reduced to the fields the pipeline stores."""

    ps_match_id: int
    league_name: str
    scheduled_ts: str  # ISO 8601 UTC with +00:00 offset
    best_of: int | None
    status: str  # not_started / running / finished / canceled / postponed
    teams: tuple[PandaTeam, ...]  # 0-2 entries; TBD bracket slots arrive short
    tournament_name: str | None = None  # the stage ("Playoffs", "Group A") is_playoffs prices
    wins: tuple[tuple[int, int], ...] = ()  # (ps_team_id, maps won so far), from finished games


def _maps_won(raw: Mapping) -> tuple[tuple[int, int], ...]:
    """Each team's finished-map count from the match's ``games`` (a forfeited map counts)."""
    counts: Counter[int] = Counter()
    for game in raw.get("games") or ():
        winner = game.get("winner") or {}
        if game.get("status") == "finished" and winner.get("type") == "Team" and winner.get("id"):
            counts[int(winner["id"])] += 1
    return tuple(sorted(counts.items()))


def parse_matches(payload: Iterable[dict]) -> list[PandaMatch]:
    """Reduce raw PandaScore match dicts to ``PandaMatch`` rows."""
    # Unscheduled matches drop out, and only ``type == "Team"`` opponent slots become teams.
    matches = []
    for raw in payload:
        scheduled_at = raw.get("scheduled_at")
        if not scheduled_at:
            continue
        teams = tuple(
            PandaTeam(
                ps_team_id=slot["opponent"]["id"],
                name=slot["opponent"]["name"],
            )
            for slot in raw.get("opponents", [])
            if slot.get("type") == "Team"
        )
        matches.append(
            PandaMatch(
                ps_match_id=raw["id"],
                league_name=raw["league"]["name"],
                scheduled_ts=_utc_iso(scheduled_at),
                best_of=raw.get("number_of_games"),
                status=raw["status"],
                teams=teams,
                tournament_name=(raw.get("tournament") or {}).get("name"),
                wins=_maps_won(raw),
            )
        )
    return matches


def _fetch_pages(url: str, token: str, *, horizon_end: str | None) -> list[PandaMatch]:
    """Page one feed; a horizon stops the walk once every row on a page is past it."""
    # Auth is the ``token`` query parameter; the free plan rejects the Bearer-header form.
    # Any non-200 raises ``requests.HTTPError`` rather than letting the run sync nothing.
    matches: list[PandaMatch] = []
    for page in range(1, _MAX_PAGES + 1):
        response = requests.get(
            url, params={"token": token, "per_page": _PAGE_SIZE, "page": page}, timeout=30
        )
        response.raise_for_status()
        page_matches = parse_matches(response.json())
        if not page_matches:
            break
        if horizon_end is None:
            matches.extend(page_matches)
            continue
        matches.extend(m for m in page_matches if m.scheduled_ts <= horizon_end)
        if all(m.scheduled_ts > horizon_end for m in page_matches):
            break
    return matches


def fetch_upcoming(
    token: str,
    *,
    now: datetime | None = None,
) -> list[PandaMatch]:
    """Fetch every running LoL match plus every one scheduled within the configured horizon."""
    now = now or clock.utc_now().replace(tzinfo=UTC)
    horizon_end = (now + timedelta(days=UPCOMING_HORIZON_DAYS)).isoformat()
    running = _fetch_pages(RUNNING_URL, token, horizon_end=None)
    upcoming = _fetch_pages(UPCOMING_URL, token, horizon_end=horizon_end)
    seen = {m.ps_match_id for m in running}
    return running + [m for m in upcoming if m.ps_match_id not in seen]


def _utc_iso(stamp: str) -> str:
    """Normalize a PandaScore timestamp to ISO 8601 UTC with a +00:00 offset."""
    return datetime.fromisoformat(stamp).astimezone(UTC).isoformat()
