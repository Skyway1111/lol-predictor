"""PandaScore→canonical identity resolution for teams and leagues."""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable

    from league_predictor.upcoming.pandascore import PandaTeam

# PandaScore league name -> ACTIVE OE league_id, both sides verified against the live feed.
# Names OE uses verbatim (LCK, LEC, ...) stay out: league_to_oe falls back to identity.
PANDASCORE_LEAGUE_TO_OE: dict[str, str] = {
    "Esports World Cup": "EWC",
    "North American Challengers League": "NACL",
    "Prime League 1st Division": "PRM",
    "KeSPA Cup": "KeSPA",
    "EMEA Masters": "EM",
    "Mid-Season Invitational": "MSI",
    "Worlds": "WLDs",
    "First Stand": "FST",
    "LCK Challengers League": "LCKC",
    "Hitpoint Masters": "HM",
    "Road Of Legends": "ROL",
    "Rift Legends": "RL",
    "Circuito Desafiante": "CD",
    "Arabian League": "AL",
    "Asia Masters": "Asia Master",
    "LTA North": "LTA N",
    "LTA South": "LTA S",
    "CBLOL Academy": "CBLOLA",
    "LFL Division 2": "LFL2",
    "Ultraliga": "UL",
    "PG Nationals": "PGN",
    "Demacia Cup": "DCup",
}


def league_to_oe(conn: sqlite3.Connection, ps_league_name: str) -> str | None:
    """Map a PandaScore league name to an existing OE league_id, or None when unknown."""
    candidate = PANDASCORE_LEAGUE_TO_OE.get(ps_league_name, ps_league_name)
    row = conn.execute("SELECT league_id FROM leagues WHERE league_id = ?", (candidate,)).fetchone()
    return row[0] if row is not None else None


def resolve_teams(
    conn: sqlite3.Connection, teams: Iterable[PandaTeam], *, now: str
) -> dict[int, str | None]:
    """Resolve PandaScore teams to ``{ps_team_id: team_id-or-None}``, seeding/healing the map."""
    # A stored mapping wins; unmatched rows retry the exact normalized-name auto-match each run.
    # Never fuzzy-match: predicting the wrong team (academy roster, rebrand collision) is worse.
    resolved: dict[int, str | None] = {}
    name_index: dict[str, set[str]] | None = None  # built lazily, once per run
    for team in teams:
        if team.ps_team_id in resolved:
            continue
        row = conn.execute(
            "SELECT team_id, matched_via FROM pandascore_team_map WHERE ps_team_id = ?",
            (team.ps_team_id,),
        ).fetchone()
        if row is not None and row["matched_via"] != "unmatched":
            resolved[team.ps_team_id] = row["team_id"]
            continue
        if name_index is None:
            name_index = _oe_name_index(conn)
        candidates = name_index.get(_normalize(team.name), set())
        team_id = next(iter(candidates)) if len(candidates) == 1 else None
        conn.execute(
            "INSERT INTO pandascore_team_map"
            " (ps_team_id, ps_name, team_id, matched_via, created_at)"
            " VALUES (?, ?, ?, ?, ?)"
            " ON CONFLICT(ps_team_id) DO UPDATE SET team_id = excluded.team_id,"
            "   matched_via = excluded.matched_via, ps_name = excluded.ps_name",
            (team.ps_team_id, team.name, team_id, "name" if team_id else "unmatched", now),
        )
        resolved[team.ps_team_id] = team_id
    return resolved


def map_team_manually(conn: sqlite3.Connection, ps_team_id: int, team_id: str) -> None:
    """Hand-adjudicate one PandaScore team onto a canonical team id."""
    if conn.execute("SELECT 1 FROM teams WHERE team_id = ?", (team_id,)).fetchone() is None:
        raise ValueError(
            f"unknown team_id {team_id!r} — pick an existing teams.team_id "
            "(look it up via oe_team_names by team name)"
        )
    updated = conn.execute(
        "UPDATE pandascore_team_map SET team_id = ?, matched_via = 'manual' WHERE ps_team_id = ?",
        (team_id, ps_team_id),
    ).rowcount
    if not updated:
        raise ValueError(
            f"unknown ps_team_id {ps_team_id} — run `lol-predict upcoming` first; "
            "its summary lists the unmatched PandaScore teams and their ids"
        )


def _oe_name_index(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Normalized OE team name -> the team ids ever seen under it."""
    index: dict[str, set[str]] = {}
    rows = conn.execute(
        "SELECT team_id, team_name FROM oe_team_names"
        " UNION SELECT team_id, name FROM teams WHERE name IS NOT NULL"
    )
    for team_id, name in rows:
        index.setdefault(_normalize(name), set()).add(team_id)
    return index


def _normalize(name: str) -> str:
    """Case/space/punctuation-insensitive form; strips the feed's invisible characters."""
    cleaned = unicodedata.normalize("NFKC", name)
    cleaned = "".join(c for c in cleaned if unicodedata.category(c) not in ("Cf", "Cc"))
    cleaned = re.sub(r"[^\w\s]", "", cleaned.casefold())
    return re.sub(r"\s+", " ", cleaned).strip()
