"""Frozen value types for the draft model + the DraftState DB constructor."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER, Role

if TYPE_CHECKING:
    import sqlite3

#: The champion id of a slot nobody has picked yet — the sentinel every partial-board
#: surface (live sandbox drafts, blend probe fill, the suggest engine) tests against.
UNPICKED = "--"

# Canonicalise raw game_players role strings so DraftState always has ROLE_ORDER's 5 roles.
_ROLE_ALIASES: dict[str, Role] = {
    "top": "top",
    "jng": "jng",
    "jungle": "jng",
    "jgl": "jng",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "adc": "bot",
    "bottom": "bot",
    "sup": "sup",
    "support": "sup",
    "supp": "sup",
}


# Game ids per IN-list; bounds the SQLite parameter count of one bulk_load round trip.
_BULK_LOAD_CHUNK = 500


def normalize_role(raw: str) -> Role:
    """The canonical role for a raw ``game_players`` spelling; the role alias table."""
    key = raw.strip().lower()
    if key not in _ROLE_ALIASES:
        raise ValueError(f"unknown role: {raw!r}")
    return _ROLE_ALIASES[key]


def _slots_by_team(
    pg_rows: list[tuple],
    team_ids: tuple[str, str],
) -> dict[str, dict[Role, PlayerSlot]]:
    """Collapse ``game_players`` rows into one PlayerSlot per (team, role).

    Args:
        pg_rows: ``(player_id, team_id, champion_id, role)`` tuples; sorted for determinism.

    Raises:
        ValueError: unknown role, team_id matching neither side, or a champion conflict.
    """
    by_team: dict[str, dict[Role, PlayerSlot]] = {team_ids[0]: {}, team_ids[1]: {}}
    for player_id, team_id, champion_id, raw_role in sorted(pg_rows):
        if team_id not in by_team:
            raise ValueError(f"player_game team_id={team_id!r} matches neither side")
        role = normalize_role(raw_role)
        slots = by_team[team_id]
        existing = slots.get(role)
        if existing is None:
            slots[role] = PlayerSlot(role=role, champion_id=champion_id, player_id=player_id)
        elif existing.champion_id != champion_id:
            raise ValueError(f"conflicting champions for role {role} on team {team_id}")
    return by_team


@dataclass(frozen=True)
class PlayerSlot:
    """One drafted pick: which role, which champion, which player."""

    role: Role
    champion_id: str
    player_id: str


@dataclass(frozen=True)
class TeamComp:
    """One team's full draft state: 5 role-ordered slots, plus optional bans and picks.

    Attributes:
        bans: Ordered by the slot *this team* banned; ``None`` entries hold slot position,
            and the field itself is ``None`` when the whole ban list is missing.
        picks: Ordered by the slot *this team* picked, same ``None`` convention.
    """

    team_id: str
    slots: tuple[PlayerSlot, ...]
    bans: tuple[str | None, ...] | None = None

    @property
    def champion_ids(self) -> list[str]:
        """The five drafted champions, in ROLE_ORDER — the one home for this walk."""
        return [s.champion_id for s in self.slots]

    def __post_init__(self) -> None:
        if len(self.slots) != N_ROLES:
            raise ValueError(f"TeamComp.slots must have length {N_ROLES}, got {len(self.slots)}")
        roles = tuple(s.role for s in self.slots)
        if roles != ROLE_ORDER:
            raise ValueError(
                f"TeamComp.slots must be ordered by ROLE_ORDER={ROLE_ORDER}, got {roles}"
            )


@dataclass(frozen=True)
class DraftState:
    """Single typed input to every modeling function in `draft_grading/`."""

    match_id: str
    date: datetime
    patch: str
    region: str
    blue: TeamComp
    red: TeamComp
    # The interleaved sequence is derivable from (blue.picks, red.picks,
    # first_pick_team_id) under standard draft order; not materialized here.

    @classmethod
    def bulk_load(
        cls,
        db: sqlite3.Connection,
        game_ids: list[str],
    ) -> dict[str, DraftState]:
        """Bulk-load DraftStates from the DB, keyed by game_id — the sole DB constructor.

        Returns:
            One DraftState per game whose five roles per side resolve, never a partial one.
            ``date`` is the series' naive-UTC scheduled timestamp; ``region`` its league_id.
        """
        if not game_ids:
            return {}

        results: dict[str, DraftState] = {}

        for start in range(0, len(game_ids), _BULK_LOAD_CHUNK):
            chunk = game_ids[start : start + _BULK_LOAD_CHUNK]
            placeholders = ",".join("?" for _ in chunk)

            game_rows = db.execute(
                f"""
                SELECT g.game_id, g.blue_team, g.red_team, g.patch,
                       s.scheduled_ts, s.league_id
                FROM games g
                JOIN series s ON s.series_id = g.series_id
                WHERE g.game_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()

            gp_rows = db.execute(
                f"""
                SELECT game_id, player_id, team_id, champion_id, role
                FROM game_players
                WHERE game_id IN ({placeholders})
                """,
                chunk,
            ).fetchall()
            gp_by_game: dict[str, list[tuple]] = {}
            for r in gp_rows:
                gp_by_game.setdefault(r["game_id"], []).append(
                    (r["player_id"], r["team_id"], r["champion_id"], r["role"])
                )

            bans_by_game = _draft_actions_by_game(db, chunk)

            for gr in game_rows:
                gid = gr["game_id"]
                blue_team, red_team = gr["blue_team"], gr["red_team"]
                if blue_team is None or red_team is None:
                    continue
                date = _parse_scheduled_ts(gr["scheduled_ts"])
                if date is None:
                    continue

                try:
                    per_team = _slots_by_team(gp_by_game.get(gid, []), (blue_team, red_team))
                except ValueError:
                    continue

                game_bans = bans_by_game.get(gid, {})
                blue_comp = _build_comp_from_map(blue_team, per_team, game_bans.get("blue"))
                red_comp = _build_comp_from_map(red_team, per_team, game_bans.get("red"))
                if blue_comp is None or red_comp is None:
                    continue

                results[gid] = cls(
                    match_id=gid,
                    date=date,
                    patch=gr["patch"] or "",
                    region=gr["league_id"] or "",
                    blue=blue_comp,
                    red=red_comp,
                )

        return results


def _draft_actions_by_game(
    db: sqlite3.Connection, game_ids: list[str]
) -> dict[str, dict[str, list[str]]]:
    """Bans per game and side in action order."""
    rows = db.execute(
        f"SELECT game_id, side, champion_id FROM draft_actions "
        f"WHERE game_id IN ({','.join('?' for _ in game_ids)}) "
        "AND kind = 'ban' ORDER BY game_id, seq",
        game_ids,
    )
    bans: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        bans.setdefault(row["game_id"], {}).setdefault(row["side"], []).append(row["champion_id"])
    return bans


def _parse_scheduled_ts(raw: str | None) -> datetime | None:
    """Parse ``series.scheduled_ts`` (UTC ISO-8601) into a NAIVE datetime; None if unparseable."""
    if raw is None:
        return None
    try:
        dt = datetime.fromisoformat(str(raw))
    except (ValueError, TypeError):
        return None
    return dt.replace(tzinfo=None)


def _build_comp_from_map(
    team_id: str,
    per_team: dict[str, dict[Role, PlayerSlot]],
    bans: list[str] | None,
) -> TeamComp | None:
    """Assemble one side's TeamComp; None when the five roles don't all resolve."""
    slots_map = per_team.get(team_id, {})
    if any(r not in slots_map for r in ROLE_ORDER):
        return None
    slots = tuple(slots_map[r] for r in ROLE_ORDER)
    return TeamComp(
        team_id=team_id,
        slots=slots,
        bans=tuple(bans) if bans else None,
    )
