"""DraftState construction and rebuilt-schema (``game_players``) DB loading."""

from __future__ import annotations

import sqlite3
from datetime import datetime

import pytest

from league_predictor.draft_grading.state import (
    ROLE_ORDER,
    DraftState,
    PlayerSlot,
    TeamComp,
)
from tests.draft_grading.grader.conftest import make_grader_test_db


def _insert_game(
    con: sqlite3.Connection,
    game_id: str,
    blue: str,
    red: str,
    *,
    scheduled_ts: str = "2026-03-15T00:00:00+00:00",
    league_id: str = "LPL",
) -> None:
    """Insert one series + game (blue/red split, date on the series)."""
    series_id = f"ser_{game_id}"
    con.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "winner, status) VALUES (?, ?, ?, ?, ?, ?, 'completed')",
        (series_id, league_id, min(blue, red), max(blue, red), scheduled_ts, blue),
    )
    con.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, "
        "winner, patch) VALUES (?, ?, 1, ?, ?, ?, '26.05')",
        (game_id, series_id, blue, red, blue),
    )


def _insert_full_team(con: sqlite3.Connection, game_id: str, team_id: str) -> None:
    """Insert one clean, complete 5-role lineup for a team."""
    for role in ROLE_ORDER:
        con.execute(
            "INSERT INTO game_players (game_id, player_id, team_id, "
            "champion_id, role) VALUES (?,?,?,?,?)",
            (game_id, f"{team_id}_{role}_pid", team_id, f"{team_id}_{role}_champ", role),
        )


def test_bulk_load_resolves_five_role_slots_per_side() -> None:
    """A complete game reconstructs five role-ordered slots per side; no actions means None."""
    con = make_grader_test_db()
    try:
        _insert_game(con, "M1", blue="BLUE", red="RED")
        _insert_full_team(con, "M1", "BLUE")
        _insert_full_team(con, "M1", "RED")

        states = DraftState.bulk_load(con, ["M1"])
        assert "M1" in states
        state = states["M1"]
        assert state.match_id == "M1"
        assert isinstance(state.date, datetime)
        assert state.region == "LPL"
        assert state.blue.team_id == "BLUE"
        assert state.red.team_id == "RED"
        assert tuple(s.role for s in state.blue.slots) == ROLE_ORDER
        assert tuple(s.role for s in state.red.slots) == ROLE_ORDER
        assert state.blue.bans is None
        assert state.red.bans is None
    finally:
        con.close()


def test_bulk_load_skips_game_on_champion_conflict() -> None:
    """Two champions on one (team, role) is a source conflict: the game is dropped, not guessed."""
    con = make_grader_test_db()
    try:
        _insert_game(con, "M2", blue="BLUE", red="RED")
        _insert_full_team(con, "M2", "BLUE")
        _insert_full_team(con, "M2", "RED")
        # A second RED top on a DIFFERENT champion -> conflict.
        con.execute(
            "INSERT INTO game_players (game_id, player_id, team_id, "
            "champion_id, role) VALUES (?,?,?,?,?)",
            ("M2", "RED_top_other", "RED", "RED_top_DIFFERENT", "top"),
        )

        states = DraftState.bulk_load(con, ["M2"])
        assert "M2" not in states, "champion conflict must drop the game"
    finally:
        con.close()


def test_bulk_load_skips_game_missing_a_role() -> None:
    """A side missing one of the five roles yields no DraftState; partials are never emitted."""
    con = make_grader_test_db()
    try:
        _insert_game(con, "M3", blue="BLUE", red="RED")
        _insert_full_team(con, "M3", "RED")
        # BLUE has only four roles (sup missing).
        for role in ("top", "jng", "mid", "bot"):
            con.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, "
                "champion_id, role) VALUES (?,?,?,?,?)",
                ("M3", f"BLUE_{role}_pid", "BLUE", f"BLUE_{role}_champ", role),
            )

        states = DraftState.bulk_load(con, ["M3"])
        assert "M3" not in states, "missing role nets <5 slots; must skip"
    finally:
        con.close()


def _make_slots(team_id: str) -> tuple[PlayerSlot, ...]:
    return tuple(
        PlayerSlot(role=r, champion_id=f"{team_id}_{r}_champ", player_id=f"{team_id}_{r}_pid")
        for r in ROLE_ORDER
    )


def test_team_comp_requires_five_role_ordered_slots() -> None:
    slots = _make_slots("blue")
    comp = TeamComp(team_id="blue", slots=slots)
    assert tuple(s.role for s in comp.slots) == ROLE_ORDER

    with pytest.raises(ValueError, match="length 5"):
        TeamComp(team_id="blue", slots=slots[:4])

    bad_order = (slots[1], slots[0], *slots[2:])
    with pytest.raises(ValueError, match="ordered by ROLE_ORDER"):
        TeamComp(team_id="blue", slots=bad_order)
