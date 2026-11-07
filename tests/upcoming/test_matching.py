"""PandaScore→canonical identity resolution: teams via the durable map, leagues via the name map."""

from __future__ import annotations

from league_predictor.upcoming.matching import league_to_oe, map_team_manually, resolve_teams
from league_predictor.upcoming.pandascore import PandaTeam

NOW = "2026-07-15T12:00:00+00:00"


def _seed_team(conn, team_id: str, *names: str) -> None:
    conn.execute("INSERT INTO teams (team_id, name) VALUES (?, ?)", (team_id, names[0]))
    conn.executemany(
        "INSERT INTO oe_team_names (team_id, team_name) VALUES (?, ?)",
        [(team_id, name) for name in names],
    )


def _resolve_one(conn, team: PandaTeam) -> str | None:
    return resolve_teams(conn, [team], now=NOW)[team.ps_team_id]


def test_league_to_oe_maps_known_names(rebuild_db):
    conn = rebuild_db
    conn.execute("INSERT INTO leagues (league_id, name) VALUES ('EWC', 'EWC')")
    conn.execute("INSERT INTO leagues (league_id, name) VALUES ('LCK', 'LCK')")
    assert league_to_oe(conn, "Esports World Cup") == "EWC"  # explicit map entry
    assert league_to_oe(conn, "LCK") == "LCK"  # identity fallback
    assert league_to_oe(conn, "Baltic Masters") is None  # unknown stays unmapped


def test_resolve_seeds_map_on_unique_name_match(rebuild_db):
    conn = rebuild_db
    _seed_team(conn, "oe:team:abc", "G2 Esports")
    team = PandaTeam(ps_team_id=101, name="G2 Esports")
    assert _resolve_one(conn, team) == "oe:team:abc"
    row = conn.execute("SELECT * FROM pandascore_team_map WHERE ps_team_id = 101").fetchone()
    assert row["team_id"] == "oe:team:abc"
    assert row["matched_via"] == "name"


def test_resolve_normalizes_case_whitespace_and_invisible_chars(rebuild_db):
    conn = rebuild_db
    _seed_team(conn, "oe:team:tpx", "Team Paradox")
    # trailing zero-width char and case noise, as seen in the live feed
    team = PandaTeam(ps_team_id=102, name="team  PARADOX\u2060")
    assert _resolve_one(conn, team) == "oe:team:tpx"


def test_resolve_returns_existing_mapping_without_rematching(rebuild_db):
    """A stored mapping wins even when the name would now match a different team."""
    conn = rebuild_db
    _seed_team(conn, "oe:team:old", "Cloud9")
    _seed_team(conn, "oe:team:new", "Cloud Nine")
    conn.execute(
        "INSERT INTO pandascore_team_map (ps_team_id, ps_name, team_id, matched_via, created_at)"
        " VALUES (103, 'Cloud Nine', 'oe:team:old', 'manual', ?)",
        (NOW,),
    )
    team = PandaTeam(ps_team_id=103, name="Cloud Nine")
    assert _resolve_one(conn, team) == "oe:team:old"


def test_ambiguous_name_stays_unmatched(rebuild_db):
    """Two OE teams sharing the name must never silently pick one."""
    conn = rebuild_db
    _seed_team(conn, "oe:team:a", "Mousesports")
    _seed_team(conn, "oe:team:b", "Mousesports")
    team = PandaTeam(ps_team_id=104, name="Mousesports")
    assert _resolve_one(conn, team) is None
    row = conn.execute("SELECT * FROM pandascore_team_map WHERE ps_team_id = 104").fetchone()
    assert row["team_id"] is None
    assert row["matched_via"] == "unmatched"


def test_unmatched_team_heals_once_the_oe_team_appears(rebuild_db):
    conn = rebuild_db
    team = PandaTeam(ps_team_id=105, name="Fresh Org")
    assert _resolve_one(conn, team) is None
    _seed_team(conn, "oe:team:fresh", "Fresh Org")
    assert _resolve_one(conn, team) == "oe:team:fresh"
    row = conn.execute(
        "SELECT matched_via FROM pandascore_team_map WHERE ps_team_id = 105"
    ).fetchone()
    assert row["matched_via"] == "name"


def test_map_team_manually_overrides_and_validates(rebuild_db):
    import pytest

    conn = rebuild_db
    _seed_team(conn, "oe:team:t1", "T1")
    team = PandaTeam(ps_team_id=106, name="T1 Esports Academy")
    assert _resolve_one(conn, team) is None
    map_team_manually(conn, 106, "oe:team:t1")
    assert _resolve_one(conn, team) == "oe:team:t1"
    with pytest.raises(ValueError, match="unknown team_id"):
        map_team_manually(conn, 106, "oe:team:nonexistent")
