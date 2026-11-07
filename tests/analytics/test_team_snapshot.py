"""Team-snapshot region resolution contracts."""

import sqlite3

import pytest

from league_predictor.analytics.team_snapshot import _team_regions


@pytest.mark.parametrize(
    ("league_id", "expected"),
    [
        ("LCKC", "LCKC"),
        ("NACL", "NACL"),
        ("LJLCS", "LJLCS"),
        ("LCSA", "LCSA"),
        ("LPLOL", "LPLOL"),
        ("DCup", "DCup"),
        ("IC", "IC"),
        ("LTA N", "LTA N"),
        ("LTA S", "LTA S"),
    ],
)
def test_team_region_resolves_known_leagues_exactly(league_id, expected):
    """The mart never infers a major region from a substring collision."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE series (league_id TEXT, team_a TEXT, team_b TEXT, scheduled_ts TEXT)"
    )
    conn.execute(
        "INSERT INTO series VALUES (?, 'teamA', 'teamB', '2025-01-01T00:00:00+00:00')",
        (league_id,),
    )

    assert _team_regions(conn, "2025-02-01").get("teamA") == expected


def test_team_region_filters_international_events_before_the_recency_limit():
    """A deep international run cannot hide the team's latest domestic league."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE series (league_id TEXT, team_a TEXT, team_b TEXT, scheduled_ts TEXT)"
    )
    conn.execute("INSERT INTO series VALUES ('LCK', 'teamA', 'teamB', '2025-01-01T00:00:00+00:00')")
    conn.executemany(
        "INSERT INTO series VALUES ('WLDs', 'teamA', 'teamB', ?)",
        [(f"2025-02-{day:02d}T00:00:00+00:00",) for day in range(1, 7)],
    )

    assert _team_regions(conn, "2025-03-01").get("teamA") == "LCK"
