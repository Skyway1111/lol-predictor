"""Facts seeders and region stubs shared by the three Impact test modules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from league_predictor.predraft.ratings.player_impact import RegionLevels
    from league_predictor.predraft.ratings.region_rating import RegionRatingTable

_ROLES = ("top", "jng", "mid", "bot", "sup")

#: Region log-odds used across the fixtures; ``_MINOR`` is a tier-2 anchor level.
_LCK, _LEC, _MINOR = 0.5, 0.1, -1.7


def _levels(**level_of: float) -> RegionLevels:
    """A RegionLevels whose unseen players fall to the weakest-tier anchor."""
    from league_predictor.predraft.ratings.player_impact import RegionLevels

    return RegionLevels(level=dict(level_of), default_level=_MINOR)


def _production_region_logodds() -> RegionRatingTable:
    """A stub region table with distinct major-pool levels and learned tier levels."""
    from league_predictor.predraft.ratings.region_rating import RegionRatingTable

    return RegionRatingTable(
        r={"LCK": 0.5, "LCS": 0.1, "LEC": 0.2, "LPL": 0.4},
        tier_effect={1: 0.0, 2: -1.6, 3: -2.4},
        home={},
    )


def _facts_conn():
    import sqlite3

    from league_predictor.db.schema import apply_current_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    return conn


def _seed_game(conn, *, gid, day, team1, team2, winner, league="LCK", blue=None, roles=None):
    """One one-game series (canonical team_a<team_b) with ``{team}_{role}`` rosters.

    Args:
        roles: ``{team: (role, ...)}`` overriding that team's five slots — the facts DB
            carries lineups narrower than five, and they must price region the same way.
    """
    team_a, team_b = sorted((team1, team2))
    blue_team = blue or team_a
    conn.execute("INSERT OR IGNORE INTO leagues (league_id, name) VALUES (?, ?)", (league, league))
    for t in (team_a, team_b):
        conn.execute("INSERT OR IGNORE INTO teams (team_id) VALUES (?)", (t,))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "best_of, winner, status) VALUES (?, ?, ?, ?, ?, 1, ?, 'completed')",
        (f"ser_{gid}", league, team_a, team_b, f"{day}T10:00:00+00:00", winner),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner, blue_team) VALUES (?, ?, 1, ?, ?)",
        (gid, f"ser_{gid}", winner, blue_team),
    )
    for team in (team1, team2):
        for role in (roles or {}).get(team, _ROLES):
            pid = f"{team}_{role}"
            conn.execute("INSERT OR IGNORE INTO players (player_id) VALUES (?)", (pid,))
            conn.execute(
                "INSERT INTO game_players (game_id, team_id, player_id, role) VALUES (?, ?, ?, ?)",
                (gid, team, pid, role),
            )
    conn.commit()


def _zero_priors(_conn, _as_of, player_ids, **_):
    return dict.fromkeys(player_ids, 0.0)
