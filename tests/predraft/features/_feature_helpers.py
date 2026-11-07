"""Synthetic-row builders and spies shared by the feature-engineering test modules."""

from tests.predraft.conftest import insert_roster

# --- Helpers ---


def _insert_match(
    conn,
    match_id,
    match_date,
    team1="teamA",
    team2="teamB",
    winner=None,
    fmt="Bo1",
    *,
    blue_team=None,
    gamelength=None,
):
    ts = str(match_date)
    ts = (ts.replace(" ", "T", 1) if len(ts) > 10 else ts + "T10:00:00") + "+00:00"
    ta, tb = sorted((team1, team2))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "best_of, status) VALUES (?, 'TestLeague', ?, ?, ?, ?, 'completed')",
        (f"ser_{match_id}", ta, tb, ts, int(fmt[2:]) if fmt else None),
    )
    red = None if blue_team is None else (team2 if blue_team == team1 else team1)
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner, "
        "duration_s) VALUES (?, ?, 1, ?, ?, ?, ?)",
        (match_id, f"ser_{match_id}", blue_team, red, winner, gamelength),
    )


def _insert_player_game(conn, match_id, player_id, team_id="teamA", result=1, **kwargs):
    defaults = {
        "champion_id": "Jax",
        "role": "top",
        "kills": 3,
        "deaths": 1,
        "assists": 5,
        "teamkills": 15,
        "teamdeaths": 8,
        "damageshare": 0.28,
        "killparticipation": 0.53,
        "deathshare": 0.12,
        "vspm": 1.8,
        "controlwardsbought": 4,
        "golddiffat15": 500.0,
        "csdiffat15": 12.0,
        "xpdiffat15": 350.0,
        "goldat15": 5500.0,
        "csat15": 142.0,
        "xpat15": 7200.0,
        "dpm": 650.0,
        "cspm": 8.5,
        "earnedgoldshare": 0.25,
        "firstblood": 1,
        "firsttower": 1,
        "firstdragon": 1,
        "firstbaron": 1,
    }
    defaults.update(kwargs)
    defaults.pop("gamelength", None)  # per-game duration lives on games.duration_s
    cols = ", ".join(["game_id", "player_id", "team_id", "result", *list(defaults.keys())])
    placeholders = ", ".join(["?"] * (4 + len(defaults)))
    conn.execute(
        f"INSERT INTO game_players ({cols}) VALUES ({placeholders})",
        [match_id, player_id, team_id, result, *list(defaults.values())],
    )


# --- build_pre_draft_features tests ---


def _setup_team(conn, team_id, roles=None):
    """Insert a minimal roster + series/games + game_players rows for a team."""
    if roles is None:
        roles = ["top", "jng", "mid", "bot", "sup"]
    for role in roles:
        player_id = f"{team_id}_{role}"
        insert_roster(conn, team_id, player_id, role, "2025-01-01")

    for i in range(5):
        mid = f"{team_id}_m{i}"
        match_date = f"2025-03-{i + 1:02d}"
        other = "teamB" if team_id == "teamA" else "teamA"
        _insert_match(conn, mid, match_date, team1=team_id, team2=other, winner=team_id)
        for role in roles:
            player_id = f"{team_id}_{role}"
            _insert_player_game(conn, mid, player_id, team_id=team_id, result=1)


# --- compute_objective_features tests ---


def _insert_team_game_stats(conn, match_id, team_id, **kwargs):
    defaults = {
        "dragons": 3,
        "barons": 1,
        "towers": 7,
        "heralds": 1,
        "void_grubs": 3,
        "firstdragon": 1,
        "firstbaron": 0,
        "firsttower": 1,
        "firstherald": 1,
        "totalgold": 55000,
        "totalcs": 800,
        "teamkills": 15,
        "teamdeaths": 10,
        "gamelength": 1920,
    }
    defaults.update(kwargs)
    cols = ", ".join(["game_id", "team_id", *list(defaults.keys())])
    placeholders = ", ".join(["?"] * (2 + len(defaults)))
    conn.execute(
        f"INSERT INTO game_team_stats ({cols}) VALUES ({placeholders})",
        [match_id, team_id, *list(defaults.values())],
    )
