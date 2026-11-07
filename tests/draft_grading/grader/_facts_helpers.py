from league_predictor.vocab.roles import ROLE_ORDER

"""Facts rows shared by temporal draft tests."""
BLUE = ["A_top", "A_jng", "A_mid", "A_bot", "A_sup"]

RED = ["B_top", "B_jng", "B_mid", "B_bot", "B_sup"]

BLUE_COMP = ["Aatrox", "LeeSin", "Azir", "Xayah", "Rakan"]

RED_COMP = ["Sion", "Viego", "Ahri", "Zeri", "Lulu"]


def _game(db, gid, day, league, blue_champs, red_champs, blue_wins):
    winner = "blue" if blue_wins else "red"
    db.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, winner, status)"
        " VALUES (?, ?, 'blue', 'red', ?, ?, 'completed')",
        (f"s_{gid}", league, f"{day}T00:00:00+00:00", winner),
    )
    db.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner)"
        " VALUES (?, ?, 1, 'blue', 'red', ?)",
        (gid, f"s_{gid}", winner),
    )
    for team, players, champs, result in (
        ("blue", BLUE, blue_champs, int(blue_wins)),
        ("red", RED, red_champs, int(not blue_wins)),
    ):
        for role, player, champ in zip(ROLE_ORDER, players, champs, strict=True):
            db.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, result)"
                " VALUES (?, ?, ?, ?, ?, ?)",
                (gid, player, team, champ, role, result),
            )
