"""Corpus seeding shared by the champion-stat and game-feature tests."""

from __future__ import annotations

from datetime import date as calendar_date
from typing import TYPE_CHECKING

from league_predictor.draft_grading.grader.pro_excess import (
    HALF_LIFE_DAYS,
    SHRINKAGE_K,
    _baseline_map,
)

if TYPE_CHECKING:
    import sqlite3


def insert_game(
    db: sqlite3.Connection,
    match_id: str,
    date: str,
    champion: str,
    role: str,
    player_id: str,
    team_id: str,
    result: int,
    enemy_player: str | None = None,
) -> None:
    """One series + game + the champion's game_players row (blue side).

    Args:
        enemy_player: Optional single opponent on ``other_team``, giving the impact
            baseline a two-sided lineup.
    """
    winner = team_id if result else "other_team"
    series_id = f"ser_{match_id}"
    db.execute(
        "INSERT OR IGNORE INTO series (series_id, league_id, team_a, team_b, "
        "scheduled_ts, winner, status) VALUES (?, 'LCK', ?, 'other_team', ?, ?, 'completed')",
        (series_id, team_id, f"{date}T00:00:00+00:00", winner),
    )
    db.execute(
        "INSERT OR IGNORE INTO games (game_id, series_id, game_no, blue_team, "
        "red_team, winner) VALUES (?, ?, 1, ?, 'other_team', ?)",
        (match_id, series_id, team_id, winner),
    )
    db.execute(
        "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, result) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (match_id, player_id, team_id, champion, role, result),
    )
    if enemy_player is not None:
        db.execute(
            "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, "
            "result) VALUES (?, ?, 'other_team', 'EnemyChamp', ?, ?)",
            (match_id, enemy_player, role, 1 - result),
        )
    db.commit()


def sql_pro_stats(
    db: sqlite3.Connection, champion: str, role: str, as_of: str, *, baseline: str = "team_form"
) -> tuple[float, float, float]:
    """Independent event sum to catch prefix-cache boundary and decay regressions."""
    rows = db.execute(
        "SELECT gp.champion_id, gp.result, gp.team_id, gp.game_id, date(s.scheduled_ts) "
        "FROM game_players gp JOIN games g ON g.game_id=gp.game_id "
        "JOIN series s ON s.series_id=g.series_id "
        "WHERE gp.role=? AND date(s.scheduled_ts) < ?",
        (role, as_of),
    ).fetchall()
    forms = _baseline_map(db, baseline)
    cutoff = calendar_date.fromisoformat(as_of)
    role_weight = own_weight = residual = 0.0
    for name, won, team, game, played in rows:
        weight = 2 ** (-(cutoff - calendar_date.fromisoformat(played)).days / HALF_LIFE_DAYS)
        role_weight += weight
        if name == champion:
            own_weight += weight
            residual += weight * (won - forms.get((team, game), 0.5))
    return (
        residual / (own_weight + SHRINKAGE_K),
        own_weight / role_weight if role_weight else 0.0,
        own_weight,
    )
