"""Leak-free one-pass Impact history materialization."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from league_predictor.predraft.ratings.impact_corpus import build_impact_corpus
from league_predictor.predraft.ratings.impact_history import build_impact_history


def _seed_game(
    conn,
    *,
    game_id: str,
    day: str,
    team_a: str = "AA",
    team_b: str = "BB",
    winner: str = "AA",
    roster_a: tuple[str, ...] = ("a", "b"),
    roster_b: tuple[str, ...] = ("c", "d"),
) -> None:
    conn.execute("INSERT OR IGNORE INTO leagues (league_id, name) VALUES ('NEW-CUP', 'New')")
    for team_id in (team_a, team_b):
        conn.execute("INSERT OR IGNORE INTO teams (team_id) VALUES (?)", (team_id,))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "best_of, winner, status) VALUES (?, 'NEW-CUP', ?, ?, ?, 1, ?, 'completed')",
        (f"s-{game_id}", team_a, team_b, f"{day}T12:00:00+00:00", winner),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner) VALUES (?, ?, 1, ?)",
        (game_id, f"s-{game_id}", winner),
    )
    for team_id, roster in ((team_a, roster_a), (team_b, roster_b)):
        for index, player_id in enumerate(roster):
            conn.execute("INSERT OR IGNORE INTO players (player_id) VALUES (?)", (player_id,))
            conn.execute(
                "INSERT INTO game_players (game_id, team_id, player_id, role) VALUES (?, ?, ?, ?)",
                (game_id, team_id, player_id, ("top", "mid")[index % 2]),
            )
    conn.commit()


def test_build_impact_history_is_preupdate_complete_and_future_invariant(rebuild_db):
    _seed_game(rebuild_db, game_id="g1", day="2017-01-10")
    corpus = build_impact_corpus(rebuild_db)

    built = build_impact_history(corpus, through=date(2017, 1, 30), grid_days=14)

    assert built.eligible_games == 1
    assert built.influential_games == 1
    assert built.first_game_date == date(2017, 1, 10)
    assert built.last_game_date == date(2017, 1, 10)
    assert {row.asof_date for row in built.rows} == {
        date(2017, 1, 15),
        date(2017, 1, 29),
        date(2017, 1, 31),
    }
    first = {
        row.player_id: row.impact_logodds
        for row in built.rows
        if row.asof_date == date(2017, 1, 15)
    }
    assert first["a"] > 0.0
    assert first["b"] > 0.0
    assert first["c"] < 0.0
    assert first["d"] < 0.0
    assert built.rows[-1].asof_date == date(2017, 1, 31)  # terminal cutoff sees the tail

    _seed_game(rebuild_db, game_id="future", day="2027-01-01", roster_a=("x",), roster_b=("y",))
    mutated = build_impact_history(
        build_impact_corpus(rebuild_db),
        through=date(2017, 1, 30),
        grid_days=14,
    )
    assert mutated == built


def test_build_impact_history_rejects_data_before_the_locked_grid_epoch(rebuild_db):
    _seed_game(rebuild_db, game_id="too-old", day="2016-12-31")

    with pytest.raises(ValueError, match=r"2016-12-31.*grid epoch.*2017-01-01"):
        build_impact_history(build_impact_corpus(rebuild_db), through=date(2017, 1, 30))


def test_build_impact_history_rejects_duplicate_players_in_one_game(rebuild_db):
    _seed_game(
        rebuild_db,
        game_id="duplicate",
        day="2017-01-10",
        roster_a=("same", "b"),
    )
    corpus = build_impact_corpus(rebuild_db)
    lineups = np.array(corpus.lineups, copy=True)
    lineups[0, 0, 1] = lineups[0, 0, 0]
    lineups.setflags(write=False)

    with pytest.raises(ValueError, match=r"duplicate.*duplicate player"):
        build_impact_history(replace(corpus, lineups=lineups), through=date(2017, 1, 30))
