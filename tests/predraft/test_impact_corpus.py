"""Tests for the reusable Impact facts corpus."""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from league_predictor.predraft.ratings.impact_corpus import (
    build_impact_corpus,
    prepare_impact_window,
)


def _seed_game(
    conn,
    *,
    game_id: str,
    day: str,
    team_a: str = "A",
    team_b: str = "B",
    roster_a: tuple[tuple[str, str], ...] = (("a_mid", "mid"),),
    roster_b: tuple[tuple[str, str], ...] = (("b_mid", "mid"),),
    winner: str = "A",
    league: str = "LCK",
    scheduled_ts: str | None = None,
) -> None:
    """Seed one facts game; either roster may be empty to exercise legacy filtering."""
    conn.execute("INSERT OR IGNORE INTO leagues (league_id, name) VALUES (?, ?)", (league, league))
    for team in {team_a, team_b, winner}:
        conn.execute("INSERT OR IGNORE INTO teams (team_id) VALUES (?)", (team,))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "best_of, winner, status) VALUES (?, ?, ?, ?, ?, 1, ?, 'completed')",
        (
            f"s_{game_id}",
            league,
            team_a,
            team_b,
            scheduled_ts or f"{day}T10:00:00+00:00",
            winner,
        ),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner) VALUES (?, ?, 1, ?)",
        (game_id, f"s_{game_id}", winner),
    )
    for team, roster in ((team_a, roster_a), (team_b, roster_b)):
        for player_id, role in roster:
            conn.execute("INSERT OR IGNORE INTO players (player_id) VALUES (?)", (player_id,))
            conn.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, role) VALUES (?, ?, ?, ?)",
                (game_id, player_id, team, role),
            )


def test_prepare_impact_window_respects_boundaries_incomplete_games_and_roles(rebuild_db):
    """One full corpus owns cutoff grouping and deterministic role semantics."""
    _seed_game(rebuild_db, game_id="old", day="2025-12-31")
    _seed_game(rebuild_db, game_id="start", day="2026-01-01", winner="B")
    _seed_game(
        rebuild_db,
        game_id="inside",
        day="2026-04-01",
        roster_a=(("a_mid", "mid"), ("a_top", "top")),
        roster_b=(("b_mid", "mid"), ("b_top", "top")),
        league="NEW-2027-CUP",
    )
    _seed_game(rebuild_db, game_id="cutoff", day="2026-05-01")
    _seed_game(rebuild_db, game_id="incomplete", day="2026-03-01", roster_b=())
    # Role ties resolve by semantic priority, independent of insertion or chronology.
    _seed_game(
        rebuild_db,
        game_id="flex_later_inserted_first",
        day="2026-04-15",
        roster_a=(("flex", "mid"),),
    )
    _seed_game(
        rebuild_db,
        game_id="flex_earlier_inserted_second",
        day="2026-02-01",
        roster_a=(("flex", "sup"),),
    )
    _seed_game(
        rebuild_db,
        game_id="reverse_earlier",
        day="2026-02-02",
        roster_a=(("flex_reverse", "sup"),),
    )
    _seed_game(
        rebuild_db,
        game_id="reverse_later",
        day="2026-04-16",
        roster_a=(("flex_reverse", "mid"),),
    )

    as_of = date(2026, 5, 1)
    corpus = build_impact_corpus(rebuild_db)
    actual_games, actual_roles = prepare_impact_window(
        corpus,
        as_of_date=as_of,
        window_months=4,
    )

    assert len(actual_games) == 6
    assert all(game.days_before_asof > 0 for game in actual_games)
    assert actual_roles["flex"] == "mid"
    assert actual_roles["flex_reverse"] == "mid"
    assert set(corpus.league_ids) >= {"NEW-2027-CUP"}
    assert corpus.lineups.dtype == np.int32
    assert corpus.lineups.shape[1:] == (2, 5)
    assert corpus.n_appearances == 19

    full_games, _ = prepare_impact_window(corpus, as_of_date=as_of, window_months=None)
    assert len(full_games) == 7
    assert max(game.days_before_asof for game in full_games) > 120


def test_build_impact_corpus_refuses_a_lineup_wider_than_the_locked_shape(rebuild_db):
    roster = tuple((f"a{i}", "mid") for i in range(6))
    _seed_game(rebuild_db, game_id="wide", day="2026-04-01", roster_a=roster)

    with pytest.raises(RuntimeError, match=r"wide.*6.*maximum 5"):
        build_impact_corpus(rebuild_db)


def test_prepare_impact_window_uses_sqlite_calendar_date_for_cutoff(rebuild_db):
    """Offset timestamps are grouped by the canonical SQLite calendar date."""
    _seed_game(
        rebuild_db,
        game_id="offset",
        day="2026-05-01",
        scheduled_ts="2026-05-01T23:30:00-02:00",
    )
    as_of = date(2026, 5, 2)

    actual_games, _ = prepare_impact_window(
        build_impact_corpus(rebuild_db),
        as_of_date=as_of,
        window_months=1,
    )

    assert actual_games == []  # SQLite normalizes the offset timestamp to 2026-05-02 UTC.


def test_build_impact_corpus_ignores_unconsumed_table_columns(rebuild_db):
    """Impact corpus construction fails closed when a locked facts table drifts."""
    rebuild_db.execute("ALTER TABLE series ADD COLUMN surprise TEXT")

    assert build_impact_corpus(rebuild_db).n_appearances == 0


def test_build_impact_corpus_quarantines_noncanonical_appearance_team(rebuild_db):
    _seed_game(rebuild_db, game_id="bad-team", day="2026-04-01")
    rebuild_db.execute("INSERT INTO teams (team_id) VALUES ('X')")
    rebuild_db.execute("INSERT INTO players (player_id) VALUES ('rogue')")
    rebuild_db.execute(
        "INSERT INTO game_players (game_id, player_id, team_id, role) "
        "VALUES ('bad-team', 'rogue', 'X', 'mid')"
    )

    corpus = build_impact_corpus(rebuild_db)

    assert corpus.source_appearances == 3
    assert corpus.n_appearances == 2
    assert corpus.rejected_appearances == 1
    assert "rogue" not in corpus.player_ids


def test_build_impact_corpus_rejects_invalid_timestamp(rebuild_db):
    _seed_game(rebuild_db, game_id="bad-time", day="2026-04-01")
    rebuild_db.execute(
        "UPDATE series SET scheduled_ts = 'not-a-time' WHERE series_id = 's_bad-time'"
    )
    with pytest.raises(RuntimeError, match=r"bad-time.*invalid scheduled_ts"):
        build_impact_corpus(rebuild_db)
