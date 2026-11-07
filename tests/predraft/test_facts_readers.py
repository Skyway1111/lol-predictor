"""Join-shape tests for the facts readers against hand-built facts-schema fixtures."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from league_predictor.db.schema import apply_current_schema
from tests.predraft.conftest import insert_game


@pytest.fixture
def facts_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    return conn


def _player_rows(conn, game_id, team_id, players, *, kills=5, deaths=5, assists=10):
    for role, pid in players:
        conn.execute(
            "INSERT INTO game_players (game_id, team_id, player_id, role, "
            "kills, deaths, assists) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (game_id, team_id, pid, role, kills, deaths, assists),
        )


_ROLES = ("top", "jng", "mid", "bot", "sup")


def _five(conn, game_id, team_id, suffix=""):
    _player_rows(conn, game_id, team_id, [(r, f"{team_id}{suffix}_{r}") for r in _ROLES])


def test_multi_map_series_share_the_series_day(facts_db):
    """A Bo3's maps all read date(series.scheduled_ts) -- one calendar day."""
    from league_predictor.predraft.cleaning import enriched_matches

    sid = insert_game(
        facts_db, game_id="g1", day="2026-01-10", team1="B", team2="A", winner="A", time="09:00:00"
    )
    insert_game(
        facts_db,
        game_id="g2",
        day="2026-01-10",
        team1="A",
        team2="B",
        winner="A",
        series_id=sid,
        game_no=2,
    )
    facts_db.commit()

    rows = enriched_matches(facts_db)
    assert [r["match_id"] for r in rows] == ["g1", "g2"]  # game_no order
    assert {str(r["date"])[:10] for r in rows} == {"2026-01-10"}
    # canonical orientation flows straight from the series row
    assert all((r["team1_id"], r["team2_id"]) == ("A", "B") for r in rows)


def test_scheduled_series_never_enter_reads(facts_db):
    """Game-less scheduled series and unplayed (winner-NULL) games stay out of every read."""
    from league_predictor.predraft.cleaning import enriched_matches

    insert_game(facts_db, game_id="g1", day="2026-01-10", team1="A", team2="B", winner="A")
    # scheduled series: no games at all
    facts_db.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "best_of, status) VALUES ('ser_up', 'LCK', 'A', 'B', "
        "'2026-02-01T13:00:00+00:00', 3, 'scheduled')"
    )
    # played series with an unfinished map (winner NULL)
    insert_game(facts_db, game_id="g2", day="2026-01-11", team1="A", team2="B", winner=None)
    facts_db.commit()

    rows = enriched_matches(facts_db)
    assert [r["match_id"] for r in rows] == ["g1"]


def test_canonical_order_start_series_game(facts_db):
    """Canonical facts order: series by start instant, maps within a series by game_no."""
    from league_predictor.predraft.cleaning import enriched_matches

    s2 = insert_game(
        facts_db,
        game_id="x1",
        day="2026-01-10",
        team1="A",
        team2="B",
        winner="A",
        series_id="ser_b",
        time="15:00:00",
    )
    insert_game(
        facts_db,
        game_id="x2",
        day="2026-01-10",
        team1="A",
        team2="B",
        winner="B",
        series_id=s2,
        game_no=2,
        time="15:00:00",
    )
    insert_game(
        facts_db,
        game_id="x0",
        day="2026-01-10",
        team1="A",
        team2="B",
        winner="A",
        series_id="ser_a",
        time="18:00:00",
    )  # later time, smaller id
    facts_db.commit()

    rows = enriched_matches(facts_db)
    # the 15:00 series (both maps) precedes the 18:00 one, whatever their ids
    assert [r["match_id"] for r in rows] == ["x1", "x2", "x0"]


def test_corpus_reads_facts_tables_day_grouped(facts_db):
    """Corpus day grouping, lineups, and the finished filter hold on the facts joins."""
    from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus

    sid = insert_game(
        facts_db, game_id="g1", day="2026-01-10", team1="B", team2="A", winner="A", duration_s=2000
    )
    insert_game(
        facts_db,
        game_id="g2",
        day="2026-01-10",
        team1="A",
        team2="B",
        winner="B",
        series_id=sid,
        game_no=2,
        duration_s=1900,
    )
    insert_game(
        facts_db, game_id="g3", day="2026-01-12", team1="A", team2="B", winner="A", duration_s=1800
    )
    for gid in ("g1", "g2", "g3"):
        _five(facts_db, gid, "A")
        _five(facts_db, gid, "B")
    facts_db.commit()

    corpus = build_corpus(facts_db, through=date(2026, 2, 1), norm_through=date(2026, 2, 1))
    assert corpus.signature[0] == 3
    assert [d.date.isoformat() for d in corpus.days] == ["2026-01-10", "2026-01-12"]
    assert corpus.days[0].match_ids == ["g1", "g2"]
    assert corpus.vocab.n_orgs == 2
    assert corpus.vocab.n_players == 10
    # gap: team A played 2026-01-10, then 2026-01-12 -> raw team gap 2 days
    assert float(corpus.days[1].team_gaps[0][0]) == 2.0
