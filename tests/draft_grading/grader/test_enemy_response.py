"""Tests for draft_grading/grader/enemy_response.py — meta-frequency prior over enemy picks."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pytest

from league_predictor.draft_grading.grader import enemy_response


def _fixture_db() -> sqlite3.Connection:
    """Minimal corpus for the prior; ``scheduled_ts`` in stored format ``...T..:..:..+00:00``."""
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE series (series_id TEXT PRIMARY KEY, scheduled_ts TEXT NOT NULL);
        CREATE INDEX idx_series_scheduled ON series(scheduled_ts);
        CREATE TABLE games (game_id TEXT PRIMARY KEY, series_id TEXT NOT NULL);
        CREATE TABLE game_players (
            game_id TEXT NOT NULL,
            champion_id TEXT NOT NULL,
            role TEXT NOT NULL
        );
        """
    )
    # 5 games on 2026-04-15 (in window), 1 on 2026-03-01 (out of window for 30d).
    in_window = [
        ("m1", "2026-04-15T12:00:00+00:00"),
        ("m2", "2026-04-15T13:00:00+00:00"),
        ("m3", "2026-04-15T14:00:00+00:00"),
        ("m4", "2026-04-15T15:00:00+00:00"),
        ("m5", "2026-04-15T16:00:00+00:00"),
    ]
    out_of_window = [("m6", "2026-03-01T12:00:00+00:00")]
    for gid, ts in in_window + out_of_window:
        db.execute("INSERT INTO series VALUES (?, ?)", (f"ser_{gid}", ts))
        db.execute("INSERT INTO games VALUES (?, ?)", (gid, f"ser_{gid}"))

    # Picks at role 'top': Aatrox 3x, Camille 2x, Renekton 1x in window (+1 outside).
    picks: list[tuple[str, str, str]] = [(gid, "Aatrox", "top") for gid in ("m1", "m2", "m3")]
    picks += [(gid, "Camille", "top") for gid in ("m4", "m5")]
    picks.append(("m1", "Renekton", "top"))  # Renekton 1x in window
    picks.append(("m6", "Renekton", "top"))  # Renekton 1x outside window

    # Different role in same window: Lee Sin at jng. Must NOT show up in role='top' query.
    picks += [(gid, "LeeSin", "jng") for gid in ("m1", "m2")]

    db.executemany("INSERT INTO game_players (game_id, champion_id, role) VALUES (?, ?, ?)", picks)
    db.commit()
    return db


def test_top_k_weights_sum_to_one_on_fixture_db() -> None:
    db = _fixture_db()
    out = enemy_response.enemy_response_distribution(
        db, "top", datetime(2026, 5, 1), window_days=30, top_k=10
    )
    assert pytest.approx(sum(w for _, w in out), abs=1e-9) == 1.0


def test_top_k_orders_by_pick_frequency() -> None:
    db = _fixture_db()
    out = enemy_response.enemy_response_distribution(
        db, "top", datetime(2026, 5, 1), window_days=30, top_k=10
    )
    champs = [c for c, _ in out]
    # Aatrox 3, Camille 2, Renekton 1 → Aatrox first.
    assert champs[0] == "Aatrox"
    assert champs[1] == "Camille"
    assert champs[2] == "Renekton"


def test_role_filter_excludes_other_roles() -> None:
    db = _fixture_db()
    out = enemy_response.enemy_response_distribution(
        db, "top", datetime(2026, 5, 1), window_days=30, top_k=10
    )
    assert "LeeSin" not in {c for c, _ in out}


def test_date_window_excludes_picks_outside_window() -> None:
    db = _fixture_db()
    out = enemy_response.enemy_response_distribution(
        db, "top", datetime(2026, 5, 1), window_days=30, top_k=10
    )
    # Renekton has 1 pick in window + 1 outside; only the in-window one should count.
    renekton_weight = dict(out).get("Renekton")
    assert renekton_weight is not None
    # 1 pick out of total 6 in-window picks at role 'top' (3 Aatrox + 2 Camille + 1 Renekton).
    assert pytest.approx(renekton_weight, abs=1e-9) == 1 / 6


def test_top_k_caps_results() -> None:
    db = _fixture_db()
    out = enemy_response.enemy_response_distribution(
        db, "top", datetime(2026, 5, 1), window_days=30, top_k=2
    )
    assert len(out) == 2
    # Weights still sum to 1 over the kept top-K (renormalized).
    assert pytest.approx(sum(w for _, w in out), abs=1e-9) == 1.0


def test_empty_window_returns_empty_list() -> None:
    db = _fixture_db()
    out = enemy_response.enemy_response_distribution(
        db, "top", datetime(2020, 1, 1), window_days=30, top_k=10
    )
    assert out == []


def _edge_fixture_db() -> sqlite3.Connection:
    """Corpus with one distinctly-named pick at each second around both window boundaries."""
    db = _fixture_db()
    db.execute("DELETE FROM game_players")
    db.execute("DELETE FROM games")
    db.execute("DELETE FROM series")
    edges = [
        ("e_before_start", "AtSecondBeforeStart", "2026-03-31T23:59:59+00:00"),
        ("e_at_start", "AtWindowStart", "2026-04-01T00:00:00+00:00"),
        ("e_mid", "MidWindow", "2026-04-15T12:00:00+00:00"),
        ("e_last_second", "AtLastSecond", "2026-04-30T23:59:59+00:00"),
        ("e_at_as_of", "AtAsOf", "2026-05-01T00:00:00+00:00"),
    ]
    for gid, champ, ts in edges:
        db.execute("INSERT INTO series VALUES (?, ?)", (f"ser_{gid}", ts))
        db.execute("INSERT INTO games VALUES (?, ?)", (gid, f"ser_{gid}"))
        db.execute(
            "INSERT INTO game_players (game_id, champion_id, role) VALUES (?, ?, 'top')",
            (gid, champ),
        )
    db.commit()
    return db


def _reference_distribution(
    db: sqlite3.Connection,
    role: str,
    as_of: datetime,
    *,
    window_days: int,
    top_k: int,
) -> list[tuple[str, float]]:
    """The ``datetime()``-wrapped query: semantic reference for the string-range query."""
    start = as_of - timedelta(days=window_days)
    rows = db.execute(
        """
        SELECT gp.champion_id, COUNT(*) AS n
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE gp.role = ?
          AND datetime(s.scheduled_ts) >= datetime(?)
          AND datetime(s.scheduled_ts) <  datetime(?)
        GROUP BY gp.champion_id
        ORDER BY n DESC, gp.champion_id
        LIMIT ?
        """,
        (role, start.isoformat(sep=" "), as_of.isoformat(sep=" "), top_k),
    ).fetchall()
    if not rows:
        return []
    total = sum(n for _, n in rows)
    return [(champ, n / total) for champ, n in rows]


def test_sargable_query_matches_datetime_reference_at_day_edges() -> None:
    """String-range window equals the datetime() reference; start second IN, as_of second OUT."""
    db = _edge_fixture_db()
    as_of = datetime(2026, 5, 1)

    got = enemy_response.enemy_response_distribution(db, "top", as_of, window_days=30, top_k=10)
    expected = _reference_distribution(db, "top", as_of, window_days=30, top_k=10)

    assert got == expected
    champs = {c for c, _ in got}
    assert champs == {"AtWindowStart", "MidWindow", "AtLastSecond"}
    assert "AtSecondBeforeStart" not in champs
    assert "AtAsOf" not in champs


def test_sargable_query_matches_reference_on_main_fixture() -> None:
    """Same equivalence over the frequency-weighted fixture (ties, counts > 1)."""
    db = _fixture_db()
    as_of = datetime(2026, 5, 1)

    got = enemy_response.enemy_response_distribution(db, "top", as_of, window_days=30, top_k=10)
    expected = _reference_distribution(db, "top", as_of, window_days=30, top_k=10)

    assert got == expected


def test_cached_returns_same_object_on_repeat_call() -> None:
    db = _fixture_db()
    a = enemy_response.cached_enemy_response_distribution(
        db, "top", datetime(2026, 5, 1, 10, 0), window_days=30, top_k=10
    )
    b = enemy_response.cached_enemy_response_distribution(
        db, "top", datetime(2026, 5, 1, 10, 0), window_days=30, top_k=10
    )
    # Same date, same window — must be the cached object (identity, not just equality).
    assert a is b


def test_cache_keys_by_role() -> None:
    db = _fixture_db()
    a = enemy_response.cached_enemy_response_distribution(
        db, "top", datetime(2026, 5, 1), window_days=30, top_k=10
    )
    b = enemy_response.cached_enemy_response_distribution(
        db, "jng", datetime(2026, 5, 1), window_days=30, top_k=10
    )
    # Different role → different cache key → not the same object.
    assert a is not b
