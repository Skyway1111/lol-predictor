"""The Served OOS walk always executes the fixed production formula."""

from __future__ import annotations

import sqlite3
from datetime import date
from types import SimpleNamespace

from league_predictor.predraft.pipeline import served_oos


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE series (
            series_id TEXT PRIMARY KEY,
            scheduled_ts TEXT,
            team_a TEXT,
            team_b TEXT
        );
        CREATE TABLE games (
            game_id TEXT PRIMARY KEY,
            series_id TEXT,
            game_no INTEGER,
            winner TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO series VALUES (?,?,?,?)",
        [
            ("s1", "2026-05-05T00:00:00", "AAA", "BBB"),
            ("s2", "2026-06-05T00:00:00", "CCC", "DDD"),
            ("future", "2026-07-15T00:00:00", "AAA", "DDD"),
        ],
    )
    conn.executemany(
        "INSERT INTO games VALUES (?,?,?,?)",
        [
            ("g1", "s1", 1, "AAA"),
            ("g2", "s2", 1, "DDD"),
            ("g3", "future", 1, "AAA"),
        ],
    )
    return conn


def test_expected_oos_ids_respect_the_exclusive_cutoff():
    conn = _db()
    assert served_oos.expected_oos_game_ids(
        conn,
        through=date(2026, 7, 1),
        months=3,
    ) == {"g1", "g2"}


def test_walk_routes_every_probability_through_the_production_transform(monkeypatch):
    conn = _db()
    calls = []
    context = SimpleNamespace(
        fams=[SimpleNamespace(name="recency")],
        indexes={},
        drop=frozenset(),
        corpus=None,
    )

    monkeypatch.setattr(served_oos, "bag_backend", object)
    monkeypatch.setattr(served_oos, "fit_window", lambda *_, **__: object())
    table = SimpleNamespace(with_betas=lambda *, cross_tier, within_tier1: None)  # noqa: ARG005 - mirrors RegionRatingTable.with_betas
    monkeypatch.setattr(served_oos, "fit_region_ratings", lambda _conn, *, through: table)  # noqa: ARG005 - mirrors fit_region_ratings's keywords
    monkeypatch.setattr(served_oos, "game_prob", lambda *_, **__: 0.7)
    monkeypatch.setattr(
        served_oos,
        "build_features",
        lambda team_a, team_b, _as_of, *, indexes, families: {  # noqa: ARG005 - mirrors serve.build_features's keywords
            "rating_logit_diff": 0.2 if team_a < team_b else -0.2,
            "diff_series_wins": 1.0 if team_a < team_b else -1.0,
        },
    )
    monkeypatch.setattr(served_oos, "predict_symmetric", lambda *_, **__: [0.7])

    def adjusted(
        probability, _league, _indexes, team_a, team_b, as_of, *, temperature, damp_facts_games
    ):
        assert temperature == served_oos.chosen_model().temperature
        assert damp_facts_games == served_oos.chosen_model().damp_facts_games
        calls.append((team_a, team_b, as_of))
        return probability

    monkeypatch.setattr(served_oos, "adjust_map_prob", adjusted)

    rows = list(
        served_oos.walk_served_oos(
            conn,
            through=date(2026, 7, 1),
            months=3,
            ctx=context,
        )
    )

    assert [row[0] for row in rows] == ["g1", "g2"]
    assert len(calls) == 2
