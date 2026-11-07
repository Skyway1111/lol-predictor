"""Tests the ``recency`` family diffs are antisymmetric, correctly signed, and leakage-safe."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta

import pytest

from league_predictor.asof import AsOf
from league_predictor.predraft.features import build_features
from league_predictor.predraft.features.families.recency import RecencyFamily

# Fixed rosters keep game_players well-formed; load_competitive_maps LEFT-JOINs it.
_ROSTERS = {
    t: {role: f"{t}_{role}" for role in ("top", "jng", "mid", "bot", "sup")} for t in ("AAA", "BBB")
}

_FEATURES = {"recency_winrate20_diff", "recency_ema_diff", "recency_momentum_diff"}
_THROUGH = date(2026, 4, 15)
_AS_OF = date(2026, 4, 14)
_Q = AsOf.of(_AS_OF)


def _facts_game(conn, mid, d, a, b, winner, tournament="LCK", gamelength=1800):
    ta, tb = sorted((a, b))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, status) "
        "VALUES (?, ?, ?, ?, ?, 'completed')",
        (f"ser_{mid}", tournament, ta, tb, f"{d}T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
        "VALUES (?, ?, 1, ?, ?)",
        (mid, f"ser_{mid}", winner, gamelength),
    )


def _empty_db() -> sqlite3.Connection:
    """A schema-only DB with the two tables load_competitive_maps needs."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    from league_predictor.db.schema import apply_current_schema

    apply_current_schema(conn)
    return conn


def _insert(conn: sqlite3.Connection, *, n: int, d: date, a: str, b: str, winner: str) -> None:
    """Insert one rating-eligible map (gamelength >= 900, a real winner) + its game_players rows."""
    mid = f"m{n}"
    _facts_game(conn, mid, d.isoformat(), a, b, winner)
    for tid in (a, b):
        for role, pid in _ROSTERS[tid].items():
            conn.execute(
                "INSERT INTO game_players (game_id, team_id, player_id, role, "
                "kills, deaths, assists) VALUES (?,?,?,?,?,?,?)",
                (mid, tid, pid, role, 5, 5, 10),
            )


def _streak_db(extra: list[tuple[date, str]] | None = None) -> sqlite3.Connection:
    """AAA beats BBB on every game -> AAA is on a pure winning streak, BBB losing.

    Args:
        extra: late ``(date, winner)`` games appended after the streak, for leakage checks.
    """
    conn = _empty_db()
    d = _THROUGH - timedelta(days=40)
    n = 0
    while d < _THROUGH - timedelta(days=1):  # all games strictly before _AS_OF's eve
        _insert(conn, n=n, d=d, a="AAA", b="BBB", winner="AAA")
        n += 1
        d += timedelta(days=1)
    for ed, ewinner in extra or []:
        _insert(conn, n=n, d=ed, a="AAA", b="BBB", winner=ewinner)
        n += 1
    conn.commit()
    return conn


def test_recency_family_emits_exactly_three_named_diffs():
    fam = RecencyFamily()
    assert fam.name == "recency"
    assert set(fam.feature_names) == _FEATURES
    assert fam.categorical_names == []
    index = fam.build(_streak_db(), through=_THROUGH, query_dates=[_AS_OF])
    out = index.features("AAA", "BBB", _Q)
    assert set(out) == _FEATURES


def test_recency_features_are_antisymmetric():
    index = RecencyFamily().build(_streak_db(), through=_THROUGH, query_dates=[_AS_OF])
    ab = index.features("AAA", "BBB", _Q)
    ba = index.features("BBB", "AAA", _Q)
    for name in _FEATURES:
        assert ab[name] == pytest.approx(-ba[name])


def test_winning_streak_team_gets_positive_diffs():
    # AAA has won every game, BBB lost every game -> every form diff favours AAA.
    index = RecencyFamily().build(_streak_db(), through=_THROUGH, query_dates=[_AS_OF])
    ab = index.features("AAA", "BBB", _Q)
    assert ab["recency_winrate20_diff"] > 0
    assert ab["recency_ema_diff"] > 0
    # AAA winrate ~1, BBB ~0; winrate20 diff is the full span.
    assert ab["recency_winrate20_diff"] == pytest.approx(1.0)


def test_unknown_teams_give_neutral_zero_diffs():
    index = RecencyFamily().build(_empty_db(), through=_THROUGH, query_dates=[_AS_OF])
    out = index.features("NOPE", "ZILCH", _Q)
    for name in _FEATURES:
        assert out[name] == pytest.approx(0.0)


def test_games_on_or_after_as_of_do_not_leak():
    # Baseline: no late games. Then add games dated ON and AFTER as_of where BBB wins
    # -- if those leaked they would shrink AAA's positive winrate diff.
    base = RecencyFamily().build(_streak_db(), through=_THROUGH, query_dates=[_AS_OF])
    base_out = base.features("AAA", "BBB", _Q)

    leaky_extra = [(_AS_OF, "BBB"), (_AS_OF + timedelta(days=1), "BBB")]
    leaked = RecencyFamily().build(_streak_db(leaky_extra), through=_THROUGH, query_dates=[_AS_OF])
    leaked_out = leaked.features("AAA", "BBB", _Q)

    for name in _FEATURES:
        assert leaked_out[name] == pytest.approx(base_out[name])


def _series(conn, series_id: str, ts: str, winners: list[str]) -> None:
    """One AAA-vs-BBB series starting at ``ts`` with one map per winner, in map order."""
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, best_of, status)"
        " VALUES (?, 'LCK', 'AAA', 'BBB', ?, ?, 'completed')",
        (series_id, ts, len(winners)),
    )
    for game_no, winner in enumerate(winners, start=1):
        mid = f"{series_id}_g{game_no}"
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner, duration_s)"
            " VALUES (?, ?, ?, ?, 1800)",
            (mid, series_id, game_no, winner),
        )
        for tid in ("AAA", "BBB"):
            for role, pid in _ROSTERS[tid].items():
                conn.execute(
                    "INSERT INTO game_players (game_id, team_id, player_id, role, "
                    "kills, deaths, assists) VALUES (?,?,?,?,?,?,?)",
                    (mid, tid, pid, role, 5, 5, 10),
                )


def test_a_map_sees_its_own_earlier_maps_and_no_sibling_at_the_same_instant():
    """The one visibility rule in the timeline: own earlier maps in, same-instant siblings out."""
    conn = _streak_db()  # AAA won every earlier game
    ts = f"{_AS_OF}T10:00:00+00:00"
    _series(conn, "ser_own", ts, ["BBB", "BBB", "BBB"])  # the query is map three
    _series(conn, "ser_sib", ts, ["BBB"])  # a sibling series at the same instant
    _series(conn, "ser_late", f"{_AS_OF}T14:00:00+00:00", ["BBB"])  # later that day
    conn.commit()
    index = RecencyFamily().build(conn, through=_THROUGH, query_dates=[_AS_OF])

    pre_series = index.features("AAA", "BBB", AsOf.game(ts, "ser_own", 1))
    assert pre_series["recency_winrate20_diff"] == pytest.approx(1.0)  # nothing of today

    map_three = index.features("AAA", "BBB", AsOf.game(ts, "ser_own", 3))
    # last 20 visible for AAA: 18 streak wins + own maps one and two (both lost) -> 0.9
    assert map_three["recency_winrate20_diff"] == pytest.approx(0.9 - 0.1)
    assert map_three["recency_momentum_diff"] < pre_series["recency_momentum_diff"]

    # a day query sees none of today; a noon instant sees both 10:00 series whole (4 losses)
    assert index.features("AAA", "BBB", _Q) == pre_series
    noon = index.features("AAA", "BBB", AsOf.of(datetime(2026, 4, 14, 12, 0)))
    assert noon["recency_winrate20_diff"] == pytest.approx(0.8 - 0.2)


def test_build_features_unions_recency():
    index = RecencyFamily().build(_streak_db(), through=_THROUGH, query_dates=[_AS_OF])
    out = build_features("AAA", "BBB", _AS_OF, indexes={"recency": index}, families=["recency"])
    assert set(out) == _FEATURES
