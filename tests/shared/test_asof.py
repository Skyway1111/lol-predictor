"""AsOf positions retain timestamps, ordering, and match context."""

from __future__ import annotations

from datetime import UTC, date, datetime

from league_predictor.asof import AsOf, MatchContext

_T = "2026-09-10T11:00:00+00:00"
_Q = AsOf.game(_T, "ser_q", 3)


def test_a_day_sees_only_earlier_days():
    q = AsOf.of(date(2026, 9, 10))
    assert q.key == "2026-09-10"
    assert q.day == date(2026, 9, 10)


def test_an_instant_sees_everything_started_before_it():
    q = AsOf.of(datetime(2026, 9, 10, 15, 30, tzinfo=UTC))
    assert q.ts == "2026-09-10T15:30:00+00:00"
    assert AsOf.of(datetime(2026, 9, 10, 15, 30)).ts == q.ts  # naive reads as UTC


def test_key_round_trips_and_orders_like_the_position():
    assert AsOf.parse(_Q.key) == _Q
    assert f"{_T}|ser_q|03" == _Q.key
    later = AsOf.game(_T, "ser_q", 10)
    assert later > _Q
    assert _Q.key < later.key
    assert AsOf.of(date(2026, 9, 10)) < _Q < AsOf.of(date(2026, 9, 11))
    assert sorted([later, _Q]) == [_Q, later]


def test_context_is_equality_not_order():
    ctx = MatchContext.of("Bo5", "Playoffs", {"T1": 2, "G2": 0})
    q = _Q.with_context(ctx)
    assert q != _Q
    assert not (q < _Q)
    assert not (q > _Q)
    assert q.position == _Q.position
    assert ctx.wins_for("T1") == 2
    assert ctx.wins_for("G2") == 0
    assert ctx.wins_for("DRX") == 0
    assert hash(q) != hash(_Q)


def test_of_is_idempotent_on_an_asof():
    assert AsOf.of(_Q) is _Q
