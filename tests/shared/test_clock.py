"""The clock bridge yields naive-UTC values, comparable to facts-domain dates."""

from datetime import UTC, datetime, timedelta

from league_predictor import clock


def test_utc_today_is_the_utc_calendar_day():
    assert clock.utc_today() in {
        datetime.now(UTC).date(),
        (datetime.now(UTC) + timedelta(seconds=1)).date(),  # midnight-crossing slack
    }
