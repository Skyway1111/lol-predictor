"""Dependency-leaf naive-UTC wall-clock bridge into the facts date domain."""

# Facts dates are naive-UTC (db.writes convention); routing every "now" through
# here keeps the operator's local zone out of walk-forward cutoffs and stamps.

from collections.abc import Iterator
from datetime import UTC, date, datetime

_DECEMBER = 12  # the month whose successor starts the next year


def utc_now() -> datetime:
    """Naive-UTC now, comparable to parsed facts timestamps."""
    return datetime.now(UTC).replace(tzinfo=None)


def utc_today() -> date:
    """Today as the UTC calendar day, the facts date domain."""
    return utc_now().date()


def utc_now_iso() -> str:
    """Current UTC timestamp with an explicit offset and seconds precision."""
    return utc_now().replace(tzinfo=UTC).isoformat(timespec="seconds")


def trailing_months(through: date, months: int) -> Iterator[tuple[date, date]]:
    """The last ``months`` months ending at ``through``'s, oldest first, as ``[start, end)``."""
    year, month = through.year, through.month
    starts: list[date] = []
    for _ in range(months):
        starts.append(date(year, month, 1))
        year, month = (year, month - 1) if month > 1 else (year - 1, 12)
    for start in reversed(starts):
        end = date(start.year + (start.month == _DECEMBER), start.month % 12 + 1, 1)
        yield start, end
