"""Shared toy-row builders and synthetic-corpus factories for the ratings tests."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from tests.predraft.conftest import facts_db_from_legacy

if TYPE_CHECKING:
    import sqlite3

_ROLES = ("top", "jng", "mid", "bot", "sup")


def _match(mid, day, t1, t2, winner, *, tournament="LCK", gn=1, patch="14.01", blue=None):
    """One 11-col match tuple for ``facts_db_from_legacy``; ``blue`` defaults to ``t1``."""
    return (
        mid,
        f"{day} 10:00:00",
        t1,
        t2,
        winner,
        tournament,
        1800,
        gn,
        patch,
        blue if blue is not None else t1,
        "regular",
    )


def _five(mid, tid, prefix, *, kills=2, gd15=100.0, vspm=1.1, side="Blue"):
    """Five 22-col player rows (one per role, ids ``<prefix>_<role>``) with simple stats."""
    return [
        (
            mid,
            tid,
            f"{prefix}_{role}",
            role,
            kills,
            1,
            3,
            50.0,
            5.0,
            80.0,
            gd15,
            10.0,
            120.0,
            0.2,
            vspm,
            0.2,
            10,
            7,
            1,
            1,
            0,
            side,
        )
        for role in _ROLES
    ]


def _weekly_conn(n_weeks=16, start=date(2026, 1, 1)) -> sqlite3.Connection:
    """AA beats BB every week -- a maximally learnable toy signal."""
    ms, pgs = [], []
    for w in range(n_weeks):
        day = start + timedelta(days=7 * w)
        mid = f"m{w}"
        ms.append(
            (mid, f"{day} 10:00:00", "AA", "BB", "AA", "LCK", 1800, 1, "14.01", "AA", "regular")
        )
        for tid, pref, gd in (("AA", "aa", 200.0), ("BB", "bb", -200.0)):
            pgs += [
                (
                    mid,
                    tid,
                    f"{pref}_{r}",
                    r,
                    2,
                    1,
                    3,
                    50.0,
                    5.0,
                    80.0,
                    gd,
                    10.0,
                    120.0,
                    0.2,
                    1.1,
                    0.2,
                    10,
                    7,
                    1,
                    1,
                    0,
                    "Blue",
                )
                for r in _ROLES
            ]
    return facts_db_from_legacy(ms, pgs)


def _mixed_conn(n_weeks=20, start=date(2026, 1, 1)) -> sqlite3.Connection:
    """AA beats BB in LCK; BB upsets AA at Worlds every 3rd week, AA keeping its box-score edge."""
    ms, pgs = [], []
    for w in range(n_weeks):
        day = start + timedelta(days=7 * w)
        intl = w % 3 == 2
        mid = f"m{w}"
        tour, winner = ("Worlds", "BB") if intl else ("LCK", "AA")
        ms.append(
            (mid, f"{day} 10:00:00", "AA", "BB", winner, tour, 1800, 1, "14.01", "AA", "regular")
        )
        for tid, pref, gd in (("AA", "aa", 200.0), ("BB", "bb", -200.0)):
            pgs += [
                (
                    mid,
                    tid,
                    f"{pref}_{r}",
                    r,
                    2,
                    1,
                    3,
                    50.0,
                    5.0,
                    80.0,
                    gd,
                    10.0,
                    120.0,
                    0.2,
                    1.1,
                    0.2,
                    10,
                    7,
                    1,
                    1,
                    0,
                    "Blue",
                )
                for r in _ROLES
            ]
    return facts_db_from_legacy(ms, pgs)


def _varied_conn(n_weeks: int = 16, start: date = date(2026, 1, 1)) -> sqlite3.Connection:
    """Role-varying weekly AA-beats-BB; the required builder for dispersion gradient pins."""
    ms, pgs = [], []
    for w in range(n_weeks):
        day = start + timedelta(days=7 * w)
        mid = f"m{w}"
        ms.append(_match(mid, day, "AA", "BB", "AA"))
        for tid, pref, sign in (("AA", "aa", 1.0), ("BB", "bb", -1.0)):
            for ri, role in enumerate(_ROLES):
                pgs.append(
                    (
                        mid,
                        tid,
                        f"{pref}_{role}",
                        role,
                        2 + ri,  # kills vary by role
                        1 + (ri % 2),
                        3 + ri,
                        45.0 + 3 * ri,
                        4.0 + ri,
                        70.0 + 5 * ri,
                        sign * (80.0 + 60.0 * ri),  # gd15 varies by role and side
                        8.0 + ri,
                        100.0 + 10 * ri,
                        0.15 + 0.02 * ri,
                        0.6 + 0.3 * ri,
                        0.2,
                        8 + ri,
                        6 + ri,
                        1,
                        1,
                        0,
                        "Blue",
                    )
                )
    return facts_db_from_legacy(ms, pgs)


def _conn() -> sqlite3.Connection:
    """Three AA-vs-BB maps (AA wins twice) plus a late CC-vs-DD map: four orgs, 60-day trend."""
    ms, pgs = [], []
    rows = (
        ("m1", "2026-01-01", "AA", "BB", "BB"),
        ("m2", "2026-02-05", "AA", "BB", "AA"),
        ("m3", "2026-03-12", "AA", "BB", "AA"),
        ("m4", "2026-03-20", "CC", "DD", "CC"),
    )
    for mid, day, t1, t2, w in rows:
        ms.append((mid, f"{day} 10:00:00", t1, t2, w, "LCK", 1800, 1, "14.01", t1, "regular"))
        for tid in (t1, t2):
            pgs += [
                (
                    mid,
                    tid,
                    f"{tid.lower()}_{r}",
                    r,
                    2,
                    1,
                    3,
                    50.0,
                    5.0,
                    80.0,
                    100.0,
                    10.0,
                    120.0,
                    0.2,
                    1.1,
                    0.2,
                    10,
                    7,
                    1,
                    1,
                    0,
                    "Blue",
                )
                for r in _ROLES
            ]
    return facts_db_from_legacy(ms, pgs)
