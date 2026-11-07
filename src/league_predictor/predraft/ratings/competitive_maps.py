"""Shared rating-data loader: rating-eligible competitive maps in reference orientation."""

from __future__ import annotations

import sqlite3
from datetime import date

from league_predictor.asof import AsOf
from league_predictor.predraft.cleaning import competitive_map_eligible, enriched_matches


def load_competitive_maps(
    conn: sqlite3.Connection, *, through: date
) -> list[tuple[AsOf, str, str, int]]:
    """Rating-eligible maps before ``through``, in reference (sorted) orientation.

    Args:
        through: exclusive, date-only, matching the splits' half-open windows.

    Returns:
        One ``(as_of, team_a, team_b, a_won)`` per eligible map, ``as_of`` the map's own
        position in the facts order and ``team_a <= team_b`` so ``a_won`` is relative to that
        canonical side, sorted by ``(as_of, team_a, team_b)``.
    """
    rows = enriched_matches(conn, end=through.isoformat())
    maps: list[tuple[AsOf, str, str, int]] = []
    for m in rows:
        if not competitive_map_eligible(m):
            continue
        a, b = str(m["team1_id"]), str(m["team2_id"])
        team_a, team_b = (a, b) if a <= b else (b, a)
        won_a = 1 if str(m["winner"]) == team_a else 0
        maps.append((AsOf.game(m["date"], m["series_id"], m["game_no"]), team_a, team_b, won_a))
    maps.sort(key=lambda r: (r[0].position, r[1], r[2]))
    return maps
