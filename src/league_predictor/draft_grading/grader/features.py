"""Corpus role pools for draft alternatives."""

from __future__ import annotations

import sqlite3

from league_predictor.vocab.roles import ROLE_ORDER

DEFAULT_MIN_ROLE_GAMES = 15


def build_role_pools(
    db: sqlite3.Connection,
    *,
    min_role_games: int = DEFAULT_MIN_ROLE_GAMES,
    max_champions: int | None = None,
) -> dict[str, set[str]]:
    """Per-role set of champions played there in the corpus, over an all-time window.

    Args:
        min_role_games: count threshold that drops flex/one-off picks from a pool.
    """
    placeholders = ",".join("?" for _ in ROLE_ORDER)
    if max_champions is not None and max_champions < 1:
        raise ValueError(f"max_champions must be >= 1, got {max_champions}")
    rows = db.execute(
        f"""
        SELECT role, champion_id, COUNT(*) AS n
        FROM game_players
        WHERE role IS NOT NULL
          AND champion_id IS NOT NULL
          AND role IN ({placeholders})
        GROUP BY role, champion_id
        HAVING n >= ?
        ORDER BY role, n DESC, champion_id
        """,
        (*ROLE_ORDER, min_role_games),
    ).fetchall()
    pools: dict[str, set[str]] = {role: set() for role in ROLE_ORDER}
    for role, champ, _n in rows:
        pool = pools.setdefault(role, set())
        if max_champions is None or len(pool) < max_champions:
            pool.add(champ)
    return pools
