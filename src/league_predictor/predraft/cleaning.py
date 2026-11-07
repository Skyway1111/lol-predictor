"""The canonical cleaning stage: series identity plus the eligibility predicates."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING

from league_predictor.db.facts_prefix import RATING_REPLAY_INPUTS, cache_prefix_digests
from league_predictor.predraft.feature_cache import (
    NAMESPACE_ENRICHED_MATCHES,
    db_file_path,
    get_cache,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping


# A game shorter than this is a remake/forfeit artifact, not a real result.
_MIN_REAL_GAME_SECONDS = 900


def remake_reason(match: Mapping) -> str | None:
    """Why a row is a data artifact, or ``None`` if it is a real result.

    Returns:
        ``'short_game'`` when ``gamelength`` is present and < 900s; ``'zero_team_stat'`` when
        a team's kills AND deaths AND assists are all a PRESENT zero. A lone zero is a
        legitimate result, and NULL stats are a data gap, not a remake.
    """
    gamelength = match.get("gamelength")
    if gamelength is not None and gamelength < _MIN_REAL_GAME_SECONDS:
        return "short_game"
    for side in ("team1", "team2"):
        kda = [match.get(f"{side}_{stat}") for stat in ("kills", "deaths", "assists")]
        if all(v == 0 for v in kda):  # all present (not None) AND all zero
            return "zero_team_stat"
    return None


def competitive_map_eligible(match: Mapping) -> bool:
    """True for a decided, non-remake, competitive result — a row that may move ratings."""
    return bool(match.get("winner") and remake_reason(match) is None)


def _as_date(value) -> date:
    """Coerce a row's ``date`` (a ``datetime`` string or a ``date``) to date-only."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


# ----------------------------------------------------------------------------
# Enriched-row loader + remake audit (the bridge from raw tables to predicates)

# The K/D/A aggregate memo is a shared-feature-cache namespace keyed on the EXACT bounds
# plus the corpus signature, so a refresh re-keys instead of serving a stale aggregate.


def clear_enriched_cache() -> None:
    """Drop the ``enriched_matches`` memo alone — a long build calls this to release memory."""
    get_cache().clear_namespace(NAMESPACE_ENRICHED_MATCHES)


def enriched_matches(
    conn: sqlite3.Connection, *, start: str | None = None, end: str | None = None
) -> list[dict]:
    """Real finished matches enriched with per-team K/D/A, in the cleaning-predicate shape.

    Returns:
        Fresh per-row dicts, memoized per ``(db file, corpus signature, start, end)`` and
        bypassed for :memory:/temp DBs. ``start``/``end`` MUST be date-only: they compare as
        lexicographic prefixes of the ISO-datetime ``scheduled_ts``, half-open by calendar day.
    """
    path = db_file_path(conn)
    if path == str(id(conn)):
        # :memory:/temp DB: no stable file path, so connection identity is ambiguous
        # across close/reopen at a reused address -- never cache, always compute fresh.
        return _query_enriched_matches(conn, start, end)

    cache = get_cache()
    cutoff = date.fromisoformat(end) if end else date.max
    digest = cache_prefix_digests(conn, [cutoff], RATING_REPLAY_INPUTS)[cutoff]
    key = (path, digest, start, end)
    cached = cache.get(NAMESPACE_ENRICHED_MATCHES, key)
    if cached is None:
        cached = _query_enriched_matches(conn, start, end)
        cache.set(NAMESPACE_ENRICHED_MATCHES, key, cached)
    # Hand out independent dicts so a consumer that mutates a row never reaches the memo.
    return [dict(r) for r in cached]


def _query_enriched_matches(
    conn: sqlite3.Connection, start: str | None, end: str | None
) -> list[dict]:
    """The uncached query backing ``enriched_matches``: a window-filtered K/D/A aggregate.

    Returns:
        Rows on the shared enriched-row contract (``match_id``/``date``/``team1_id``/...),
        where ``team1_id``/``team2_id`` are the series' CANONICAL pair, ``team_a < team_b``.
    """
    where = "WHERE g.winner IS NOT NULL"
    params: list[str] = []
    if start is not None:
        where += " AND s.scheduled_ts >= ?"
        params.append(start)
    if end is not None:
        where += " AND s.scheduled_ts < ?"
        params.append(end)
    # The CTE repeats the SAME window filter so the K/D/A GROUP BY aggregates only the
    # window's rows; totals are read through joins to the filtered games, so results match.
    cte_where = where.replace("g.", "g2.").replace("s.", "s2.")
    rows = conn.execute(
        f"""
        WITH team_totals AS (
            SELECT gp.game_id, gp.team_id,
                   SUM(gp.kills) AS k, SUM(gp.deaths) AS d, SUM(gp.assists) AS a
            FROM game_players gp
            JOIN games g2 ON g2.game_id = gp.game_id
            JOIN series s2 ON s2.series_id = g2.series_id
            {cte_where}
            GROUP BY gp.game_id, gp.team_id
        )
        SELECT g.game_id AS match_id, s.scheduled_ts AS date,
               s.series_id AS series_id, g.game_no AS game_no,
               s.team_a AS team1_id, s.team_b AS team2_id, g.winner AS winner,
               s.league_id AS tournament, g.duration_s AS gamelength,
               t1.k AS team1_kills, t1.d AS team1_deaths, t1.a AS team1_assists,
               t2.k AS team2_kills, t2.d AS team2_deaths, t2.a AS team2_assists
        FROM games g
        JOIN series s ON s.series_id = g.series_id
        LEFT JOIN team_totals t1 ON t1.game_id = g.game_id AND t1.team_id = s.team_a
        LEFT JOIN team_totals t2 ON t2.game_id = g.game_id AND t2.team_id = s.team_b
        {where}
        ORDER BY s.scheduled_ts, s.series_id, g.game_no
        """,
        params + params,
    ).fetchall()
    return [dict(r) for r in rows]


def eligible_games_by_day(conn: sqlite3.Connection, *, through: date) -> dict[date, list[dict]]:
    """Rating-eligible enriched matches strictly before ``through``, grouped by calendar day."""
    by_day: dict[date, list[dict]] = defaultdict(list)
    for m in enriched_matches(conn, end=through.isoformat()):
        if competitive_map_eligible(m):
            by_day[_as_date(m["date"])].append(m)
    return by_day
