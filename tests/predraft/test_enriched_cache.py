"""Equivalence tests proving the ``enriched_matches`` memo is invisible to results."""

from __future__ import annotations

import sqlite3

from league_predictor.predraft.cleaning import (
    _query_enriched_matches,
    clear_enriched_cache,
    enriched_matches,
)
from league_predictor.predraft.feature_cache import NAMESPACE_ENRICHED_MATCHES, get_cache
from tests.predraft.conftest import facts_seed

_MATCHES = [
    ("A", "2026-05-01", "X", "Y", "X", "LCK", 1800),
    ("B", "2026-05-02", "X", "Z", "X", "LCK", 1700),
    ("C", "2026-05-03", "Y", "Z", "Z", "LCK", 1600),
]
_PLAYER_GAMES = [
    ("A", "X", 6, 2, 10),
    ("A", "X", 4, 3, 10),
    ("A", "Y", 3, 12, 8),
    ("B", "X", 5, 5, 5),
    ("C", "Y", 7, 4, 9),
    ("C", "Z", 2, 8, 3),
]


def _seed(conn: sqlite3.Connection) -> None:
    facts_seed(conn, _MATCHES, _PLAYER_GAMES)


def _file_db(tmp_path) -> sqlite3.Connection:
    """A file-backed DB -- the cache only engages off :memory: (stable file path)."""
    conn = sqlite3.connect(str(tmp_path / "enrich.db"))
    _seed(conn)
    return conn


def test_identical_bounds_hit_cache_and_return_equal_rows(tmp_path):
    """Two calls with identical bounds: equal rows, exactly one cache entry created."""
    clear_enriched_cache()
    conn = _file_db(tmp_path)

    first = enriched_matches(conn, end="2026-06-01")
    n_after_first = get_cache().size(NAMESPACE_ENRICHED_MATCHES)
    second = enriched_matches(conn, end="2026-06-01")

    assert first == second  # byte-identical rows
    assert n_after_first == 1  # the first call populated the memo
    assert (
        get_cache().size(NAMESPACE_ENRICHED_MATCHES) == 1
    )  # the second was a pure hit, no new entry


def test_cached_rows_equal_fresh_from_db_compute(tmp_path):
    """The memoized result is byte-identical to a from-scratch query (no transform)."""
    clear_enriched_cache()
    conn = _file_db(tmp_path)

    cached = enriched_matches(conn, end="2026-06-01")
    fresh = _query_enriched_matches(conn, None, "2026-06-01")
    assert cached == fresh


def test_different_bounds_do_not_collide(tmp_path):
    """A narrower window is a distinct key -- it must not be served the wider rows."""
    clear_enriched_cache()
    conn = _file_db(tmp_path)

    wide = enriched_matches(conn, end="2026-06-01")  # all 3 matches
    narrow = enriched_matches(conn, end="2026-05-02")  # only match A (< 05-02)

    assert {r["match_id"] for r in wide} == {"A", "B", "C"}
    assert {r["match_id"] for r in narrow} == {"A"}
    assert get_cache().size(NAMESPACE_ENRICHED_MATCHES) == 2  # two distinct windows cached
    assert narrow == _query_enriched_matches(conn, None, "2026-05-02")


def test_each_call_returns_independent_dicts(tmp_path):
    """Mutating a returned row never corrupts the shared memo."""
    clear_enriched_cache()
    conn = _file_db(tmp_path)

    first = enriched_matches(conn, end="2026-06-01")
    first[0]["winner"] = "TAMPERED"
    second = enriched_matches(conn, end="2026-06-01")
    assert second[0]["winner"] != "TAMPERED"


def test_memory_db_bypasses_cache():
    """:memory: DBs have no stable file path, so they are never cached."""
    clear_enriched_cache()
    mem = sqlite3.connect(":memory:")
    _seed(mem)

    rows = enriched_matches(mem, end="2026-06-01")
    assert {r["match_id"] for r in rows} == {"A", "B", "C"}
    assert get_cache().size(NAMESPACE_ENRICHED_MATCHES) == 0  # nothing cached for the in-memory DB


def test_corpus_refresh_rekeys_no_stale_aggregate(tmp_path):
    """A new finished match moves the corpus signature, so the memo re-keys."""
    clear_enriched_cache()
    conn = _file_db(tmp_path)

    before = enriched_matches(conn, end="2026-06-01")
    assert {r["match_id"] for r in before} == {"A", "B", "C"}

    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "status) VALUES ('ser_D', 'LCK', 'X', 'Z', '2026-05-04T10:00:00+00:00', "
        "'completed')"
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
        "VALUES ('D', 'ser_D', 1, 'X', 1900)"
    )
    conn.commit()
    after = enriched_matches(conn, end="2026-06-01")
    assert {r["match_id"] for r in after} == {"A", "B", "C", "D"}  # fresh, not stale
    clear_enriched_cache()
