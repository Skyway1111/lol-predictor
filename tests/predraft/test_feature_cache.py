"""Tests for the shared FeatureCache and its connection-based tokens."""

import sqlite3

from league_predictor.predraft.feature_cache import FeatureCache


def test_make_token_stable_without_writes(full_db):
    cache = FeatureCache()
    assert cache.make_token(full_db) == cache.make_token(full_db)


def test_make_token_tracks_uncommitted_writes(full_db):
    cache = FeatureCache()
    before = cache.make_token(full_db)
    full_db.execute("INSERT INTO teams (team_id, name) VALUES ('teamA', 'Alpha')")
    assert cache.make_token(full_db) != before


def test_cache_releases_the_previous_connection(full_db):
    import gc
    import weakref

    class Connection(sqlite3.Connection):
        pass

    conn = sqlite3.connect(":memory:", factory=Connection)
    reference = weakref.ref(conn)
    cache = FeatureCache()
    cache.make_token(conn)
    conn.close()
    del conn
    cache.make_token(full_db)
    gc.collect()
    assert reference() is None
