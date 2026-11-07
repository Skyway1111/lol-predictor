"""Tests for the build_corpus disk cache: key identity, hits, rebuilds, and pruning."""

from __future__ import annotations

import dataclasses
import os
import sqlite3
from datetime import date

import pytest

from league_predictor import config
from league_predictor.predraft.ratings.skill_filter import corpus as corpus_mod
from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus, Vocab
from tests.predraft.conftest import facts_db_from_legacy
from tests.predraft.ratings.conftest import _five, _match

_THROUGH = date(2026, 2, 1)


def _toy_rows():
    """A two-day toy history (AA vs BB) with full five-role lineups — enough for a real corpus."""
    matches = [
        _match("m1", "2026-01-01", "AA", "BB", "AA"),
        _match("m2", "2026-01-10", "AA", "BB", "BB"),
    ]
    pgs = (
        _five("m1", "AA", "a")
        + _five("m1", "BB", "b")
        + _five("m2", "AA", "a")
        + _five("m2", "BB", "b")
    )
    return matches, pgs


def _file_db(path) -> sqlite3.Connection:
    """Open connection on a file-backed facts DB, so the cache has a real file to key on."""
    mem = facts_db_from_legacy(*_toy_rows())
    disk = sqlite3.connect(str(path))
    mem.backup(disk)
    mem.close()
    disk.row_factory = sqlite3.Row
    return disk


@pytest.fixture
def cache_in_tmp(tmp_path, monkeypatch):
    """Point data_dir at a tmp dir and clear _CACHE_DIR_OVERRIDE, exercising real resolution."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    return tmp_path


@pytest.fixture
def spy_build(monkeypatch):
    """Wrap ``_build_corpus_uncached`` with a call counter (the miss path); returns the list."""
    calls: list[int] = []
    real = corpus_mod._build_corpus_uncached

    def counting(conn, *, through, norm_through):
        calls.append(1)
        return real(conn, through=through, norm_through=norm_through)

    monkeypatch.setattr(corpus_mod, "_build_corpus_uncached", counting)
    return calls


def _corpus_equal(a: Corpus, b: Corpus) -> bool:
    """Deep equality on two corpora: signature, vocab, every day's tensors, and the norm stats."""
    import torch

    def fields_equal(x, y) -> bool:
        """Every dataclass field of ``x`` equals ``y``'s, comparing tensors elementwise."""
        return all(
            torch.equal(va, vb) if isinstance(va, torch.Tensor) else va == vb
            for va, vb in ((getattr(x, f.name), getattr(y, f.name)) for f in dataclasses.fields(x))
        )

    return (
        a.signature == b.signature
        and a.vocab == b.vocab
        and len(a.days) == len(b.days)
        and all(fields_equal(da, db) for da, db in zip(a.days, b.days, strict=True))
        and fields_equal(a.norm, b.norm)
    )


@pytest.mark.usefixtures("cache_in_tmp")
def test_cache_round_trip_equals_a_fresh_build(spy_build, tmp_path):
    """First call builds + caches; the second call hits the pickle (no rebuild) and deep-equals."""
    conn = _file_db(tmp_path / "facts.db")
    try:
        first = build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 1
        second = build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 1  # a cache HIT: no second build
        assert _corpus_equal(first, second)
    finally:
        conn.close()


@pytest.mark.usefixtures("cache_in_tmp")
def test_touched_mtime_preserves_a_content_cache_hit(spy_build, tmp_path):
    """Bumping the DB file mtime mints a fresh key, so the next build misses and rebuilds."""
    db_path = tmp_path / "facts.db"
    conn = _file_db(db_path)
    try:
        build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 1
        st = db_path.stat()
        os.utime(db_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
        build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 1
    finally:
        conn.close()


def test_memory_connection_bypasses_the_cache(cache_in_tmp, spy_build):
    """A :memory: DB has no file to key on: every call rebuilds, nothing is written."""
    conn = facts_db_from_legacy(*_toy_rows())
    try:
        build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 2  # bypass: no memoization
    finally:
        conn.close()
    cache_dir = cache_in_tmp / "data" / "cache"
    assert not cache_dir.exists() or not list(cache_dir.glob("*.pkl"))


def test_store_prunes_superseded_corpus_entries(cache_in_tmp):
    """A store prunes its DB identity's entries to the newest _CORPUS_CACHE_KEEP (~141MB each)."""
    import os
    import time

    cache_dir = cache_in_tmp / "data" / "cache"
    cache_dir.mkdir(parents=True)
    group = "a" * 12
    now = time.time()
    for i in range(4):
        p = cache_dir / f"corpus_{group}_{i:032x}.pkl"
        p.write_bytes(b"x")
        os.utime(p, (now - 100 + i, now - 100 + i))

    stored = cache_dir / f"corpus_{group}_{'f' * 32}.pkl"
    tiny = Corpus(days=[], vocab=Vocab({}, {}, {}, []), norm=None, signature=())
    corpus_mod._store_cached_corpus(stored, tiny)

    left = sorted(p.name for p in cache_dir.glob("corpus_*.pkl"))
    assert stored.name in left  # the just-stored entry survives
    assert len(left) == corpus_mod._CORPUS_CACHE_KEEP  # pruned to the keep budget
    assert f"corpus_{group}_{0:032x}.pkl" not in left  # the oldest orphans are gone
    assert f"corpus_{group}_{1:032x}.pkl" not in left


def test_prune_is_per_db_identity(cache_in_tmp, monkeypatch, tmp_path):
    """Pruning is per DB identity: one database's stores can never evict another's entries."""
    conn_a = _file_db(tmp_path / "facts_a.db")
    conn_b = _file_db(tmp_path / "facts_b.db")
    try:
        build_corpus(conn_b, through=_THROUGH, norm_through=_THROUGH)  # B's only entry
        # Four A-identity stores (distinct cutoff keys): one past the keep budget.
        for day in (1, 5, 10, 15):
            build_corpus(conn_a, through=_THROUGH, norm_through=date(2026, 1, day))
        cache_dir = cache_in_tmp / "data" / "cache"
        left = sorted(p.name for p in cache_dir.glob("corpus_*.pkl"))
        assert len(left) == corpus_mod._CORPUS_CACHE_KEEP + 1  # A pruned to keep; B intact
        # B's entry survived the A-side prunes: B's key still HITS (no rebuild).
        calls: list[int] = []
        real = corpus_mod._build_corpus_uncached

        def counting(conn, *, through, norm_through):
            calls.append(1)
            return real(conn, through=through, norm_through=norm_through)

        monkeypatch.setattr(corpus_mod, "_build_corpus_uncached", counting)
        build_corpus(conn_b, through=_THROUGH, norm_through=_THROUGH)
        assert calls == []  # cache hit: the other identity's prune never touched B
    finally:
        conn_a.close()
        conn_b.close()


def test_prune_removes_legacy_groupless_entries(cache_in_tmp):
    """Groupless ``corpus_<digest>.pkl`` names can never be hit, so the prune removes them."""
    cache_dir = cache_in_tmp / "data" / "cache"
    cache_dir.mkdir(parents=True)
    legacy = cache_dir / f"corpus_{0:032x}.pkl"
    legacy.write_bytes(b"x")
    keeper = cache_dir / f"corpus_{'a' * 12}_{1:032x}.pkl"
    keeper.write_bytes(b"x")

    corpus_mod._prune_superseded_corpus_entries(cache_dir)
    left = sorted(p.name for p in cache_dir.glob("corpus_*.pkl"))
    assert left == [keeper.name]


def test_prune_skips_a_concurrently_deleted_entry(cache_in_tmp, monkeypatch):
    """An entry deleted between the prune's glob and stat is skipped, never raised."""
    import pathlib
    import time

    cache_dir = cache_in_tmp / "data" / "cache"
    cache_dir.mkdir(parents=True)
    group = "a" * 12
    now = time.time()
    for i in range(4):
        p = cache_dir / f"corpus_{group}_{i:032x}.pkl"
        p.write_bytes(b"x")
        os.utime(p, (now - 100 + i, now - 100 + i))

    ghost = f"corpus_{group}_{2:032x}.pkl"  # "deleted" mid-prune, mtime unreadable
    real_stat = pathlib.Path.stat

    def racing_stat(self, **kwargs):
        if self.name == ghost:
            raise FileNotFoundError(self)
        return real_stat(self, **kwargs)

    monkeypatch.setattr(pathlib.Path, "stat", racing_stat)
    corpus_mod._prune_superseded_corpus_entries(cache_dir)  # must not raise
    left = sorted(p.name for p in cache_dir.glob("corpus_*.pkl"))
    assert ghost not in left  # sorted oldest -> pruned
    assert f"corpus_{group}_{3:032x}.pkl" in left  # the newest survivors kept
    assert len(left) == corpus_mod._CORPUS_CACHE_KEEP  # budget still honored


def test_uncached_build_releases_the_enriched_memo(tmp_path):
    """_build_corpus_uncached leaves the enriched_matches memo empty (~2.0 GB in production)."""
    from league_predictor.predraft.cleaning import clear_enriched_cache
    from league_predictor.predraft.feature_cache import NAMESPACE_ENRICHED_MATCHES, get_cache

    clear_enriched_cache()
    conn = _file_db(tmp_path / "facts.db")
    try:
        corpus = corpus_mod._build_corpus_uncached(conn, through=_THROUGH, norm_through=_THROUGH)
        assert corpus.signature[0] > 0  # the build really consumed rows
        assert get_cache().size(NAMESPACE_ENRICHED_MATCHES) == 0  # and released the memo behind it
    finally:
        conn.close()


@pytest.mark.usefixtures("cache_in_tmp")
def test_version_bump_invalidates(spy_build, tmp_path, monkeypatch):
    """Bumping CORPUS_CACHE_VERSION changes the key, so the next build misses."""
    conn = _file_db(tmp_path / "facts.db")
    try:
        build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 1
        monkeypatch.setattr(corpus_mod, "CORPUS_CACHE_VERSION", corpus_mod.CORPUS_CACHE_VERSION + 1)
        build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 2  # version bump -> new key -> rebuild
    finally:
        conn.close()


@pytest.mark.usefixtures("cache_in_tmp")
def test_committed_wal_update_rebuilds_corpus(spy_build, tmp_path):
    conn = _file_db(tmp_path / "facts.db")
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        first = build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        conn.execute("UPDATE games SET winner = 'BB' WHERE game_id = 'm1'")
        conn.commit()
        second = build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
        assert len(spy_build) == 2
        assert not _corpus_equal(first, second)
    finally:
        conn.close()
