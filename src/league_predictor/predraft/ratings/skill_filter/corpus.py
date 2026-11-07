"""The disk-cached ``build_corpus``: the memo in front of the corpus scan."""

from __future__ import annotations

import contextlib
import hashlib
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor import config
from league_predictor.db.facts_prefix import (
    RATING_REPLAY_INPUTS,
    cache_prefix_digests,
    main_db_path,
)
from league_predictor.predraft.atomic_write import write_pickle
from league_predictor.predraft.ratings.skill_filter.corpus_build import (
    _build_corpus_uncached,
)
from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

#: Bump on any ``corpus_schema`` change (a channel, a dataclass field, a dtype) so every
#: stale pickle misses and is rebuilt rather than deserialized to a wrong shape.
CORPUS_CACHE_VERSION = 4


def build_corpus(conn: sqlite3.Connection, *, through: date, norm_through: date) -> Corpus:
    """One chronological pass over rating-eligible maps before ``through``, disk-cached.

    Args:
        norm_through: bounds the games entering ``NormStats`` — training passes train_end,
            serving passes the artifact's recorded value.

    Returns:
        The corpus, memoized under ``data/cache/`` keyed by DB identity + cutoffs + version.
    """
    path = main_db_path(conn)
    identity = (
        (path, cache_prefix_digests(conn, [through], RATING_REPLAY_INPUTS)[through])
        if path
        else None
    )
    if identity is None:  # :memory: / unresolvable -> no key, no cache
        return _build_corpus_uncached(conn, through=through, norm_through=norm_through)
    cache_path = _corpus_cache_path(identity, through, norm_through)
    cached = _load_cached_corpus(cache_path)
    if cached is not None:
        return cached
    corpus = _build_corpus_uncached(conn, through=through, norm_through=norm_through)
    _store_cached_corpus(cache_path, corpus)
    return corpus


def _corpus_cache_path(identity: tuple[str, str], through: date, norm_through: date) -> Path:
    """The ``data/cache/`` pickle path for one (DB identity, cutoffs) key."""
    # The version constant is folded into the digest so a structure bump orphans old pickles; the
    # ``<group>`` prefix hashes the resolved DB path ALone so the prune can group per database.
    key = (CORPUS_CACHE_VERSION, identity, through.isoformat(), norm_through.isoformat())
    digest = hashlib.sha256(repr(key).encode("utf-8")).hexdigest()[:32]
    group = hashlib.sha256(identity[0].encode("utf-8")).hexdigest()[:12]
    cache_dir = config.DATA_DIR / "cache"
    return cache_dir / f"corpus_{group}_{digest}.pkl"


def _load_cached_corpus(cache_path: Path) -> Corpus | None:
    """Load a cached corpus, or ``None`` on a miss / unreadable / wrong-type pickle (→ rebuild)."""
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("rb") as fh:
            corpus = pickle.load(fh)  # noqa: S301 - repo-written cache pickle
    except (OSError, EOFError, pickle.PickleError, ImportError, AttributeError, ValueError):
        return None
    return corpus if isinstance(corpus, Corpus) else None


#: How many corpus cache entries survive a store, per DB identity. Entries are keyed on the DB
#: content, so correcting facts replaces the previous entry.
_CORPUS_CACHE_KEEP = 3


def _store_cached_corpus(cache_path: Path, corpus: Corpus) -> None:
    """Write ``corpus`` to its cache path atomically (temp sibling + ``replace``), then prune."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_pickle(cache_path, corpus)
    _prune_superseded_corpus_entries(cache_path.parent)


def _prune_superseded_corpus_entries(cache_dir: Path, keep: int = _CORPUS_CACHE_KEEP) -> None:
    """Drop all but the ``keep`` newest ``corpus_<group>_<digest>.pkl`` entries per DB identity."""

    def mtime(p: Path) -> float:
        try:
            return p.stat().st_mtime
        except OSError:  # deleted by a concurrent prune -> sorts oldest, unlink skips it
            return 0.0

    # Grouping by DB identity is load-bearing: a tiny test-DB run must never evict the real
    # corpus blobs. Group-less names can never be hit again, so they are removed.
    by_group: dict[str, list[Path]] = defaultdict(list)
    doomed: list[Path] = []
    for p in cache_dir.glob("corpus_*.pkl"):
        parts = p.stem.split("_")
        if len(parts) == 3:  # noqa: PLR2004 - corpus_<group>_<digest> has three parts
            by_group[parts[1]].append(p)
        else:
            doomed.append(p)
    for entries in by_group.values():
        entries.sort(key=mtime, reverse=True)
        doomed.extend(entries[keep:])
    for p in doomed:
        with contextlib.suppress(OSError):
            p.unlink()
