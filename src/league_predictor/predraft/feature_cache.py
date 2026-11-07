"""Namespaced, size-bounded feature caches with connection-token invalidation."""

from __future__ import annotations

import sqlite3
from collections.abc import Hashable
from typing import Any

from league_predictor.db.facts import cache_revision
from league_predictor.db.facts_prefix import main_db_path

# Namespace constants for cache isolation (the player-prefix namespace lives beside its owner)
NAMESPACE_ENRICHED_MATCHES = "enriched_matches"


class FeatureCache:
    """Namespace-isolated, size-bounded cache with wholesale-namespace eviction."""

    # A value's type is the NAMESPACE's, not the cache's (a player prefix beside a row
    # list): get() hands back what the writer stored.

    def __init__(self, max_size: int = 50_000):
        """``max_size`` bounds total entries across all namespaces before eviction."""
        self._max_size = max_size
        self._stores: dict[str, dict[Hashable, Any]] = {}
        self._token_state: tuple[sqlite3.Connection, object] | None = None

    def make_token(self, conn: sqlite3.Connection) -> tuple[object, Hashable]:
        """Unique connection identity + visible facts; retain only the latest connection."""
        state = self._token_state
        if state is None or state[0] is not conn:
            state = (conn, object())
            self._token_state = state
        revision = cache_revision(conn)
        return state[1], revision if revision is not None else object()

    def _get_store(self, namespace: str) -> dict[Hashable, Any]:
        """Get or create the cache store for a namespace."""
        if namespace not in self._stores:
            self._stores[namespace] = {}
        return self._stores[namespace]

    def size(self, namespace: str | None = None) -> int:
        """Entries in one namespace, or across every namespace when ``namespace`` is None."""
        if namespace is None:
            return sum(len(store) for store in self._stores.values())
        return len(self._stores.get(namespace, {}))

    def _evict_if_full(self) -> None:
        """Drop whole namespaces, largest first, until entries fall to 75% of ``max_size``."""
        if self.size() <= self._max_size:
            return
        target = int(self._max_size * 0.75)
        while self.size() > target and self._stores:
            largest = max(self._stores, key=lambda ns: len(self._stores[ns]))
            del self._stores[largest]

    # sightline-ok: 1 - value type is the namespace's (see class docstring), never the cache's
    def get(self, namespace: str, key: Hashable) -> Any | None:
        """Return the cached value for (namespace, key), or None if absent."""
        store = self._stores.get(namespace)
        if store is None:
            return None
        return store.get(key)

    # sightline-ok: 1 - value type is the namespace's (see class docstring), never the cache's
    def set(self, namespace: str, key: Hashable, value: Any) -> None:
        """Store a value under (namespace, key), evicting first if over the size bound."""
        self._evict_if_full()
        store = self._get_store(namespace)
        store[key] = value

    def clear_namespace(self, namespace: str) -> None:
        """Drop one namespace's entries, leaving every other namespace warm."""
        self._stores.pop(namespace, None)


# Global singleton cache instance
_FEATURE_CACHE = FeatureCache(max_size=50_000)


def get_cache() -> FeatureCache:
    """Return the global feature cache singleton (import this, not _FEATURE_CACHE)."""
    return _FEATURE_CACHE


def db_file_path(conn: sqlite3.Connection) -> str:
    """File identity, or the connection identity for an in-memory database."""
    return main_db_path(conn) or str(id(conn))
