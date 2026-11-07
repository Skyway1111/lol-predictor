"""Process-wide memo for corpus builds keyed by transaction-visible facts."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Hashable
from typing import TypeVar

from league_predictor.db.facts import cache_revision

_T = TypeVar("_T")

_MEMO: dict[Hashable, object] = {}
_STATE: dict[str, tuple[sqlite3.Connection, int, int, int | None]] = {}


def memo(db: sqlite3.Connection, key: Hashable, build: Callable[[], _T]) -> _T:
    """``build()`` once per ``(db identity, key)``: immutable corpus state every reader shares."""
    revision = cache_revision(db)
    if revision is None:
        return build()
    identity = (db, *revision)
    if identity != _STATE.get("identity"):
        _MEMO.clear()  # one corpus state per process: an older state's builds are gone
        _STATE["identity"] = identity
    if key not in _MEMO:
        _MEMO[key] = build()
    return _MEMO[key]  # type: ignore[return-value]
