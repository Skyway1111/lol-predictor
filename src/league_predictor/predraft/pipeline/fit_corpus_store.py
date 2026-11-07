"""Content-addressed, facts-digest-validated storage for the served fit corpus and index blobs."""

from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from league_predictor import config
from league_predictor.db.facts_prefix import (
    SERVED_CORPUS_INPUTS,
    cache_prefix_digests,
)
from league_predictor.predraft.atomic_write import write_pickle

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Sequence

#: ``SnapshotKey.as_tuple``'s shape: cache version, families, columns, drop set, span start,
#: rating-artifact identity, feature-schema fingerprint.
SnapshotIdentity = tuple[
    int, tuple[str, ...], tuple[str, ...], tuple[str, ...], str, tuple | None, str
]

#: Bump when featurization row semantics change shape, so every persisted snapshot misses
#: instead of deserializing rows a current build would not produce.
_FIT_CORPUS_CACHE_VERSION = 5
_DIGEST_ENGINE = "facts-prefix-v2"


#: Snapshots kept per group at store time.
_KEEP = 2

#: Groups kept per kind by last use: a served fit holds its recipe's corpus, and one
#: spare keeps an ad-hoc config from evicting either.
_KEEP_GROUPS = 3


def _cache_dir() -> Path:
    """Return the configured snapshot directory."""
    return config.DATA_DIR / "cache"


class StoreKey(Protocol):
    """The schema half of a blob's identity: a comparable tuple and its filename bucket."""

    def as_tuple(self) -> tuple[object, ...]: ...
    @property
    def group(self) -> str: ...


@dataclass(frozen=True)
class SnapshotKey:
    """The schema half of a fit corpus's identity; the data half is its facts-prefix digest."""

    families: tuple[str, ...]  # family names, config order
    feature_names: tuple[str, ...]  # the corpus's columns (families' universe minus drop)
    drop: tuple[str, ...]  # sorted drop set
    start: str  # ISO span start (date.min for the uncapped served window)
    rating_artifact: tuple | None  # chosen rating file identity; None without learned_rating
    schema_fingerprint: str  # feature-schema version + content fingerprint (manifest)

    @classmethod
    def for_context(cls, fams: Sequence, *, drop: frozenset[str], start: date) -> SnapshotKey:
        """The key for one ``build_fit_context`` shape; a changed universe mints a new key."""
        from league_predictor.predraft.features.manifest import (
            SCHEMA_VERSION,
            feature_schema_fingerprint,
        )

        names = tuple(f.name for f in fams)
        # The rating artifact's identity rides along: a rating retrain changes that family's
        # values with no schema change.
        return cls(
            families=names,
            feature_names=tuple(fn for f in fams for fn in f.feature_names if fn not in drop),
            drop=tuple(sorted(drop)),
            start=start.isoformat(),
            rating_artifact=_rating_artifact_identity() if "learned_rating" in names else None,
            schema_fingerprint=f"{SCHEMA_VERSION}:{feature_schema_fingerprint()}",
        )

    def as_tuple(self) -> SnapshotIdentity:
        """The persisted/compared identity tuple (version constant folded in)."""
        return (
            _FIT_CORPUS_CACHE_VERSION,
            self.families,
            self.feature_names,
            self.drop,
            self.start,
            self.rating_artifact,
            self.schema_fingerprint,
        )

    @property
    def group(self) -> str:
        """The 12-hex filename bucket for this key."""
        return group_of(self.as_tuple())


def group_of(identity: tuple[object, ...]) -> str:
    """The 12-hex filename bucket of an identity tuple."""
    return hashlib.sha256(repr(identity).encode("utf-8")).hexdigest()[:12]


def find_extendable(
    conn: sqlite3.Connection, key: StoreKey, *, end: date, kind: str = "fit_corpus"
) -> tuple[dict | None, str]:
    """The newest still-valid snapshot for ``key`` with data-end <= ``end``, plus ``end``'s digest.

    Args:
        kind: the blob family's filename prefix; each kind is its own store.

    Returns:
        ``(payload | None, current facts digest at end)``. A candidate is skipped fail-closed
        when it is unreadable, its key tuple differs, or its recorded digest disagrees with
        this DB (a historical correction), so a returned payload always extends soundly.
    """
    candidates = _group_snapshots(key.group, through=end, kind=kind)
    digests = _facts_prefix_digests(conn, {end, *(d for _, d in candidates)})
    for path, data_end in candidates:  # newest data-end first
        payload = _load_payload(path)
        if payload is None or payload.get("key") != key.as_tuple():
            continue
        if payload.get("digest_engine") != _DIGEST_ENGINE:
            continue
        if payload.get("digest") != digests[data_end]:
            continue
        with contextlib.suppress(OSError):
            os.utime(path)  # a hit is a use: the prune ranks groups by it
        return payload, digests[end]
    return None, digests[end]


def store(
    key: StoreKey, *, end: date, digest: str, kind: str = "fit_corpus", **fields: object
) -> None:
    """Persist ``fields`` under ``key`` at ``end`` atomically, then prune the ``kind`` store."""
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{kind}_{key.group}_{end.isoformat()}.pkl"
    payload = {
        "key": key.as_tuple(),
        "end": end.isoformat(),
        "digest": digest,
        "digest_engine": _DIGEST_ENGINE,
        **fields,
    }
    write_pickle(path, payload)
    _prune(kind, current=key.group)


def _rating_artifact_identity() -> tuple | None:
    """``(resolved path, mtime_ns, size)`` of the chosen rating artifact, or ``None`` if absent."""
    from league_predictor.predraft.pipeline.chosen_skill_filter import SKILL_FILTER_ARTIFACT

    resolved = Path(SKILL_FILTER_ARTIFACT).resolve()
    try:
        st = resolved.stat()
    except OSError:
        return None
    return (str(resolved), st.st_mtime_ns, st.st_size)


def _snapshots(kind: str) -> dict[str, list[tuple[Path, date]]]:
    """Each group's ``kind`` snapshots as ``(path, data_end)``, newest data-end first."""
    groups: dict[str, list[tuple[Path, date]]] = defaultdict(list)
    for p in _cache_dir().glob(f"{kind}_*.pkl"):
        group, _, end = p.stem.removeprefix(f"{kind}_").rpartition("_")
        try:
            groups[group].append((p, date.fromisoformat(end)))
        except ValueError:
            continue
    for entries in groups.values():
        entries.sort(key=lambda pair: pair[1], reverse=True)
    return groups


def _group_snapshots(group: str, *, through: date, kind: str) -> list[tuple[Path, date]]:
    """The group's snapshots with data-end <= ``through``, newest first, as ``(path, data_end)``."""
    return [(p, d) for p, d in _snapshots(kind).get(group, []) if d <= through]


def _load_payload(path: Path) -> dict | None:
    """One snapshot payload, or ``None`` on any read/unpickle failure (never propagates)."""
    try:
        with path.open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - repo-written snapshot
    except (OSError, EOFError, pickle.PickleError, ImportError, AttributeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _prune(kind: str, *, current: str) -> None:
    """Keep ``_KEEP`` snapshots of ``current`` and of the most recently used other groups.

    A new recipe, rating artifact, or schema mints a new group and strands the old one, so
    every group past ``_KEEP_GROUPS`` by last use is dropped whole.
    """
    groups = _snapshots(kind)
    by_use = sorted(groups.keys() - {current}, key=lambda g: _last_use(groups[g]), reverse=True)
    kept = {current, *by_use[: _KEEP_GROUPS - 1]}
    for group, entries in groups.items():
        for path, _data_end in entries[_KEEP if group in kept else 0 :]:
            with contextlib.suppress(OSError):
                path.unlink()


def _last_use(entries: Iterable[tuple[Path, date]]) -> float:
    """The group's newest snapshot mtime: a store writes it and a hit touches it."""
    times = []
    for path, _data_end in entries:
        with contextlib.suppress(OSError):  # removed by a concurrent prune
            times.append(path.stat().st_mtime)
    return max(times, default=0.0)


# --------------------------------------------------------------------------- #
# Facts prefix digests — the data half of the snapshot identity.


def _facts_prefix_digests(conn, cutoffs: Iterable[date]) -> dict[date, str]:
    """Chained facts digests before each cutoff; one ~1M-row scan per ``build_fit_context``."""
    return cache_prefix_digests(conn, cutoffs, SERVED_CORPUS_INPUTS)
