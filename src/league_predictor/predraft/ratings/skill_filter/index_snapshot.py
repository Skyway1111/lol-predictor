"""Verified skill-filter serving-index snapshots."""

from __future__ import annotations

import pickle
from datetime import date
from typing import TYPE_CHECKING

from league_predictor.predraft.atomic_write import write_pickle
from league_predictor.predraft.ratings.skill_filter.serving_bag import (
    BaggedSkillFilterIndex,
)

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

_FORMAT = 3


def snapshot_path(artifact_path: Path) -> Path:
    """The slot-adjacent replay snapshot path for ``rating_<slot>.pt``."""
    return artifact_path.with_name(f"{artifact_path.stem}.index.pkl")


def facts_identity(conn: sqlite3.Connection, *, through: date) -> str:
    """Return the stable facts-prefix identity for this sidecar."""
    from league_predictor.db.facts_prefix import (
        RATING_REPLAY_INPUTS,
        cache_prefix_digests,
    )

    digest = cache_prefix_digests(conn, [through], RATING_REPLAY_INPUTS, policy="file_snapshot")[
        through
    ]
    return f"facts:{digest}"


def save_index_snapshot(
    index: BaggedSkillFilterIndex, *, artifact_path: Path, artifact_sha: str, source_identity: str
) -> Path:
    """Atomically persist ``index`` with the identities required for safe reuse."""
    source_signature = getattr(index, "source_signature", None)
    through = getattr(index, "through", None)
    if source_signature is None or not isinstance(through, date):
        raise ValueError("skill-filter index lacks source_signature/through provenance")
    path = snapshot_path(artifact_path)
    payload = {
        "format": _FORMAT,
        "artifact_sha": artifact_sha,
        "source_signature": source_signature,
        "source_identity": source_identity,
        "through": through.isoformat(),
        "index": index,
    }
    write_pickle(path, payload)
    return path


def load_index_snapshot(
    *,
    artifact_path: Path,
    artifact_sha: str,
    source_signature: object,
    source_identity: str,
    through: date,
) -> BaggedSkillFilterIndex | None:
    """Return the verified bag index, or ``None`` for missing/stale/corrupt materialization."""
    try:
        with snapshot_path(artifact_path).open("rb") as fh:
            payload = pickle.load(fh)  # noqa: S301 - repo-written artifact, never untrusted input
    except (
        OSError,
        EOFError,
        pickle.PickleError,
        ImportError,
        AttributeError,
        ValueError,
        TypeError,
    ):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("format") != _FORMAT
        or payload.get("artifact_sha") != artifact_sha
        or payload.get("source_identity") != source_identity
        or not _source_covers(payload.get("source_signature"), source_signature)
    ):
        return None
    try:
        covered = date.fromisoformat(payload["through"])
    except (KeyError, TypeError, ValueError):
        return None
    index = payload.get("index")
    if covered < through or not isinstance(index, BaggedSkillFilterIndex):
        return None
    return index


def _source_covers(snapshot: object, requested: object) -> bool:
    """Whether the replay corpus is a superset of the requested corpus prefix."""
    # ``Corpus.signature`` is (row_count, latest_day, max_match_id); accept only a request whose
    # terminal id matches and whose count/day are no larger. Unknown shapes need exact equality.
    if snapshot == requested:
        return True
    if not (
        isinstance(snapshot, tuple)
        and isinstance(requested, tuple)
        and len(snapshot) == len(requested) == 3  # noqa: PLR2004 - the 3-field signature
        and isinstance(snapshot[0], int)
        and isinstance(requested[0], int)
        and isinstance(snapshot[1], str)
        and isinstance(requested[1], str)
    ):
        return False
    return (
        requested[0] <= snapshot[0] and requested[1] <= snapshot[1] and requested[2] == snapshot[2]
    )
