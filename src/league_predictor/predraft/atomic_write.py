"""Atomic pickle writes — the durable-write primitive every predraft disk cache uses."""

from __future__ import annotations

import os
import pickle
from pathlib import Path


def write_pickle(path: Path, payload: object) -> None:
    """Write ``payload`` to ``path`` via a temp sibling + ``Path.replace``; no half-written blob."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)  # a crashed write leaves no residue
