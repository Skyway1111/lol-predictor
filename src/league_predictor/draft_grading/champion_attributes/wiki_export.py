"""The wiki champion export: the one path + loader every builder and test reads."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_MODULE_DIR = Path(__file__).resolve().parent

# LOL_WIKI_CHAMPIONS_JSON points at an alternate wiki export per machine (the test suite
# reads the same env var); the in-repo export is the fallback.
DEFAULT_WIKI_EXPORT_PATH = Path(
    os.environ.get(
        "LOL_WIKI_CHAMPIONS_JSON",
        str(_MODULE_DIR / "data" / "champions.json"),
    )
)


# sightline-ok: 11 - a champion export read, not a model artifact
def load_wiki_champion_export(
    path: Path = DEFAULT_WIKI_EXPORT_PATH,
) -> dict[str, dict[str, Any]]:
    """The champion-keyed wiki export dict (``scripts/fetch_wiki_export.py``'s output)."""
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
