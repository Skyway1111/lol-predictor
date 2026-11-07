"""Keep runtime configuration patchable after consumer modules are imported."""

from __future__ import annotations

import ast
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "league_predictor"


def test_package_consumers_import_the_config_module() -> None:
    """Direct symbol imports capture values and bypass the documented patch seam."""
    offenders: list[str] = []
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "league_predictor.config":
                names = ", ".join(alias.name for alias in node.names)
                relative = path.relative_to(PACKAGE_ROOT.parent).as_posix()
                offenders.append(f"{relative}:{node.lineno} imports {names}")

    assert not offenders, (
        "Import the config module and read its attributes at call time; direct imports "
        "capture stale values:\n" + "\n".join(offenders)
    )
