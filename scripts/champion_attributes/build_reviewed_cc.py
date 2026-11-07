"""Vendor the merged-calculator's reviewed per-slot control declarations as our drift oracle."""

# Run directly (the calculator is a sibling checkout, never a dependency):
#   python <this file> --calculator ../merged-calculator

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

from league_predictor import clock
from scripts.champion_attributes.calculator_dump import read_cli_dump
from scripts.champion_attributes.paths import DATA_DIR

_ARTIFACT_PATH = DATA_DIR / "reviewed_cc.json"

# Runs inside the calculator's own interpreter: its registry is the one derivation
# of a slot's control kind, so we read that rather than re-parsing MODULE_CC.
_DUMP_PROGRAM = """
import json, sys
sys.path.insert(0, "src")
from calculator.champions import (
    get_champion_module_contract, registered_champion_names, module_basename,
)
from calculator.control_spec import (
    ACTION_BLOCKING_CC_KINDS, CC_KIND_VOCABULARY, DISPLACEMENT_CC_KINDS,
)
out = {}
for name in registered_champion_names():
    contract = get_champion_module_contract(name)
    out[name] = {
        "slots": dict(contract.cc_kinds or {}),
        "module": module_basename(contract.module) + ".py",
    }
json.dump(
    {
        "champions": out,
        "action_blocking": sorted(ACTION_BLOCKING_CC_KINDS),
        "displacement": sorted(DISPLACEMENT_CC_KINDS),
        "vocabulary": sorted(CC_KIND_VOCABULARY),
    },
    sys.stdout,
)
"""


def build_artifact(payload: Mapping, calculator: Path) -> dict:
    """Shape the dump into the vendored artifact: declarations plus a per-module sha256."""
    modules = calculator / "src" / "calculator" / "champions"
    declarations: dict[str, dict[str, str]] = {}
    receipts: dict[str, str] = {}
    for name, entry in sorted(payload["champions"].items()):
        declarations[name] = dict(entry["slots"])
        module_path = modules / entry["module"]
        receipts[name] = hashlib.sha256(module_path.read_bytes()).hexdigest()
    return {
        "source": "merged-calculator champion modules (MODULE_CC via its own registry)",
        "generated": clock.utc_today().isoformat(),
        "vocabulary": payload["vocabulary"],
        "action_blocking": payload["action_blocking"],
        "displacement": payload["displacement"],
        "module_sha256": receipts,
        "declarations": declarations,
    }


def main() -> None:
    payload, calculator = read_cli_dump(_DUMP_PROGRAM)
    artifact = build_artifact(payload, calculator)
    _ARTIFACT_PATH.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    slots = sum(len(v) for v in artifact["declarations"].values())
    controlled = sum(
        1 for v in artifact["declarations"].values() for k in v.values() if k != "none"
    )
    print(
        f"Wrote {len(artifact['declarations'])} champions, {slots} slots "
        f"({controlled} carrying control) to {_ARTIFACT_PATH}"
    )


if __name__ == "__main__":
    sys.exit(main())
