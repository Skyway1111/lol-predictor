"""Generate the 360-dim ability-feature artifact: prose read, contract applied."""

# Run directly:
#   python scripts/champion_attributes/generate_abilities.py

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Repo root (one above src/): the anchor for the scratch dumps below.
from league_predictor.draft_grading.champion_attributes.ability_schema import (
    ABILITY_SLOTS,
    ALL_FEATURE_KEYS,
    SCHEMA_VERSION,
)
from league_predictor.draft_grading.champion_attributes.wiki_export import (
    DEFAULT_WIKI_EXPORT_PATH,
    load_wiki_champion_export,
)
from scripts.champion_attributes.ability_features import assemble, extract_all_features
from scripts.champion_attributes.cc_labels import (
    audit_family_labels,
)
from scripts.champion_attributes.damage_shares import (
    fallback_damage_shares,
    get_cached_damage_shares,
    load_damage_profile_cache,
)
from scripts.champion_attributes.paths import DATA_DIR
from scripts.champion_attributes.reviewed_kit import (
    SOURCEABLE_DIMS,
    resolved_contracts,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_ARTIFACT_PATH = DATA_DIR / "ability_features.json"
_EDGE_CASES_PATH = DATA_DIR / "edge_cases.json"
_ROSTERS_PATH = DATA_DIR / "mechanic_rosters.json"
_ROSTER_PREVIEW = 10  # champions printed per mechanic roster before the elision

#: The schema-v3 keys whose rosters get audited (per-slot bases).
_AUDITED_KEYS = (
    "cc_type_charm",
    "cc_type_taunt",
    "cc_type_fear",
    "cc_type_sleep",
    "cc_type_knockback",
    "cc_type_pull",
    "cc_type_ground",
    "cc_type_disarm",
    "cc_type_polymorph",
    "has_untargetable",
    "has_invulnerable",
    "has_stasis",
    "has_spellshield",
    "has_projectile_block",
    "has_damage_reduction",
    "has_revive",
    "affects_allies",
    "has_ms_boost",
)


def resolve_damage_shares(
    cache: Mapping[str, Any] | None, display_name: str, champion: Mapping[str, Any]
) -> tuple[float, float, float]:
    """Cache → stat fallback; NEVER network (refresh lives in fetch_leagueofgraphs.py)."""
    cached = get_cached_damage_shares(cache, display_name)
    if cached is not None:
        return cached
    return fallback_damage_shares(champion)


def mechanic_rosters(artifact: Mapping[str, dict[str, float]]) -> dict[str, list[str]]:
    """new-key → champions-with-flag map (any slot), for the roster audit."""
    rosters: dict[str, list[str]] = {}
    for base in _AUDITED_KEYS:
        rosters[base] = sorted(
            name
            for name, features in artifact.items()
            if any(features.get(f"{slot}_{base}", 0.0) == 1.0 for slot in ABILITY_SLOTS)
        )
    return rosters


def apply_contract(
    prose: Mapping[str, dict[str, float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """Take the reviewed contract's reading over prose's, and record where each came from.

    Raises:
        SystemExit: a reviewed champion the wiki export has no row for, whose kit
            would read from prose alone.
    """
    contracts, unresolved = resolved_contracts(prose)
    if unresolved:
        raise SystemExit(
            f"the wiki export spells none of {', '.join(unresolved)}, which the "
            "calculator reviews; their kits would read from prose alone."
        )
    artifact: dict[str, dict[str, float]] = {}
    sources: dict[str, dict[str, str]] = {}
    for name, read in prose.items():
        artifact[name], source = assemble(name, read, contracts.get(name, {}))
        if source:
            sources[name] = source
    return artifact, sources


def source_report(
    artifact: Mapping[str, dict[str, float]], sources: Mapping[str, dict[str, str]]
) -> str:
    """Per-dim provenance over the whole store: what the agreement gate resolved."""
    counts = Counter(source for row in sources.values() for source in row.values())
    cells = sum(len(row) for row in artifact.values())
    reached = {dim for row in sources.values() for dim in row}
    sourceable = SOURCEABLE_DIMS & set(ALL_FEATURE_KEYS)
    silent = sum(
        1
        for name, row in artifact.items()
        for dim in sourceable
        if row.get(dim, 0.0) and dim not in sources.get(name, {})
    )
    return "\n".join(
        (
            f"{counts['contract']:6d}  cells the contract sources",
            f"{counts['exception']:6d}  cells a declared exception sources from the contract",
            f"{cells - sum(counts.values()):6d}  cells prose sources",
            f"{silent:6d}  of those on a family the contract can source and is silent on",
            f"{len(reached):6d}  distinct dims the contract reaches",
        )
    )


# sightline-ok: 11 - two standalone scripts' own CLIs; neither may grow the other's flags
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-export",
        type=Path,
        default=DEFAULT_WIKI_EXPORT_PATH,
        help="Path to the wiki-derived champions.json export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wiki_data = load_wiki_champion_export(args.wiki_export)
    # Fail closed before any extraction: a renamed control label must stop the build,
    # never leave its dim at zero across the whole roster.
    audit_family_labels(wiki_data)
    damage_cache = load_damage_profile_cache()
    print(f"Loaded {len(wiki_data)} champions from wiki export")

    edge_cases: dict[str, list[str]] = {}
    prose: dict[str, dict[str, float]] = {}
    fallback_shares = 0

    for champion_key in sorted(wiki_data):
        champion = wiki_data[champion_key]
        name = str(champion.get("name") or champion_key)
        if get_cached_damage_shares(damage_cache, name) is None:
            fallback_shares += 1
        shares = resolve_damage_shares(damage_cache, name, champion)
        prose[name] = extract_all_features(champion_key, champion, edge_cases, damage_shares=shares)

    artifact, sources = apply_contract(prose)

    n_features = len(ALL_FEATURE_KEYS)
    for name, features in artifact.items():
        if len(features) != n_features:
            print(f"  WARNING: {name} has {len(features)} features (expected {n_features})")

    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_PATH.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source_default": "prose",
                "champions": artifact,
                "sources": sources,
            },
            indent=2,
            sort_keys=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(artifact)} champions x {n_features} features to {_ARTIFACT_PATH} "
        f"({fallback_shares} damage-share stat fallbacks)"
    )
    print(source_report(artifact, sources))

    rosters = mechanic_rosters(artifact)
    _ROSTERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ROSTERS_PATH.write_text(json.dumps(rosters, indent=2) + "\n", encoding="utf-8")
    print(f"\nMechanic rosters ({_ROSTERS_PATH}):")
    for base, roster in rosters.items():
        shown = ", ".join(roster[:_ROSTER_PREVIEW])
        elision = "…" if len(roster) > _ROSTER_PREVIEW else ""
        print(f"  {base:<22} {len(roster):>3}: {shown}{elision}")

    if edge_cases:
        _EDGE_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        _EDGE_CASES_PATH.write_text(
            json.dumps(edge_cases, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nEdge cases ({len(edge_cases)} champions):")
        for champ, issues in sorted(edge_cases.items()):
            print(f"  {champ}: {'; '.join(issues)}")
    else:
        print("\nNo edge cases detected.")


if __name__ == "__main__":
    main()
