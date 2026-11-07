"""Incrementally fetch and cache League of Graphs champion damage shares."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor.draft_grading.champion_attributes.wiki_export import (
    DEFAULT_WIKI_EXPORT_PATH,
    load_wiki_champion_export,
)
from scripts.champion_attributes.damage_shares import (
    DEFAULT_DAMAGE_CACHE_PATH,
    cache_damage_profile_entry,
    fetch_leagueofgraphs_damage_shares,
    load_damage_profile_cache,
    write_damage_profile_cache,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki-export",
        type=Path,
        default=DEFAULT_WIKI_EXPORT_PATH,
        help="Path to the wiki-derived champions.json export.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=DEFAULT_DAMAGE_CACHE_PATH,
        help="Path to write the League of Graphs damage-profile cache.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=20.0,
        help="Sleep between requests to avoid rate limiting.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of champion requests to attempt this run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refetch champions even if they already have a cached success entry.",
    )
    return parser.parse_args()


def _champion_names(wiki_export: Mapping[str, dict]) -> list[str]:
    return sorted(str(data.get("name") or key) for key, data in wiki_export.items())


def main() -> None:
    args = parse_args()
    wiki_export = load_wiki_champion_export(args.wiki_export)
    cache = load_damage_profile_cache(args.cache_path)

    attempted = 0
    succeeded = 0
    skipped = 0
    for champion_name in _champion_names(wiki_export):
        if attempted >= args.limit:
            break
        existing = (cache.get("profiles") or {}).get(champion_name, {})
        if not args.force and existing.get("status") == "ok":
            skipped += 1
            continue
        attempted += 1
        try:
            shares = fetch_leagueofgraphs_damage_shares(champion_name)
            if shares is None:
                cache_damage_profile_entry(
                    cache,
                    champion_name,
                    None,
                    status="missing",
                    error="No damage share parsed from page.",
                )
                print(f"[miss] {champion_name}")
            else:
                cache_damage_profile_entry(cache, champion_name, shares, status="ok")
                succeeded += 1
                physical, magic, true = shares
                print(f"[ok] {champion_name}: {physical:.3f}/{magic:.3f}/{true:.3f}")
        except Exception as exc:  # noqa: BLE001 - scrape loop: any per-champion failure is cached as status='error'
            cache_damage_profile_entry(
                cache,
                champion_name,
                None,
                status="error",
                error=str(exc),
            )
            print(f"[err] {champion_name}: {exc}")
        write_damage_profile_cache(args.cache_path, cache)
        if attempted < args.limit and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print(
        f"Attempted {attempted} champion fetches, succeeded on {succeeded}, "
        f"skipped {skipped} cached successes. Cache saved to {args.cache_path}."
    )


if __name__ == "__main__":
    main()
