"""Scrape a fresh champions.json wiki export from the LoL wiki (~170 champions, minutes-scale)."""

# Run directly:
#   python scripts/champion_attributes/fetch_wiki_export.py

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from scripts.champion_attributes.paths import DATA_DIR

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


from league_predictor.draft_grading.champion_attributes.lolstaticdata.pull_champions_wiki import (
    LolWikiDataHandler,
)
from league_predictor.draft_grading.champion_attributes.lolstaticdata.utils import (
    download_json,
    get_latest_patch_version,
)

_DEFAULT_OUT_PATH = DATA_DIR / "champions.json"


# sightline-ok: 11 - two standalone scripts' own CLIs; neither may grow the other's flags
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT_PATH,
        help="Path to write the champions.json export.",
    )
    return parser.parse_args()


def _build_champion_payload(champion: Any) -> dict[str, Any]:
    """Serialize one parsed champion, dropping the fields nothing here consumes."""
    payload = json.loads(champion.__json__(ensure_ascii=False))
    payload.pop("skins", None)
    payload.pop("lore", None)
    payload.pop("faction", None)
    return payload


def _bulk_pass(
    ddragon_champions: Mapping[str, Any],
    *,
    champions: list[dict[str, Any]],
    skipped: list[str],
    processed_keys: set[str],
) -> bool:
    """Parse every champion the bulk wiki generator yields, into the caller's accumulators.

    Returns:
        True when the generator itself crashed mid-stream — everything it yielded is fine,
        and the keys it never reached are the retry pass's work.
    """
    total = len(ddragon_champions)
    handler = LolWikiDataHandler(
        use_cache=False,
        process_stats=True,
        process_abilities=True,
        process_skins=False,
    )
    try:
        for champion in handler.get_champions():
            try:
                processed_keys.add(champion.key)
                if champion.key not in ddragon_champions:
                    continue
                champions.append(_build_champion_payload(champion))
                print(f"  [{len(champions)}/{total}] {champion.name}")
            except Exception as exc:  # noqa: BLE001 - one unparsable wiki champion is skipped, never fatal
                name = getattr(champion, "name", "Unknown")
                processed_keys.add(getattr(champion, "key", name))
                skipped.append(name)
                print(f"  skipped {name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - a crashed third-party generator must hand off to the retry pass
        print(f"  bulk parse error ({exc}) -- switching to per-champion mode")
        return True
    return False


def _retry_pass(
    remaining_keys: Iterable[str],
    total: int,
    *,
    champions: list[dict[str, Any]],
    skipped: list[str],
    processed_keys: set[str],
) -> None:
    """One wiki call per unreached champion — resilient, and slower: each re-parses the page."""
    for champion_key in remaining_keys:
        try:
            single_handler = LolWikiDataHandler(
                use_cache=True,
                target_champion=champion_key,
                process_stats=True,
                process_abilities=True,
                process_skins=False,
            )
            found = False
            for champion in single_handler.get_champions():
                found = True
                processed_keys.add(champion.key)
                champions.append(_build_champion_payload(champion))
                print(f"  [{len(champions)}/{total}] {champion.name}")
            if not found:
                print(f"  {champion_key}: in ddragon but not in wiki data")
        except Exception as exc:  # noqa: BLE001 - per-champion retry pass: a failure is skipped, never fatal
            skipped.append(champion_key)
            print(f"  skipped {champion_key}: {exc}")


def fetch_champions(latest_version: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Fetch and parse every champion from the wiki.

    Returns:
        ``(payloads, skipped names)`` — champions the bulk parse chokes on are retried
        individually so one unparseable champion cannot block the rest.
    """
    ddragon_champions = cast(
        "dict[str, Any]",
        download_json(
            f"http://ddragon.leagueoflegends.com/cdn/{latest_version}/data/en_US/championFull.json"
        ),
    )["data"]
    total = len(ddragon_champions)
    print(f"Processing {total} champions (patch {latest_version})...")

    champions: list[dict[str, Any]] = []
    skipped: list[str] = []
    processed_keys: set[str] = set()

    accumulators = {"champions": champions, "skipped": skipped, "processed_keys": processed_keys}
    if _bulk_pass(ddragon_champions, **accumulators):
        remaining = [key for key in ddragon_champions if key not in processed_keys]
        _retry_pass(remaining, total, **accumulators)

    return champions, skipped


def main() -> None:
    args = parse_args()
    latest_version = get_latest_patch_version()
    champions, skipped = fetch_champions(latest_version)

    champion_dict = {payload["key"]: payload for payload in champions}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        json.dump(champion_dict, handle, indent=2)

    print(f"Wrote {len(champion_dict)} champions to {args.out}")
    if skipped:
        print(f"Skipped {len(skipped)}: {', '.join(skipped)}")


if __name__ == "__main__":
    main()
