"""Champion damage-share knowledge: the LeagueOfGraphs cache + the stat fallback."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests

from league_predictor import clock
from league_predictor.draft_grading.champion_attributes.normalize import (
    champion_to_leagueofgraphs_slug,
)
from scripts.champion_attributes.paths import DATA_DIR

logger = logging.getLogger(__name__)

_MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DAMAGE_CACHE_PATH = DATA_DIR / "damage_profiles.json"


def load_damage_profile_cache(
    path: Path = DEFAULT_DAMAGE_CACHE_PATH,
) -> dict[str, Any]:
    """The cache dict (``meta`` + per-champion ``profiles``); fresh-shaped when absent."""
    if not path.exists():
        return {
            "meta": {
                "source": "leagueofgraphs",
                "created_at": clock.utc_now_iso(),
                "updated_at": clock.utc_now_iso(),
            },
            "profiles": {},
        }
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        "meta": dict(raw.get("meta", {})),
        "profiles": dict(raw.get("profiles", {})),
    }


def write_damage_profile_cache(path: Path, cache: Mapping[str, Any]) -> None:
    """Persist the cache with a refreshed ``updated_at`` stamp."""
    payload = {
        "meta": dict(cache.get("meta", {})),
        "profiles": dict(cache.get("profiles", {})),
    }
    payload["meta"]["updated_at"] = clock.utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def cache_damage_profile_entry(
    cache: dict[str, Any],
    champion_name: str,
    damage_shares: tuple[float, float, float] | None,
    *,
    status: str,
    error: str | None = None,
) -> None:
    """Record one fetch outcome (``ok``/``missing``/``error``) into the cache dict."""
    profiles = cache.setdefault("profiles", {})
    entry: dict[str, Any] = {
        "status": status,
        "fetched_at": clock.utc_now_iso(),
    }
    if damage_shares is not None:
        entry["physical"] = round(float(damage_shares[0]), 4)
        entry["magic"] = round(float(damage_shares[1]), 4)
        entry["true"] = round(float(damage_shares[2]), 4)
    if error:
        entry["error"] = error
    profiles[champion_name] = entry


def get_cached_damage_shares(
    cache: Mapping[str, Any] | None,
    champion_name: str,
) -> tuple[float, float, float] | None:
    """The cached (physical, magic, true) shares, or ``None`` without a complete entry."""
    if not cache:
        return None
    entry = (cache.get("profiles") or {}).get(champion_name)
    if not isinstance(entry, Mapping):
        return None
    required = ("physical", "magic", "true")
    if not all(key in entry for key in required):
        return None
    return (
        float(entry["physical"]),
        float(entry["magic"]),
        float(entry["true"]),
    )


def fetch_leagueofgraphs_damage_shares(
    champion_name: str,
    session: requests.Session | None = None,
    timeout: float = 10.0,
) -> tuple[float, float, float] | None:
    """Scrape one champion's damage split from LeagueOfGraphs; network, cache-refresh path only."""
    sess = session or requests.Session()
    slug = champion_to_leagueofgraphs_slug(champion_name)
    url = f"https://www.leagueofgraphs.com/champions/stats/{slug}"
    try:
        response = sess.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "league-predictor-attributes/0.1 (research tool)"},
        )
        response.raise_for_status()
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        logger.debug("LeagueOfGraphs request failed for %s: %s", champion_name, exc)
        return None

    text = response.text
    patterns = {
        "physical": r'tooltip="(\d+(?:\.\d+)?)%\s*Physical Damage"',
        "magic": r'tooltip="(\d+(?:\.\d+)?)%\s*Magic Damage"',
        "true": r'tooltip="(\d+(?:\.\d+)?)%\s*True Damage"',
    }
    extracted: dict[str, float] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        extracted[key] = float(match.group(1)) / 100.0
    total = extracted["physical"] + extracted["magic"] + extracted["true"]
    if total <= 0:
        return None
    return (
        extracted["physical"] / total,
        extracted["magic"] / total,
        extracted["true"] / total,
    )


#: (physical, magic, true) weight of an ability whose damageType declares one type.
_DECLARED_DAMAGE_WEIGHTS: dict[str, tuple[float, float, float]] = {
    "PHYSICAL_DAMAGE": (1.0, 0.0, 0.0),
    "MAGIC_DAMAGE": (0.0, 1.0, 0.0),
    "TRUE_DAMAGE": (0.0, 0.0, 1.0),
}

#: Text fallback for an ability with no usable damageType, first term found winning.
_TEXT_DAMAGE_WEIGHTS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("magic damage", (0.0, 0.75, 0.0)),
    ("physical damage", (0.75, 0.0, 0.0)),
    ("true damage", (0.0, 0.0, 0.75)),
)


def _ability_damage_weights(ability: Mapping[str, Any]) -> tuple[float, float, float]:
    """One ability's (physical, magic, true) weight: its declared type, else what its text names."""
    damage_type = str(ability.get("damageType") or "").upper()
    text = _ability_damage_text(ability)
    declared = _DECLARED_DAMAGE_WEIGHTS.get(damage_type)
    if declared is not None:
        return declared
    if damage_type == "OTHER_DAMAGE":
        # Mixed-damage ability: every type its text names takes a half share.
        return (
            0.5 if "physical damage" in text else 0.0,
            0.5 if "magic damage" in text else 0.0,
            0.5 if "true damage" in text else 0.0,
        )
    for term, weights in _TEXT_DAMAGE_WEIGHTS:
        if term in text:
            return weights
    return (0.0, 0.0, 0.0)


def fallback_damage_shares(champion: Mapping[str, Any]) -> tuple[float, float, float]:
    """Stat-heuristic shares for champions missing LoG data, normalized to sum 1."""
    physical = 0.0
    magic = 0.0
    true = 0.0
    adaptive = str(champion.get("adaptiveType") or "").upper()
    if adaptive == "PHYSICAL_DAMAGE":
        physical += 1.5
    elif adaptive == "MAGIC_DAMAGE":
        magic += 1.5
    else:
        physical += 0.75
        magic += 0.75

    for abilities in (champion.get("abilities") or {}).values():
        p, m, t = _ability_damage_weights((abilities or [{}])[0])
        physical += p
        magic += m
        true += t

    total = physical + magic + true
    if total <= 0:
        return (0.34, 0.33, 0.33)  # degenerate default — shares must sum to 1
    return (physical / total, magic / total, true / total)


def _ability_damage_text(ability: Mapping[str, Any]) -> str:
    """One ability's lowercased text corpus (blurb + notes + effects + leveling labels)."""
    parts: list[str] = []
    for key in ("blurb", "notes"):
        value = ability.get(key)
        if value:
            parts.append(str(value))
    for effect in ability.get("effects") or []:
        if isinstance(effect, Mapping):
            if effect.get("description"):
                parts.append(str(effect["description"]))
            parts.extend(
                str(leveling["attribute"])
                for leveling in effect.get("leveling") or []
                if isinstance(leveling, Mapping) and leveling.get("attribute")
            )
    return " ".join(parts).lower()
