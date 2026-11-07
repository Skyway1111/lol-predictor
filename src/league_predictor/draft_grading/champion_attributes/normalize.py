"""Normalize champion names between Oracle's Elixir display style and canonical class-file style."""

from __future__ import annotations

import re

# game_players.champion_id display name -> Champion_classes.md canonical name.
# Only differing names live here; every other name passes through unchanged.
DISPLAY_TO_CANONICAL: dict[str, str] = {
    "Aurelion Sol": "AurelionSol",
    "Bel'Veth": "Belveth",
    "Cho'Gath": "Chogath",
    "Dr. Mundo": "DrMundo",
    "Fiddlesticks": "FiddleSticks",
    "Jarvan IV": "JarvanIV",
    "K'Sante": "KSante",
    "Kai'Sa": "Kaisa",
    "Kha'Zix": "Khazix",
    "Kog'Maw": "KogMaw",
    "LeBlanc": "Leblanc",
    "Lee Sin": "LeeSin",
    "Master Yi": "MasterYi",
    "Miss Fortune": "MissFortune",
    "Nunu & Willump": "Nunu",
    "Rek'Sai": "RekSai",
    "Renata Glasc": "Renata",
    "Tahm Kench": "TahmKench",
    "Twisted Fate": "TwistedFate",
    "Vel'Koz": "Velkoz",
    "Wukong": "MonkeyKing",
    "Xin Zhao": "XinZhao",
}

# Dominant pro form per champion with multiple class-file entries; the pipeline is
# dominant-form only. If pro meta shifts, edit this table, not the lookup logic.
FORM_DEFAULTS: dict[str, str] = {
    "Kayn": "Kayn_Red",
    "Karma": "Karma_Enchanter",
    "Senna": "Senna_Enchanter",
    "Shaco": "Shaco_Stab",
    "Varus": "Varus",  # dominant pro form is the unsuffixed base (identity)
}

# Champions absent from Champion_classes.md; they get a None class label at feature
# time rather than failing the build. Update when the class file is amended.
CLASS_FILE_GAPS: frozenset[str] = frozenset({"Heimerdinger", "Skarner"})


def to_canonical(display_name: str) -> str:
    """Canonical (class-file) form of a display name; identity passthrough, no form resolution."""
    return DISPLAY_TO_CANONICAL.get(display_name, display_name)


def resolve_form(display_name: str) -> str:
    """Canonical name for a base display name, taking the dominant form where variants exist."""
    if display_name in FORM_DEFAULTS:
        return FORM_DEFAULTS[display_name]
    return to_canonical(display_name)


# Lowercase-alphanumeric aliasing for the attribute/ability artifacts:
# artifact keys are display names; lookups normalize through these.
_CHAMPION_NAME_ALIASES = {
    "nunu": "Nunu & Willump",
    "nunu & willump": "Nunu & Willump",
    "nunuandwillump": "Nunu & Willump",
    "wukong": "Wukong",
    "monkeyking": "Wukong",
}

_LEAGUEOFGRAPHS_SLUG_ALIASES = {
    "Aurelion Sol": "aurelionsol",
    "Dr. Mundo": "drmundo",
    "Jarvan IV": "jarvaniv",
    "Lee Sin": "leesin",
    "Master Yi": "masteryi",
    "Miss Fortune": "missfortune",
    "Nunu & Willump": "nunu",
    "Renata Glasc": "renata",
    "Tahm Kench": "tahmkench",
    "Twisted Fate": "twistedfate",
    "Wukong": "monkeyking",
    "Xin Zhao": "xinzhao",
}


def normalize_champion_name(name: str) -> str:
    """Lowercase-alphanumeric key form — the artifact-lookup normalization."""
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def champion_to_leagueofgraphs_slug(name: str) -> str:
    """The champion's League of Graphs URL slug; alias table first, punctuation-strip fallback."""
    if name in _LEAGUEOFGRAPHS_SLUG_ALIASES:
        return _LEAGUEOFGRAPHS_SLUG_ALIASES[name]
    lowered = name.lower()
    lowered = lowered.replace("&", " ")
    lowered = lowered.replace("'", "")
    lowered = lowered.replace(".", "")
    lowered = re.sub(r"[^a-z0-9]+", "-", lowered)
    return lowered.strip("-")
