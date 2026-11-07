"""Per-role champion class labels parsed from ``Champion_classes.md``."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from league_predictor.draft_grading.champion_attributes.normalize import (
    CLASS_FILE_GAPS,
    resolve_form,
)

CLASS_FILE = Path(__file__).resolve().parents[1] / "Champion_classes.md"

UNKNOWN_CLASS = "Unknown"

# Hand-assigned clusters for the CLASS_FILE_GAPS champions. Roles left out here are
# roles they are effectively never played in, and resolve to UNKNOWN_CLASS.
MANUAL_CLASS_OVERRIDES: dict[tuple[str, str], str] = {
    ("Skarner", "top"): "Stat_Stick_Tanks",
    ("Skarner", "jng"): "Ult_Tanks",
    ("Skarner", "sup"): "Engage_Tanks",
    ("Heimerdinger", "top"): "Lane_Bullies",
    ("Heimerdinger", "mid"): "Pick_Mages",
    ("Heimerdinger", "bot"): "Mage_Bots",
    ("Heimerdinger", "sup"): "Mage_Supports",
}

# Role keys used in the class file's section headers map to ROLE_ORDER.
_SECTION_TO_ROLE = {
    "Jungle": "jng",
    "Top": "top",
    "Mid": "mid",
    "Bot": "bot",
    "Support": "sup",
}


def _section_role(line: str) -> str | None:
    """Role a ``## <Section>`` header opens; None for a section no role is clustered under."""
    for header, role in _SECTION_TO_ROLE.items():
        if line.startswith(f"## {header}"):
            return role
    return None


def _cluster_row(line: str) -> tuple[str, list[str]] | None:
    """``(class_label, champion tokens)`` for a cluster table row; None for any other line."""
    if not line.startswith("|") or line.startswith("|---"):
        return None
    cells = [c.strip() for c in line.split("|") if c.strip()]
    if len(cells) < 4:  # noqa: PLR2004 - the table's four columns
        return None
    if cells[0] == "ID":  # header row: | ID | Color | Coined label | Champions |
        return None
    # Class label cell is wrapped in **...** in the source markdown.
    label_match = re.fullmatch(r"\*\*(.+?)\*\*", cells[2])
    if not label_match:
        return None
    tokens = [raw.strip() for raw in cells[3].split(",")]
    return label_match.group(1), [t for t in tokens if re.fullmatch(r"[A-Za-z_]+", t)]


def _parse_class_file(path: Path = CLASS_FILE) -> dict[tuple[str, str], str]:
    """Parse the class markdown into ``{(canonical_champion_name, role): class_label}``."""
    text = path.read_text(encoding="utf-8")
    out: dict[tuple[str, str], str] = {}
    current_role: str | None = None
    for line in text.splitlines():
        if line.startswith("## "):
            current_role = _section_role(line)
            continue
        if current_role is None:
            continue
        row = _cluster_row(line)
        if row is None:
            continue
        label, champions = row
        for tok in champions:
            out[(tok, current_role)] = label
    return out


_MULTI_ROLE_PRIMARY: dict[str, str] = {
    "Aatrox": "top",
    "Akali": "mid",
    "Ambessa": "top",
    "Brand": "bot",
    "Diana": "jng",
    "DrMundo": "jng",
    "Ekko": "mid",
    "Elise": "sup",
    "Gwen": "top",
    "Irelia": "top",
    "Jax": "top",
    "Jayce": "top",
    "Lux": "sup",
    "Malphite": "top",
    "Mel": "mid",
    "Naafiri": "jng",
    "Pantheon": "sup",
    "Poppy": "sup",
    "Qiyana": "jng",
    "Ryze": "mid",
    "Senna_Damage": "bot",
    "Smolder": "bot",
    "Swain": "mid",
    "Sylas": "mid",
    "TahmKench": "sup",
    "Taliyah": "mid",
    "Talon": "jng",
    "Varus": "bot",
    "Vayne": "top",
    "Veigar": "mid",
    "Velkoz": "mid",
    "Vladimir": "top",
    "Volibear": "jng",
    "Warwick": "top",
    "Xerath": "mid",
    "Yasuo": "mid",
    "Yone": "mid",
    "Zaahen": "top",
    "Zed": "jng",
    "Zyra": "jng",
}

# Lazy-loaded on first access so importing this module doesn't crash
# if Champion_classes.md is absent (e.g. in minimal test environments).
_CHAMP_ROLE_CLASS: dict[tuple[str, str], str] = {}
_PRIMARY_CLUSTERED_ROLE: dict[str, str] = {}
CLASS_LABEL_TO_INDEX: dict[str, int] = {}
_loaded = False


def ensure_loaded() -> None:
    global _loaded  # noqa: PLW0603 - one-shot module loader flag
    if _loaded:
        return
    parsed = _parse_class_file()
    _CHAMP_ROLE_CLASS.update(parsed)

    roles_of: dict[str, set[str]] = defaultdict(set)
    for champ, role in _CHAMP_ROLE_CLASS:
        roles_of[champ].add(role)
    primary: dict[str, str] = {}
    for champ, roles in roles_of.items():
        if len(roles) == 1:
            primary[champ] = next(iter(roles))
        elif champ in _MULTI_ROLE_PRIMARY:
            primary[champ] = _MULTI_ROLE_PRIMARY[champ]
        else:
            raise ValueError(
                f"{champ} is clustered in {len(roles)} roles but missing "
                f"from _MULTI_ROLE_PRIMARY in classes.py — add it"
            )
    _PRIMARY_CLUSTERED_ROLE.update(primary)

    labels = [*sorted(set(_CHAMP_ROLE_CLASS.values())), UNKNOWN_CLASS]
    CLASS_LABEL_TO_INDEX.update({label: idx for idx, label in enumerate(labels)})
    _loaded = True


def effective_class(display_name: str, role: str) -> str:
    """Class label for ``(champion, role)`` — override, direct lookup, then primary-role borrow."""
    ensure_loaded()
    override = MANUAL_CLASS_OVERRIDES.get((display_name, role))
    if override is not None:
        return override
    if display_name in CLASS_FILE_GAPS:
        return UNKNOWN_CLASS
    canonical = resolve_form(display_name)
    direct = _CHAMP_ROLE_CLASS.get((canonical, role))
    if direct is not None:
        return direct
    borrow_role = _PRIMARY_CLUSTERED_ROLE.get(canonical)
    if borrow_role is not None:
        return _CHAMP_ROLE_CLASS[(canonical, borrow_role)]
    return UNKNOWN_CLASS
