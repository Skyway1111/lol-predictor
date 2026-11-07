"""Champion class resolution: clustered labels, manual overrides, the cross-role borrow."""

from __future__ import annotations

from collections import defaultdict

from league_predictor.draft_grading.champion_attributes.classes import (
    _CHAMP_ROLE_CLASS,
    _MULTI_ROLE_PRIMARY,
    MANUAL_CLASS_OVERRIDES,
    effective_class,
    ensure_loaded,
)
from tests.draft_grading._class_claims import champion_roles

ensure_loaded()

_ROLES = ("top", "jng", "mid", "bot", "sup")


def _roles_of() -> dict[str, set[str]]:
    """``{canonical champion: set of roles machineloling clustered it in}``."""
    out: dict[str, set[str]] = defaultdict(set)
    for champ, role in _CHAMP_ROLE_CLASS:
        out[champ].add(role)
    return out


def test_effective_class_applies_manual_overrides() -> None:
    """The two CLASS_FILE_GAPS champions resolve to their hand-assigned class."""
    assert effective_class("Skarner", "jng") == "Ult_Tanks"
    assert effective_class("Skarner", "top") == "Stat_Stick_Tanks"
    assert effective_class("Heimerdinger", "bot") == "Mage_Bots"
    assert effective_class("Heimerdinger", "mid") == "Pick_Mages"


def test_effective_class_leaves_unassigned_override_roles_unknown() -> None:
    """Roles omitted from an override stay Unknown: an override is not borrowed across roles."""
    assert effective_class("Skarner", "mid") == "Unknown"
    assert effective_class("Heimerdinger", "jng") == "Unknown"


def test_effective_class_direct_hit_matches_the_class_file() -> None:
    """A champion clustered directly in a role returns that file label."""
    (champ, role), label = next(iter(_CHAMP_ROLE_CLASS.items()))
    assert effective_class(champ, role) == label


def test_effective_class_borrows_a_flex_role_from_the_primary_role() -> None:
    """A role the class file never clustered borrows the champion's primary-role class."""
    top_label = effective_class("Aatrox", "top")
    assert top_label != "Unknown"
    assert "mid" not in _roles_of()["Aatrox"]  # genuinely a flex role
    assert effective_class("Aatrox", "mid") == top_label


def test_effective_class_resolves_every_clustered_champion_in_all_roles() -> None:
    """A champion clustered in any role gets a real class in every role."""
    clustered = {champ for champ, _ in _CHAMP_ROLE_CLASS if "_" not in champ}
    for champ in clustered:
        for role in _ROLES:
            assert effective_class(champ, role) != "Unknown", (champ, role)


def test_effective_class_is_unknown_for_an_unclustered_champion() -> None:
    """A champion machineloling never clustered, with no override, is Unknown."""
    assert effective_class("NotARealChampion", "mid") == "Unknown"


def test_multi_role_primary_matches_the_class_file() -> None:
    """``_MULTI_ROLE_PRIMARY`` covers exactly the 2+-role champions, each primary clustered."""
    roles_of = _roles_of()
    multi = {champ for champ, roles in roles_of.items() if len(roles) >= 2}

    assert set(_MULTI_ROLE_PRIMARY) == multi
    for champ, role in _MULTI_ROLE_PRIMARY.items():
        assert role in roles_of[champ], (champ, role)


def test_champion_roles_returns_every_clustered_role() -> None:
    """``champion_roles`` reports exactly the roles the class file clusters a champion in."""
    roles_of = _roles_of()

    assert champion_roles("Ashe") == frozenset({"bot"})  # single-role
    assert champion_roles("Aatrox") == roles_of["Aatrox"]  # flex: top + jng
    assert len(champion_roles("Aatrox")) >= 2


def test_champion_roles_is_empty_for_an_unclustered_champion() -> None:
    """An unclustered champion resolves to the empty set, so flex treats it as single-role."""
    assert champion_roles("NotARealChampion") == frozenset()
    assert champion_roles("Skarner") == frozenset()  # a CLASS_FILE_GAPS champion


def test_manual_override_labels_are_real_role_classes() -> None:
    """Each hand-assigned label is a class the class file already uses for that role."""
    role_labels: dict[str, set[str]] = defaultdict(set)
    for (_champ, role), label in _CHAMP_ROLE_CLASS.items():
        role_labels[role].add(label)

    for (champ, role), label in MANUAL_CLASS_OVERRIDES.items():
        assert label in role_labels[role], (champ, role, label)
