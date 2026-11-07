"""Tests for the display-name <-> canonical-name normalizer."""

from __future__ import annotations

import pytest

from league_predictor.draft_grading.champion_attributes.normalize import (
    FORM_DEFAULTS,
    resolve_form,
    to_canonical,
)


def test_identity_passthrough_for_single_word_names() -> None:
    for name in ("Aatrox", "Jinx", "Zed", "Yasuo", "Mel"):
        assert to_canonical(name) == name


@pytest.mark.parametrize(
    ("display", "canonical"),
    [
        ("Dr. Mundo", "DrMundo"),
        ("Xin Zhao", "XinZhao"),
        ("Wukong", "MonkeyKing"),
        ("Bel'Veth", "Belveth"),
        ("Kai'Sa", "Kaisa"),
        ("Nunu & Willump", "Nunu"),
        ("Renata Glasc", "Renata"),
        ("LeBlanc", "Leblanc"),
    ],
)
def test_known_display_to_canonical(display: str, canonical: str) -> None:
    assert to_canonical(display) == canonical


@pytest.mark.parametrize(
    ("base", "resolved"),
    list(FORM_DEFAULTS.items()),
)
def test_resolve_form_uses_dominant_form(base: str, resolved: str) -> None:
    assert resolve_form(base) == resolved


def test_resolve_form_falls_through_to_canonical() -> None:
    assert resolve_form("Dr. Mundo") == "DrMundo"
    assert resolve_form("Aatrox") == "Aatrox"
