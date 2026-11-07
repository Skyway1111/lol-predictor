"""Counterfactual draft-pick helpers: legal alternatives, and one-champion swaps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from league_predictor.draft_grading.state import DraftState
from league_predictor.vocab.roles import ROLE_ORDER


# sightline-ok: 37 - two implementations: grader.embeddings.ChampionIndex and blend._BlendIndex
class SupportsKnownChampions(Protocol):
    """Any serving index exposing the champion universe."""

    @property
    def known_champions(self) -> Sequence[str]: ...


def legal_alternatives(
    draft: DraftState,
    champion_index: SupportsKnownChampions,
) -> list[str]:
    """Every champion not already picked or banned; role-appropriateness is the caller's."""
    unavailable: set[str] = set()
    for comp in (draft.blue, draft.red):
        unavailable.update(slot.champion_id for slot in comp.slots)
        if comp.bans:
            unavailable.update(b for b in comp.bans if b)

    return [name for name in champion_index.known_champions if name not in unavailable]


def validate_swap_query(side: str, role: str, enemy_role: str) -> None:
    """Raise on a malformed swap query; the rule every family's swap surfaces apply.

    Raises:
        ValueError: an unknown role or side, or ``enemy_role != role``, the only shape the
            marginalized engines support.
    """
    if role not in ROLE_ORDER:
        raise ValueError(f"unknown role: {role!r}")
    if enemy_role != role:
        raise ValueError(
            f"only enemy_role == role is supported; got role={role!r}, enemy_role={enemy_role!r}"
        )
    if side not in ("blue", "red"):
        raise ValueError(f"side must be 'blue' or 'red', got {side!r}")


def swap_champion(draft: DraftState, side: str, role: str, new_champion: str) -> DraftState:
    """Create a new DraftState with one champion swapped."""

    validate_swap_query(side, role, role)
    comp = getattr(draft, side)
    slots = tuple(
        replace(slot, champion_id=new_champion) if slot.role == role else slot
        for slot in comp.slots
    )
    return replace(draft, **{side: replace(comp, slots=slots)})
