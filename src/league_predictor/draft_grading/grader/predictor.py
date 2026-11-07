"""Draft scores and swap requests shared by serving and scoring."""

from __future__ import annotations

from dataclasses import dataclass

from league_predictor.draft_grading.grader.grades import Grade

FORWARD_ROWS = 512


@dataclass(frozen=True)
class DraftScore:
    """Result of scoring one game's draft.

    Attributes:
        edge_pp_blue: blue's calibrated expected edge in percentage points — red's is
            exactly its negation, never stored twice. ``None`` without a frozen curve.
    """

    prob_blue_wins: float
    grade_blue: Grade
    grade_red: Grade
    edge_pp_blue: float | None = None


@dataclass(frozen=True)
class SwapRequest:
    """One marginalized-swap query over a shared draft (a batch element).

    Attributes:
        side: focal slot's side.
        role: focal slot's role.
        alternatives: champions to substitute into the focal slot.
        enemy_role: must equal ``role``.
        enemy_responses: ``(champion, weight)`` distribution the score marginalizes over;
            a single entry is the known-enemy counter-pick query, several the blind pick.
    """

    side: str
    role: str
    alternatives: list[str]
    enemy_role: str
    enemy_responses: list[tuple[str, float]]
