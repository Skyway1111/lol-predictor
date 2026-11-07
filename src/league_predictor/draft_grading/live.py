"""Live draft-grader loading and partial champion-only scoring."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from league_predictor.draft_grading.grader.predictor import (
    DraftScore,
    SwapRequest,
)
from league_predictor.draft_grading.set_variant import SetDraftPredictor
from league_predictor.draft_grading.state import (
    ROLE_ORDER,
    DraftState,
    PlayerSlot,
    TeamComp,
)
from league_predictor.draft_grading.state import (
    UNPICKED as UNPICKED,  # noqa: PLC0414 - PEP 484 explicit re-export
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping
    from datetime import datetime

    import numpy as np
    import torch

    from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
    from league_predictor.draft_grading.grader.swaps import SupportsKnownChampions


class DraftGrader(Protocol):
    """The prediction and swap reads shared by scoring consumers."""

    artifact_stem: str

    @property
    def index(self) -> SupportsKnownChampions: ...

    @property
    def pro_excess_baseline(self) -> str: ...

    @property
    def thresholds(self) -> tuple[float, float, float, float]: ...

    def predict(self, draft: DraftState, stats_cache: BulkStatsCache | None = None) -> DraftScore:
        """Score one draft."""
        ...

    def predict_batch(
        self, drafts: list[DraftState], stats_cache: BulkStatsCache | None = None
    ) -> list[DraftScore]:
        """Score N drafts, row-aligned."""
        ...

    def predict_swaps(
        self,
        draft: DraftState,
        side: str,
        role: str,
        alternatives: list[str],
        *,
        stats_cache: BulkStatsCache | None = None,
    ) -> np.ndarray:
        """Rank one slot's alternatives against the known enemy."""
        ...

    def predict_swaps_marginalized(
        self,
        draft: DraftState,
        side: str,
        role: str,
        alternatives: list[str],
        *,
        enemy_role: str,
        enemy_responses: list[tuple[str, float]],
        stats_cache: BulkStatsCache | None = None,
    ) -> np.ndarray:
        """The same ranking, marginalized over an enemy-response distribution."""
        ...

    def predict_swaps_marginalized_batch(
        self,
        draft: DraftState,
        requests: list[SwapRequest],
        stats_cache: BulkStatsCache | None = None,
    ) -> list[np.ndarray]:
        """Many requests over one draft."""
        ...

    def predict_swaps_marginalized_many(
        self,
        batches: list[tuple[DraftState, list[SwapRequest]]],
        stats_cache: BulkStatsCache | None = None,
    ) -> list[list[np.ndarray]]:
        """Many drafts, each with its own requests."""
        ...


def load_grader(
    conn: sqlite3.Connection,
    *,
    artifact: Path,
    device: torch.device | str | None = None,
) -> SetDraftPredictor:
    """Load the format-4 production family with its frozen knowledge."""
    return SetDraftPredictor.load(Path(artifact), conn, device=device)


def build_hypothetical_draft(
    *,
    blue: dict[str, str],
    red: dict[str, str],
    blue_bans: list[str],
    red_bans: list[str],
    when: datetime,
) -> DraftState:
    """Build a champion-only draft, neutrally filling every unpicked role."""

    def comp(team_id: str, by_role: Mapping[str, str], bans: list[str]) -> TeamComp:
        return TeamComp(
            team_id=team_id,
            slots=tuple(
                PlayerSlot(
                    role=role,
                    champion_id=by_role.get(role, UNPICKED),
                    player_id=f"sandbox:{team_id}:{role}",
                )
                for role in ROLE_ORDER
            ),
            bans=tuple(bans) or None,
        )

    return DraftState(
        match_id="sandbox",
        date=when,
        patch="",
        region="LCK",
        blue=comp("sandbox_blue", blue, blue_bans),
        red=comp("sandbox_red", red, red_bans),
    )
