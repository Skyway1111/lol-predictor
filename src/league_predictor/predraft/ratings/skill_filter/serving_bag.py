"""The seed-bag index: member win-probabilities averaged, never member latents."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date
from typing import TYPE_CHECKING

from league_predictor.probability import logit

if TYPE_CHECKING:
    from league_predictor.asof import AsOf
    from league_predictor.predraft.ratings.skill_filter.artifact import RatingDependency
    from league_predictor.predraft.ratings.skill_filter.serving_index import (
        Query,
        SkillFilterIndex,
    )


def _sigmoid(z: float) -> float:
    """Logistic with a ±700 input clamp; local so this module stays torch-free at import."""
    return 1.0 / (1.0 + math.exp(-max(-700.0, min(700.0, z))))


class BaggedSkillFilterIndex:
    """A K-member seed-bag whose ``raw_score`` averages member WIN-PROBABILITIES, not latents.

    Attributes:
        members: independently-seeded ``SkillFilterIndex`` replays. Scoring is
            ``logit(mean_k sigmoid(member_k logit))`` — the proper-score bag (Jensen), still
            antisymmetric, holding the same contract as one ``SkillFilterIndex``. Latent
            spaces are not aligned across seeds, so strengths are NEVER averaged.
    """

    def __init__(self, members: Sequence[SkillFilterIndex]) -> None:
        if not members:
            raise ValueError("a seed-bag needs at least one member")
        self.members = list(members)
        self.source_signature: object | None = None
        self.through: date | None = None
        self.rating_dependency: RatingDependency | None = None

    @property
    def game_inputs(self) -> bool:
        """Whether every member reads a facts game's own inputs."""
        return all(member.game_inputs for member in self.members)

    # sightline-ok: 11 - the one-query form of each class's OWN batch; the index protocol's shape
    def raw_score(self, team_a: str, team_b: str, as_of: AsOf | date) -> float:
        """The bagged matchup logit -- the single-query form of ``raw_score_batch``."""
        return self.raw_score_batch([(team_a, team_b, as_of)])[0]

    # sightline-ok: 11 - the one-query form of each class's OWN batch; the index protocol's shape
    def derived_features(self, team_a: str, team_b: str, as_of: AsOf | date) -> dict[str, float]:
        """The one-query ``derived_features_batch``: member 0's family, bagged logit diff."""
        return self.derived_features_batch([(team_a, team_b, as_of)])[0]

    def raw_score_batch(self, queries: Sequence[Query]) -> list[float]:
        """The bagged matchup logits: member win logits averaged in PROBABILITY space.

        Returns:
            ``logit(mean_k sigmoid(member_k logit))`` per query — the proper-score bag whose
            expected log-loss cannot exceed the members' average (Jensen).
        """
        if not queries:
            return []
        member_logits = [member.raw_score_batch(queries) for member in self.members]
        n = len(self.members)
        return [  # one tuple of member logits per query
            logit(sum(_sigmoid(z) for z in col) / n, eps=1e-12)
            for col in zip(*member_logits, strict=True)
        ]

    def derived_features_batch(self, queries: Sequence[Query]) -> list[dict[str, float]]:
        """The bagged family: member 0's batched family with ``rating_logit_diff`` overridden.

        Returns:
            Per-seed latents/org/spread from one representative member rather than an
            average — they are not aligned across seeds.
        """
        if not queries:
            return []
        rows = self.members[0].derived_features_batch(queries)
        for row, bagged_logit in zip(rows, self.raw_score_batch(queries), strict=True):
            row["rating_logit_diff"] = bagged_logit
        return rows
