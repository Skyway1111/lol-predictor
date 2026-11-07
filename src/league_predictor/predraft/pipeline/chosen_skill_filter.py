"""The locked production skill-filter seed bag, pinned by ``models/rating/rating.pt``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from league_predictor import artifacts
from league_predictor.predraft.ratings.skill_filter import BaggedSkillFilterBackend
from league_predictor.predraft.ratings.skill_filter.artifact import rating_dependency

if TYPE_CHECKING:
    from league_predictor.predraft.ratings.skill_filter.artifact import RatingDependency

# --- The chosen skill filter ------------------------------------------------------------

# Full-history production bag; per-member configs ride the artifact's ``member_configs``.
# ``LATENT_STRENGTH_D``/``_ORG`` must match its stored config (a test asserts both).
SKILL_FILTER_ARTIFACT = artifacts.primary_path("rating")


def chosen_skill_filter_backend() -> BaggedSkillFilterBackend:
    """The production skill-filter backend; a stale index snapshot falls back to replay."""
    return BaggedSkillFilterBackend(SKILL_FILTER_ARTIFACT, use_index_snapshot=True)


def chosen_rating_dependency() -> RatingDependency:
    """Return the exact SHA, K, and ordered-member identity of the live rating artifact."""
    return rating_dependency(SKILL_FILTER_ARTIFACT)
