"""The chosen rating is pinned by the rating.pt slot file and structurally correct."""

from __future__ import annotations

import pytest

from league_predictor.predraft.pipeline.chosen_skill_filter import (
    SKILL_FILTER_ARTIFACT,
    chosen_skill_filter_backend,
)
from league_predictor.predraft.ratings.skill_filter import BaggedSkillFilterBackend
from league_predictor.predraft.ratings.skill_filter.artifact import load_bag_artifact
from league_predictor.predraft.ratings.skill_filter.recipe import (
    LATENT_STRENGTH_D,
    LATENT_STRENGTH_D_ORG,
)

# --- The chosen rating: the slot-file-pinned skill-filter seed-bag ---------------------


def test_chosen_skill_filter_backend_is_the_prd_slot_bag():
    backend = chosen_skill_filter_backend()
    assert isinstance(backend, BaggedSkillFilterBackend)
    assert backend.artifact_path == SKILL_FILTER_ARTIFACT
    # The artifact file is the pin -- no code-frozen sha is verified at load.
    assert backend.expected_sha is None
    # Fresh stateless instances each call.
    assert chosen_skill_filter_backend() is not chosen_skill_filter_backend()


@pytest.mark.slow
def test_chosen_skill_filter_shape_matches_the_artifact_config():
    # The frozen (d, d_org) fix the skill-filter column set; the bag is otherwise
    # config-diverse, so every member must share those dims or feature names drift.
    models, meta = load_bag_artifact(SKILL_FILTER_ARTIFACT)
    assert len(models) == 5  # the locked full-history refresh bag
    assert all(m.config.d == LATENT_STRENGTH_D for m in models)
    assert all(m.config.d_org == LATENT_STRENGTH_D_ORG for m in models)
    assert meta["sha"]  # the meta carries the on-disk byte sha
