"""Torch-free production recipe for the skill filter: architecture + fit schedule."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping


class SkillFilterArchitecture(TypedDict):
    """The structural dimensions of the production filter."""

    d: int
    d_org: int
    hidden: int


@dataclass(frozen=True, slots=True)
class RatingRecipe:
    """The one frozen rating-model definition: structural dims plus the production fit schedule."""

    version: int
    d: int
    d_org: int
    hidden: int
    seeds: int
    lr: float
    patience: int
    margin_weight: float
    weight_decay: float
    max_epochs: int

    def architecture(self) -> SkillFilterArchitecture:
        """The structural dims as ``FilterConfig(**...)`` keyword arguments."""
        return {"d": self.d, "d_org": self.d_org, "hidden": self.hidden}

    def to_dict(self) -> dict[str, int | float | None]:
        """The exact dict persisted as the artifact payload's ``recipe`` entry."""
        return asdict(self)


RATING_RECIPE = RatingRecipe(
    version=2,
    d=2,
    d_org=4,
    hidden=32,
    seeds=5,
    lr=0.010,
    patience=15,
    margin_weight=0.3,
    weight_decay=0.01,
    max_epochs=150,
)

#: From this version a facts-game query reads the game's own five and context, the inputs the
#: replay trained on; an older or unstamped artifact answers every query from the last record.
GAME_INPUTS_VERSION = 2


def serves_game_inputs(recipe: Mapping | None) -> bool:
    """Whether an artifact's persisted recipe scores a facts game from that game's own inputs."""
    return recipe is not None and int(recipe["version"]) >= GAME_INPUTS_VERSION


#: Derived views of the one recipe — the latent-strength feature family's documented pin
#: (re-exported by ``pipeline/chosen_skill_filter.py``; must match the stored artifact config).
LATENT_STRENGTH_D = RATING_RECIPE.d
LATENT_STRENGTH_D_ORG = RATING_RECIPE.d_org
