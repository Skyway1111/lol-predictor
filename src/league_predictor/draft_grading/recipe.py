"""The frozen production draft recipe and evidence gate."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any

from league_predictor.draft_grading.grader.pro_excess import OUTCOME_OFFSETS

GATE_K = 32.0
AUX_HEAD_NAMES: tuple[str, ...] = ("dpm", "damage_share", "gd15", "long_game", "class")


@dataclass(frozen=True)
class SetTrainConfig:
    """Configure the one role-conditioned set encoder: its inputs, its heads, and its fit."""

    # The three computed input blocks; `grader/slot_inputs.py` names every column once and
    # leaves a denied block OUT of the row rather than zeroing it.
    kit: bool = True
    power: bool = True
    pro_form: bool = True
    aux_heads: tuple[str, ...] = AUX_HEAD_NAMES
    # Floor on every head's Kendall log-variance, so a head weighs at most ``exp(-floor)``:
    # -2.0 caps one near seven times the main head. At 0.0, 4b's smoke fit read five of six
    # heads pinned on the floor; that is the fit corpus, not the window.
    log_var_floor: float = -2.0
    # A per-champion residual on the projected slot token, penalised and dropped so the
    # computed input carries the champion. Both constants are set here, never tuned.
    identity_residual: bool = False
    identity_lambda: float = 1e-2
    identity_dropout: float = 0.5
    # The residual target the main head rides: "none", or "team_form", the walk-forward
    # team-strength logit difference (grader/pro_excess.py). Training-time only.
    outcome_offset: str = "team_form"
    outcome_since: str = "2017-01-01"
    seeds: int = 5
    width: int = 64
    heads: int = 4
    layers: int = 2
    ff_mult: int = 2
    # The named offence-against-defence pairs over the two teams' kit-family sums
    # (set_encoder.KIT_CROSS_PAIRS); they read the kit block, so they need it.
    cross_pairs: bool = True
    max_epochs: int = 12
    patience: int = 2
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-2
    oof_folds: int = 1
    device: str | None = None

    def __post_init__(self) -> None:
        """Refuse an input, head or optimizer setting no fit could carry, before any work."""
        if not (self.kit or self.power or self.pro_form):
            raise ValueError(
                "SetTrainConfig: every input block is denied — the slot row would be the "
                "role one-hot alone and the encoder would score every draft the same."
            )
        unknown = tuple(h for h in self.aux_heads if h not in AUX_HEAD_NAMES)
        if unknown or len(set(self.aux_heads)) != len(self.aux_heads):
            raise ValueError(
                f"SetTrainConfig.aux_heads={self.aux_heads}; known, each at most once: "
                f"{AUX_HEAD_NAMES}"
            )
        if self.outcome_offset not in OUTCOME_OFFSETS:
            raise ValueError(
                f"SetTrainConfig.outcome_offset={self.outcome_offset!r}; "
                f"known residual targets: {OUTCOME_OFFSETS}."
            )
        if not 0.0 <= self.identity_dropout <= 1.0 or self.identity_lambda < 0.0:
            raise ValueError(
                f"SetTrainConfig identity residual: dropout={self.identity_dropout!r} must be "
                f"a probability and lambda={self.identity_lambda!r} non-negative."
            )
        if self.cross_pairs and not self.kit:
            raise ValueError(
                "SetTrainConfig.cross_pairs names kit families on both sides, so it has no "
                "input under kit=False. Set cross_pairs=False."
            )
        if self.width % self.heads:
            raise ValueError(
                f"SetTrainConfig.width={self.width} must split over heads={self.heads}."
            )
        if min(self.seeds, self.layers, self.max_epochs, self.patience, self.batch_size) < 1:
            raise ValueError(
                "SetTrainConfig: seeds, layers, max_epochs, patience and batch_size are counts."
            )
        if self.lr <= 0.0 or self.weight_decay < 0.0 or self.oof_folds < 1:
            raise ValueError("SetTrainConfig: lr > 0, weight_decay >= 0, oof_folds >= 1.")

    def to_dict(self) -> dict[str, Any]:
        """The model-meaning fields, -persisted; ``device`` is a resource knob, not recipe."""
        values = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "device"}
        return {k: list(v) if isinstance(v, tuple) else v for k, v in values.items()}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SetTrainConfig:
        """Rebuild a persisted config; JSON turns the head tuple into a list and back."""
        known = {f.name for f in fields(cls)}
        values = {k: v for k, v in payload.items() if k in known}
        if "aux_heads" in values:
            values["aux_heads"] = tuple(values["aux_heads"])
        return cls(**values)


@dataclass(frozen=True)
class DraftRecipe:
    """The persisted set-family definition."""

    version: int = 7
    kind: str = "set"
    set_side: SetTrainConfig = field(default_factory=SetTrainConfig)

    def __post_init__(self):
        if self.kind != "set":
            raise ValueError(f"unsupported draft recipe kind: {self.kind!r}")

    def set_recipe(self) -> SetTrainConfig:
        return self.set_side

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": int(self.version),
            "kind": self.kind,
            "set_side": self.set_side.to_dict(),
        }


DRAFT_RECIPE = DraftRecipe()
