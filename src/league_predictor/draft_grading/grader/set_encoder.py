"""The role-conditioned set encoder: one antisymmetric score plus the dense auxiliary heads."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import torch
from torch import nn

from league_predictor.draft_grading.champion_attributes.classes import CLASS_LABEL_TO_INDEX
from league_predictor.draft_grading.champion_attributes.classes import (
    ensure_loaded as _ensure_classes_loaded,
)
from league_predictor.draft_grading.champion_attributes.kit_vector import KIT_FAMILY_NAMES
from league_predictor.draft_grading.grader.slot_inputs import IDENTITY_COLUMN, SlotInputs
from league_predictor.vocab.roles import N_ROLES, N_SLOTS

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from league_predictor.draft_grading.grader.slot_inputs import BoundSlotInputs
    from league_predictor.draft_grading.recipe import SetTrainConfig

#: The five own-side tokens a perspective pools over; the other five are the enemy's.
N_SIDE = N_ROLES

#: Heads whose output is per slot, ``(N, 10, ...)``, against heads on the score difference.
_SLOT_HEADS = ("dpm", "damage_share", "class")

#: THE cross-team pairs, ``(offence kit family, the enemy's defence family)`` on team sums:
#: each carries ``blue_off * red_def - red_off * blue_def``, weightless and antisymmetric.
KIT_CROSS_PAIRS: tuple[tuple[str, str], ...] = (
    ("armor_pen_pct", "armor"),
    ("lethality", "armor"),
    ("armor_shred_pct", "armor"),
    ("physical_damage", "armor"),
    ("magic_pen_pct", "magic_resist"),
    ("magic_damage", "magic_resist"),
    ("pct_hp_damage", "health"),
    ("true_damage", "health"),
)


#: The pair columns an arm without the cross block carries: none.
_NO_COLUMNS = (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))


def _as_config(config: SetTrainConfig | Mapping[str, Any]) -> SetTrainConfig:
    """The recipe object, whether a caller holds one or the dict a family persisted."""
    from league_predictor.draft_grading.recipe import SetTrainConfig

    return config if isinstance(config, SetTrainConfig) else SetTrainConfig.from_dict(config)


def _cross_pair_columns(inputs: SlotInputs) -> tuple[torch.Tensor, torch.Tensor]:
    """`KIT_CROSS_PAIRS` as offence and defence column indices into the slot row.

    Raises:
        ValueError: a pair names a kit family the current vector does not carry.
    """
    start = inputs.block("kit").start
    unknown = sorted({n for pair in KIT_CROSS_PAIRS for n in pair} - set(KIT_FAMILY_NAMES))
    if unknown:
        raise ValueError(f"KIT_CROSS_PAIRS names kit families the vector lost: {unknown}")
    columns = [
        (start + KIT_FAMILY_NAMES.index(off), start + KIT_FAMILY_NAMES.index(defence))
        for off, defence in KIT_CROSS_PAIRS
    ]
    return (
        torch.tensor([o for o, _ in columns], dtype=torch.long),
        torch.tensor([d for _, d in columns], dtype=torch.long),
    )


class _SetLayer(nn.Module):
    """Pre-norm multi-head self-attention over the ten tokens, then a position-wise MLP."""

    def __init__(self, width: int, heads: int, ff_hidden: int):
        super().__init__()
        self.heads = heads
        self.head_dim = width // heads
        self.scale = self.head_dim**-0.5
        self.norm_attn = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)
        self.norm_ff = nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, ff_hidden), nn.GELU(), nn.Linear(ff_hidden, width))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(B, 10, W) -> (B, 10, W)``; every token attends over all ten, no mask."""
        batch, tokens, width = z.shape
        qkv = self.qkv(self.norm_attn(z))
        q, k, v = qkv.view(batch, tokens, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        weights = torch.softmax(q @ k.transpose(-2, -1) * self.scale, dim=-1)
        context = (weights @ v).transpose(1, 2).reshape(batch, tokens, width)
        z = z + self.out(context)
        return z + self.ff(self.norm_ff(z))


class _IdentityResidual(nn.Module):
    """``r_c`` added to a projected slot token: the phase-4 identity arm, zero at init."""

    def __init__(self, n_champions: int, width: int, dropout: float):
        super().__init__()
        self.dropout = float(dropout)
        self.residual = nn.Embedding(n_champions, width)
        nn.init.zeros_(self.residual.weight)
        #: Rows ``[0, n_known)`` are champions; the last row is UNKNOWN, pinned at zero so
        #: an out-of-universe pick reads its computed kit alone.
        self.n_known = n_champions - 1
        mask = torch.ones(n_champions, 1)
        mask[self.n_known] = 0.0
        self.register_buffer("_grad_mask", mask)
        self.residual.weight.register_hook(lambda g: g * cast("torch.Tensor", self._grad_mask))

    def forward(self, champion_ids: torch.Tensor) -> torch.Tensor:
        """``(B, 10) -> (B, 10, W)``; one dropout draw per champion per batch."""
        residual = self.residual(champion_ids)
        if self.training and self.dropout > 0.0:
            keep = torch.rand(len(self.residual.weight), device=residual.device)
            keep = (keep >= self.dropout).to(residual.dtype)
            residual = residual * keep[champion_ids].unsqueeze(-1)
        return residual

    def mean_sq_norm(self) -> torch.Tensor:
        """``mean_c ‖r_c‖²`` over known champions (UNKNOWN excluded — pinned zero)."""
        return self.residual.weight[: self.n_known].pow(2).sum(dim=1).mean()


class DraftSetEncoder(nn.Module):
    """``score = w · (h_blue - h_red)`` over a shared per-perspective encoder.

    Attributes:
        d_in: the slot row's width; `slot_inputs.SlotInputs.d_in` is its one derivation.
    """

    def __init__(self, config: SetTrainConfig | Mapping[str, Any], d_in: int, n_champions: int = 0):
        """Build the shared encoder, the named cross-team pairs and the configured heads.

        Args:
            config: a `SetTrainConfig`, or the dict a family persisted (the loader's form).
            n_champions: embedding rows for the identity arm, UNKNOWN included; the
                base arm carries no identity table and ignores it.

        Raises:
            ValueError: ``d_in`` is not the config's own input width, or the identity arm
                was built without a champion universe.
        """
        super().__init__()
        _ensure_classes_loaded()
        config = _as_config(config)
        inputs = SlotInputs(config)
        if d_in != inputs.width:
            raise ValueError(
                f"DraftSetEncoder(d_in={d_in}) against a config whose slot row is "
                f"{inputs.width} wide ({[block.name for block in inputs.spec]})."
            )
        self.config = config
        self.d_in = d_in
        self.d_feat = d_in - (1 if config.identity_residual else 0)
        self._id_column = inputs.block(IDENTITY_COLUMN).start if config.identity_residual else 0

        # The own-side flag is written in per perspective, so it is never an input column.
        self.in_proj = nn.Linear(self.d_feat + 1, config.width)
        self.identity = None
        if config.identity_residual:
            if n_champions < 2:  # noqa: PLR2004 - one champion plus the UNKNOWN row
                raise ValueError(
                    "DraftSetEncoder(identity_residual=True) needs n_champions, the "
                    "champion universe with its UNKNOWN row; the table is data, not recipe."
                )
            self.identity = _IdentityResidual(n_champions, config.width, config.identity_dropout)
        self.layers = nn.ModuleList(
            _SetLayer(config.width, config.heads, config.ff_mult * config.width)
            for _ in range(config.layers)
        )
        self.norm = nn.LayerNorm(config.width)
        self.cross_pairs = bool(config.cross_pairs)
        offence, defence = _cross_pair_columns(inputs) if self.cross_pairs else _NO_COLUMNS
        self.register_buffer("cross_offence", offence)
        self.register_buffer("cross_defence", defence)
        pooled_dim = config.width + (len(KIT_CROSS_PAIRS) if self.cross_pairs else 0)
        self.score = nn.Linear(pooled_dim, 1, bias=False)
        # Zero-init the score: an untrained arm reads every draft at exactly one half, and
        # the gradient through ``w · diff`` is the difference itself, so it still learns.
        nn.init.zeros_(self.score.weight)
        self.heads = nn.ModuleDict(
            {name: self._head(name, config.width, pooled_dim) for name in config.aux_heads}
        )
        side = torch.cat([torch.ones(N_SIDE), torch.zeros(N_SIDE)]).view(1, N_SLOTS, 1)
        self.register_buffer("side_flag", side)

    @staticmethod
    def _head(name: str, width: int, pooled_dim: int) -> nn.Module:
        """One auxiliary head: per slot off a token, or antisymmetric off the difference."""
        if name == "class":
            return nn.Linear(width, len(CLASS_LABEL_TO_INDEX))
        if name in _SLOT_HEADS:
            return nn.Linear(width, 1)
        return nn.Linear(pooled_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score ``(N, 10, D)`` slot rows.

        Returns:
            The antisymmetric score per draft and the configured heads' outputs (empty when
            none are on). Mirroring the ten rows negates the score bit for bit: the two
            perspectives are the same forward's two halves, so the swap permutes them.
        """
        n = x.shape[0]
        feats = x[..., : self.d_feat]
        own = torch.cat([feats[:, :N_SIDE], feats[:, N_SIDE:]], dim=0)
        enemy = torch.cat([feats[:, N_SIDE:], feats[:, :N_SIDE]], dim=0)
        side = cast("torch.Tensor", self.side_flag).expand(2 * n, N_SLOTS, 1)
        z = self.in_proj(torch.cat([torch.cat([own, enemy], dim=1), side], dim=-1))
        if self.identity is not None:
            z = z + self.identity(self._perspective_ids(x))
        for layer in self.layers:
            z = layer(z)
        z = self.norm(z)

        pooled = z[:, :N_SIDE].mean(dim=1)
        diff = pooled[:n] - pooled[n:]
        if self.cross_pairs:
            diff = torch.cat([diff, self._cross_terms(feats)], dim=-1)
        return self.score(diff).squeeze(-1), self._aux(z, diff, n)

    def _cross_terms(self, feats: torch.Tensor) -> torch.Tensor:
        """``(N, len(KIT_CROSS_PAIRS))``: each named pair's antisymmetric offence-defence term."""
        offence = cast("torch.Tensor", self.cross_offence)
        defence = cast("torch.Tensor", self.cross_defence)
        blue, red = feats[:, :N_SIDE].sum(dim=1), feats[:, N_SIDE:].sum(dim=1)
        return blue[:, offence] * red[:, defence] - red[:, offence] * blue[:, defence]

    def _perspective_ids(self, x: torch.Tensor) -> torch.Tensor:
        """The ten champion ids of each perspective, own five then enemy five."""
        ids = x[..., self._id_column].long()
        blue, red = ids[:, :N_SIDE], ids[:, N_SIDE:]
        return torch.cat([torch.cat([blue, red], dim=1), torch.cat([red, blue], dim=1)], dim=0)

    def _aux(self, z: torch.Tensor, diff: torch.Tensor, n: int) -> dict[str, torch.Tensor]:
        """Each configured head's output, slot heads re-laid as ``(N, 10, ...)`` blue then red."""
        out: dict[str, torch.Tensor] = {}
        for name, head in self.heads.items():
            if name not in _SLOT_HEADS:
                out[name] = head(diff).squeeze(-1)
                continue
            per_slot = head(z[:, :N_SIDE])
            if name != "class":
                per_slot = per_slot.squeeze(-1)
            out[name] = torch.cat([per_slot[:n], per_slot[n:]], dim=1)
        return out

    def identity_penalty(self) -> torch.Tensor:
        """``λ·mean_c‖r_c‖²``, the identity arm's residual prior; zero without the arm."""
        if self.identity is None:
            return torch.zeros((), device=self.score.weight.device)
        return self.config.identity_lambda * self.identity.mean_sq_norm()


def production_wiring() -> tuple[type[DraftSetEncoder], Callable[..., BoundSlotInputs]]:
    """``(encoder_factory, inputs_factory)``, so a loader names no module of this package."""
    return DraftSetEncoder, SlotInputs.build_for
