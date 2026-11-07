"""The ``(N, 10, D)`` computed slot input the set encoder eats: one named block per source."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from league_predictor.draft_grading.champion_attributes.kit_vector import KIT_FAMILY_NAMES
from league_predictor.draft_grading.champion_attributes.power_store import BREAKPOINTS, CHANNELS
from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from league_predictor.vocab.roles import N_SLOTS, ROLE_ORDER, SLOT_ROLES

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Mapping, Sequence

    from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
    from league_predictor.draft_grading.grader.knowledge_tables import KnowledgeTables
    from league_predictor.draft_grading.recipe import SetTrainConfig
    from league_predictor.draft_grading.state import DraftState

#: The block carrying the walk-forward stats, and the one carrying a champion id rather
#: than a feature: an EMBEDDING ROW, so a reader that DRAWS a row draws a legal id there.
_PRO_FORM = "pro_form"
IDENTITY_COLUMN = "champion_id"


def _power_columns() -> tuple[str, ...]:
    """The slot's own-role power curve, breakpoint-major and channel-minor."""
    return tuple(f"power_{channel}_k{k}" for k in BREAKPOINTS for channel in CHANNELS)


#: Every block this module can build, in row order, with the columns it contributes.
_BLOCK_COLUMNS: dict[str, tuple[str, ...]] = {
    "role": tuple(f"role_{role}" for role in ROLE_ORDER),
    "kit": tuple(f"kit_{name}" for name in KIT_FAMILY_NAMES),
    "power": _power_columns(),
    _PRO_FORM: ("pro_excess", "pro_pr"),
    IDENTITY_COLUMN: (IDENTITY_COLUMN,),
}


def design_columns(input_spec: Mapping[str, Any]) -> list[str]:
    """A frozen layout's ``(N, 10, D)`` rows flattened: each slot's columns, blue then red."""
    return [
        f"{column}_{side}_{role}"
        for side in ("blue", "red")
        for role in ROLE_ORDER
        for column in input_spec["columns"]
    ]


def design_blocks(input_spec: Mapping[str, Any], d_in: int) -> dict[str, np.ndarray]:
    """Each named block's columns in that flattened design: its own span, over ten slots."""
    slots = np.arange(N_SLOTS)[:, None] * d_in
    return {
        str(block["name"]): (slots + np.arange(block["start"], block["stop"])).ravel()
        for block in input_spec["blocks"]
    }


@dataclass(frozen=True)
class SlotBlock:
    """One named column block of a slot row, and the columns it contributes in order."""

    name: str
    columns: tuple[str, ...]


class SlotInputs:
    """Build the set encoder's slot rows from the frozen knowledge and the walk-forward stats.

    Attributes:
        spec: the row's blocks in column order; a denied block is ABSENT, never zeroed, so
            an arm's input width names what it reads.
        width: ``D``, the row's column count.
    """

    def __init__(self, config: SetTrainConfig):
        wanted = ["role"]
        wanted += ["kit"] if config.kit else []
        wanted += ["power"] if config.power else []
        wanted += [_PRO_FORM] if config.pro_form else []
        # The champion's index id, not a feature: the encoder slices it off and embeds it.
        # It rides the row's tail so the feature columns keep their arm's layout.
        wanted += [IDENTITY_COLUMN] if config.identity_residual else []
        self._build_from(tuple(SlotBlock(name, _BLOCK_COLUMNS[name]) for name in wanted))

    def _build_from(self, blocks: tuple[SlotBlock, ...]) -> None:
        """Fill the layout view both constructors share."""
        self.spec = blocks
        self.width = sum(len(block.columns) for block in blocks)
        start = 0
        self._slices: dict[str, slice] = {}
        for block in blocks:
            self._slices[block.name] = slice(start, start + len(block.columns))
            start += len(block.columns)
        #: One-entry cache of what a knowledge member implies: its index and its z-scored
        #: power table, keyed by the member itself.
        self._bound: tuple[KnowledgeTables, ChampionIndex, np.ndarray | None] | None = None

    @classmethod
    def from_spec(cls, input_spec: Mapping[str, Any]) -> SlotInputs:
        """Rebuild the FROZEN layout a fitted family persisted, verbatim.

        Raises:
            ValueError: the spec names a block this module cannot build, or its blocks do
                not partition its columns; a serving build must not guess a layout.
        """
        columns = [str(name) for name in input_spec["columns"]]
        blocks: list[SlotBlock] = []
        start = 0
        for entry in input_spec["blocks"]:
            name, stop = str(entry["name"]), int(entry["stop"])
            if name not in _BLOCK_COLUMNS or int(entry["start"]) != start or stop <= start:
                raise ValueError(
                    f"slot input spec: block {name!r} at [{entry['start']}, {stop}) does not "
                    f"continue the row at {start}, or names no block this code builds "
                    f"({sorted(_BLOCK_COLUMNS)})."
                )
            blocks.append(SlotBlock(name, tuple(columns[start:stop])))
            start = stop
        if start != len(columns):
            raise ValueError(f"slot input spec: its blocks span {start} of {len(columns)} columns.")
        inputs = cls.__new__(cls)
        inputs._build_from(tuple(blocks))
        return inputs

    def to_spec(self) -> dict[str, Any]:
        """The persisted layout: every column named once, and the blocks that partition them."""
        return {
            "columns": list(self.columns()),
            "blocks": [
                {
                    "name": block.name,
                    "start": self._slices[block.name].start,
                    "stop": self._slices[block.name].stop,
                }
                for block in self.spec
            ],
        }

    @staticmethod
    def d_in(config: SetTrainConfig) -> int:
        """``D`` for one config — the width `DraftSetEncoder` is built against."""
        return SlotInputs(config).width

    @staticmethod
    def build_for(
        input_spec: Mapping[str, Any],
        knowledge: KnowledgeTables,
        db: sqlite3.Connection | None = None,
    ) -> BoundSlotInputs:
        """Bind a frozen layout, its knowledge and its stats source: call the result with drafts."""
        return BoundSlotInputs(SlotInputs.from_spec(input_spec), knowledge, db)

    def columns(self) -> tuple[str, ...]:
        """Every column's name once, in row order."""
        return tuple(name for block in self.spec for name in block.columns)

    def has(self, name: str) -> bool:
        """Whether this arm's row carries the named block."""
        return name in self._slices

    def block(self, name: str) -> slice:
        """The column slice one named block occupies; raises when this arm denies it."""
        if name not in self._slices:
            raise KeyError(f"{name!r} is not a block of this input: {[b.name for b in self.spec]}")
        return self._slices[name]

    @property
    def needs_stats(self) -> bool:
        """Whether a build needs a `BulkStatsCache`; only the pro-form block reads one."""
        return self.has(_PRO_FORM)

    def build(
        self,
        drafts: Sequence[DraftState],
        knowledge: KnowledgeTables,
        stats_cache: BulkStatsCache | None = None,
    ) -> torch.Tensor:
        """The ``(N, 10, D)`` rows for ``drafts``, each read as of its own game's date.

        Args:
            stats_cache: the walk-forward pro-form source; required by the pro_form block.

        Returns:
            float32 rows, blue in ``ROLE_ORDER`` then red. A mirrored draft's rows are this
            tensor under `vocab.roles.MIRROR_SLOTS`, bit for bit, every column included.

        Raises:
            RuntimeError: the knowledge member carries no table a configured block needs,
                or the pro-form block has no stats cache.
        """
        drafts = list(drafts)
        index, power = self._bind(knowledge)
        ids = np.array([index.encode_draft(draft) for draft in drafts], dtype=np.int64).reshape(
            len(drafts), N_SLOTS
        )
        out = np.zeros((len(drafts), N_SLOTS, self.width), dtype=np.float32)
        out[:, np.arange(N_SLOTS), self.block("role").start + SLOT_ROLES] = 1.0
        if self.has("kit"):
            out[..., self.block("kit")] = self._table(knowledge.kit, "kit")[ids]
        if power is not None:
            out[..., self.block("power")] = power[ids, SLOT_ROLES].reshape(len(drafts), N_SLOTS, -1)
        if self.needs_stats:
            self._fill_pro_form(out, drafts, stats_cache)
        if self.has(IDENTITY_COLUMN):
            out[..., self.block(IDENTITY_COLUMN)] = ids[:, :, None]
        return torch.from_numpy(out)

    def _bind(self, knowledge: KnowledgeTables) -> tuple[ChampionIndex, np.ndarray | None]:
        """The champion index and z-scored power table one member implies, built once."""
        if self._bound is not None and self._bound[0] is knowledge:
            return self._bound[1], self._bound[2]
        index = ChampionIndex.from_names(list(knowledge.champions))
        power = None
        if self.has("power"):
            power = _standardised_power(self._table(knowledge.power, "power").table)
        self._bound = (knowledge, index, power)
        return index, power

    def _table(self, table, name: str):
        """One configured knowledge table, or a named refusal.

        Raises:
            RuntimeError: the family's frozen member carries no such table.
        """
        if table is None:
            raise RuntimeError(
                f"SetTrainConfig.{name}=True but the knowledge member carries no {name} "
                f"table — the family was frozen without it, so the {name} block cannot be "
                "built. Refit under this config, or deny the block."
            )
        return table

    def _fill_pro_form(
        self, out: np.ndarray, drafts: Iterable[DraftState], stats_cache: BulkStatsCache | None
    ) -> None:
        """Write each slot's ``(excess, pick_rate)`` as of its own game's date.

        Raises:
            RuntimeError: no stats cache was handed in, so the pair has no source.
        """
        if stats_cache is None:
            raise RuntimeError(
                "the pro_form block needs a BulkStatsCache — the pair is the walk-forward "
                "(champion, role) excess and pick rate as of the game's date."
            )
        columns = self.block(_PRO_FORM)
        for row, draft in enumerate(drafts):
            slots = [slot for comp in (draft.blue, draft.red) for slot in comp.slots]
            for position, slot in enumerate(slots):
                excess, pick_rate, _ = stats_cache.pro_stats(
                    slot.champion_id, slot.role, draft.date
                )
                out[row, position, columns] = (excess, pick_rate)


class BoundSlotInputs:
    """A frozen layout with its knowledge and stats source bound; call it with drafts.

    Attributes:
        inputs: the layout the family persisted, rebuilt by `SlotInputs.from_spec`.
    """

    def __init__(
        self,
        inputs: SlotInputs,
        knowledge: KnowledgeTables,
        db: sqlite3.Connection | None = None,
    ):
        self.inputs = inputs
        self._knowledge = knowledge
        self._db = db
        self._cache: BulkStatsCache | None = None

    def __call__(
        self, drafts: Sequence[DraftState], stats_cache: BulkStatsCache | None = None
    ) -> torch.Tensor:
        """The ``(N, 10, D)`` rows; the bound connection builds a stats cache when none is given."""
        if stats_cache is None and self.inputs.needs_stats:
            stats_cache = self._stats()
        return self.inputs.build(drafts, self._knowledge, stats_cache)

    def _stats(self) -> BulkStatsCache:
        """The bound connection's stats cache, built once.

        Raises:
            RuntimeError: the binder holds no connection and the caller passed no cache.
        """
        if self._cache is None:
            if self._db is None:
                raise RuntimeError(
                    "this slot-input binder holds no connection, so it cannot build the "
                    "pro-form block's BulkStatsCache; pass one to the call, or bind a db."
                )
            from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache

            self._cache = BulkStatsCache(self._db)
        return self._cache


def _standardised_power(table: np.ndarray) -> np.ndarray:
    """Per-channel z-scores of a frozen power table; a cell the grid never ran stays zero.

    Returns:
        ``table``'s shape, standardised off the FROZEN member's own statistics, so a serving
        build and the fit that wrote it agree exactly and a zero row reads the corpus mean.
    """
    run = table.any(axis=(2, 3))
    cells = table[run]
    mu, sd = cells.mean(axis=(0, 1)), cells.std(axis=(0, 1))
    out = np.zeros_like(table)
    out[run] = (cells - mu) / np.where(sd > 0.0, sd, 1.0)
    return out.astype(np.float32)
