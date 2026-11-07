"""Format-4 scaffolding: the stub encoder and slot inputs, a stub family, one on disk."""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
import torch

from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from league_predictor.draft_grading.grader.grades import EDGE_CURVE_KNOTS, EdgeCurve
from league_predictor.draft_grading.grader.knowledge_tables import (
    KNOWLEDGE_SCHEMA_VERSION,
    KnowledgeTables,
)
from league_predictor.draft_grading.live import build_hypothetical_draft
from league_predictor.draft_grading.set_variant import SetDraftPredictor
from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER
from tests.draft_grading.conftest import ALL_CHAMPS, BLUE_PICKS, RED_PICKS

if TYPE_CHECKING:
    from pathlib import Path

    from league_predictor.draft_grading.state import DraftState

WHEN = datetime(2026, 7, 1)
THRESHOLDS = (0.40, 0.45, 0.55, 0.60)
EDGE_CUTS_PP = (8.0, 4.0)
#: A stamp in the shape a v7 fit writes; only ``k`` reaches the serving path.
RECIPE = {"version": 7, "kind": "set", "k": 32.0}


class StubSetEncoder(torch.nn.Module):
    """4b's contract in miniature: one shared row map, pooled per side, then the difference."""

    def __init__(self, config, d_in: int, n_champions: int = 0):
        super().__init__()
        self.n_champions = int(n_champions)  # the factory's third argument, unread here
        self.rows = torch.nn.Linear(int(d_in), int(config["width"]), bias=False)
        self.head = torch.nn.Linear(int(config["width"]), 1, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``(score, aux)`` for ``(N, 10, D)`` rows; the half-swap negates the score exactly."""
        hidden = torch.tanh(self.rows(x))
        sides = hidden[:, :N_ROLES].sum(dim=1) - hidden[:, N_ROLES:].sum(dim=1)
        return self.head(sides).squeeze(-1), {"row_hidden": hidden}


class BiasedSetEncoder(StubSetEncoder):
    """The same encoder with an intercept: antisymmetric in nothing, refused at construction."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        score, aux = super().forward(x)
        return score + 0.25, aux


class DropoutSetEncoder(StubSetEncoder):
    """The same encoder with training-mode noise: a fit hands the predictor one of these."""

    def __init__(self, config, d_in: int, n_champions: int = 0):
        super().__init__(config, d_in, n_champions)
        self.drop = torch.nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return super().forward(self.drop(x))


class StubInputs:
    """A slot-input builder in 4b's shape: a row per (champion, role), no cross-board term."""

    COLUMNS = ("kit_0", "kit_1", "kit_2", "role_index")

    def __init__(self, index: ChampionIndex):
        self._index = index
        self._rows = np.random.default_rng(7).normal(size=(len(index), 3)).astype(np.float32)

    def __call__(self, drafts, stats_cache=None) -> torch.Tensor:  # noqa: ARG002 - the builder contract binds stats_cache; the stub reads no stats
        out = torch.zeros((len(drafts), 2 * N_ROLES, len(self.COLUMNS)), dtype=torch.float32)
        for i, draft in enumerate(drafts):
            for half, comp in enumerate((draft.blue, draft.red)):
                for r, slot in enumerate(comp.slots):
                    row = half * N_ROLES + r
                    cell = self._rows[self._index.encode(slot.champion_id)]
                    out[i, row, :3] = torch.from_numpy(cell)
                    out[i, row, 3] = float(r)
        return out


def cell_digest(champion: str, role: str) -> float:
    """A stable number in [0, 1) per cell; `hash()` is salted per process, so not that."""
    packed = hashlib.blake2s(f"{champion}/{role}".encode(), digest_size=2).digest()
    return int.from_bytes(packed, "big") / 2**16


class StubStats:
    """A ``BulkStatsCache`` stand-in: ``pro_stats`` third element is the decayed evidence.

    Attributes:
        champions: the facts' spellings, holding `ALIASED` where the index holds `Ksante`,
            so a reader that skips `blend.facts_spelling` reads that cell as cold.
        vary: read the excess and pick rate off the cell, so a swap moves the slot row's
            pro-form pair; off, both are 0.0 and the evidence count alone carries.
    """

    #: The one cell the facts know nothing about: a gather's cold row, gate exactly 0.
    COLD_CELL = ("Thresh", "sup")
    #: The facts' spelling of an index name, the TRAPS alias case (`Ksante` in the index).
    ALIASED = "K'Sante"

    def __init__(self, n_eff: float = 40.0, *, vary: bool = False):
        self._n_eff = n_eff
        self.vary = vary
        self.champions = frozenset({*ALL_CHAMPS, self.ALIASED} - {"Ksante"})

    def pro_stats(self, champion, role, as_of):  # noqa: ARG002 - mirrors BulkStatsCache.pro_stats
        digest = cell_digest(champion, role)
        # Per cell, so a gather that transposes its axes or shifts a half board reads
        # different numbers than the reference walk; a spelling the facts lack reads cold.
        known = champion in self.champions and (champion, role) != self.COLD_CELL
        n_eff = self._n_eff * (0.25 + digest) if known else 0.0
        return (digest - 0.5, digest / 2.0, n_eff) if self.vary else (0.0, 0.0, n_eff)

    def n_eff(self, champion, role, as_of):
        return float(self.pro_stats(champion, role, as_of)[2])


def stub_index() -> ChampionIndex:
    return ChampionIndex.from_names(ALL_CHAMPS)


def stub_knowledge(index: ChampionIndex) -> KnowledgeTables:
    """The smallest member the family contract accepts: z rows and a zero UNKNOWN row."""
    return KnowledgeTables(
        schema_version=KNOWLEDGE_SCHEMA_VERSION,
        feature_names=("kit_0", "kit_1"),
        champions=tuple(index.known_champions),
        z_matrix=np.zeros((len(index), 2), dtype=np.float32),
        power=None,
        kit=None,
    )


def stub_weights(*, seeds: int = 3, d_in: int = 4, width: int = 3) -> dict:
    """A ``SetWeights.to_state()`` payload over ``StubSetEncoder``, one state per seed."""
    config = {"width": width}
    states = []
    for seed in range(seeds):
        torch.manual_seed(seed)
        states.append(StubSetEncoder(config, d_in).state_dict())
    return {
        "config": config,
        "d_in": d_in,
        "n_champions": 0,
        "states": states,
        "input_spec": {
            "columns": list(StubInputs.COLUMNS),
            "blocks": [
                {"name": "kit", "start": 0, "stop": 3},
                {"name": "role", "start": 3, "stop": 4},
            ],
        },
        "history": {"epochs": [seeds]},
        "label_scale": {},
    }


def stub_wiring(encoder_cls=StubSetEncoder):
    """The ``(encoder_factory, inputs_factory)`` pair a stub family loads through."""

    def inputs_factory(input_spec, knowledge, db):  # noqa: ARG001 - the wiring contract binds all three; the stub reads the index only
        return StubInputs(ChampionIndex.from_names(knowledge.champions))

    return encoder_cls, inputs_factory


def set_source_recipe(monkeypatch, recipe) -> None:
    """Move the ONE source recipe for a test, the way a campaign arm moves it in source."""
    from league_predictor.draft_grading import recipe as recipe_module

    monkeypatch.setattr(recipe_module, "DRAFT_RECIPE", recipe)


def patch_wiring(monkeypatch, encoder_cls=StubSetEncoder) -> None:
    """Point the loader's production wiring at the stub pair, at its ONE resolved home."""
    from league_predictor.draft_grading.grader import set_encoder

    monkeypatch.setattr(set_encoder, "production_wiring", lambda: stub_wiring(encoder_cls))


def stub_family(
    index: ChampionIndex | None = None,
    *,
    seeds: int = 3,
    temperature: float = 1.3,
    encoder_cls=StubSetEncoder,
    calibrated: bool = True,
) -> SetDraftPredictor:
    """A live format-4 predictor over the stub encoder and the stub slot inputs."""
    index = index or stub_index()
    weights = stub_weights(seeds=seeds)
    modules = []
    for state in weights["states"]:
        module = encoder_cls(weights["config"], weights["d_in"])
        module.load_state_dict(state)
        modules.append(module.eval())
    curve = _identity_curve() if calibrated else None
    return SetDraftPredictor(
        modules,
        weights=weights,
        build_inputs=StubInputs(index),
        champion_index=index,
        thresholds=THRESHOLDS,
        temperature=temperature,
        edge_curve=curve,
        edge_cuts_pp=EDGE_CUTS_PP if calibrated else None,
        recipe=RECIPE,
        knowledge=stub_knowledge(index),
    )


def write_set_family(root: Path, index: ChampionIndex | None = None, **kwargs) -> Path:
    """A complete format-4 family on disk at ``root``; returns the family root."""
    stub_family(index, **kwargs).save(root)
    return root


def stub_draft(blue=None, red=None, *, blue_bans=(), red_bans=(), when=WHEN) -> DraftState:
    """A champion-only board; the defaults are the conftest corpus's ten picks."""
    return build_hypothetical_draft(
        blue=dict(BLUE_PICKS if blue is None else blue),
        red=dict(RED_PICKS if red is None else red),
        blue_bans=list(blue_bans),
        red_bans=list(red_bans),
        when=when,
    )


def mirrored(draft: DraftState) -> DraftState:
    """The same board with the sides swapped, bans included."""
    return stub_draft(
        {slot.role: slot.champion_id for slot in draft.red.slots},
        {slot.role: slot.champion_id for slot in draft.blue.slots},
        blue_bans=list(draft.red.bans or ()),
        red_bans=list(draft.blue.bans or ()),
        when=draft.date,
    )


def role_swapped(draft: DraftState, side: str, role: str, champion: str) -> DraftState:
    """The board with one slot replaced, built through the same constructor as the original."""
    picks = {
        name: {slot.role: slot.champion_id for slot in comp.slots}
        for name, comp in (("blue", draft.blue), ("red", draft.red))
    }
    picks[side][role] = champion
    return stub_draft(picks["blue"], picks["red"], when=draft.date)


def slot_row(side: str, role: str) -> int:
    """The slot-input row a (side, role) cell owns: blue top..sup, then red."""
    return (0 if side == "blue" else N_ROLES) + ROLE_ORDER.index(role)


# --- The real encoder's scaffolding: a kit-and-power member and boards drawn from a pool ---

#: Four candidates per role, so a board can swap one slot and nothing else.
POOL: tuple[tuple[str, ...], ...] = (
    ("Gnar", "Renekton", "Aatrox", "Jax"),
    ("LeeSin", "Viego", "Elise", "Sejuani"),
    ("Ahri", "Syndra", "Azir", "Orianna"),
    ("Jinx", "Caitlyn", "Ezreal", "Aphelios"),
    ("Thresh", "Lulu", "Nautilus", "Rakan"),
)
CHAMPIONS: tuple[str, ...] = tuple(name for role_pool in POOL for name in role_pool)


def kit_knowledge(champions: tuple[str, ...] = CHAMPIONS, seed: int = 5) -> KnowledgeTables:
    """A member carrying a kit matrix and a power table, both zero on the UNKNOWN row."""
    from league_predictor.draft_grading.champion_attributes.kit_vector import KIT_FAMILY_NAMES
    from league_predictor.draft_grading.grader.power_curve_features import (
        PowerTables,
        power_table_shape,
    )

    rng = np.random.default_rng(seed)
    rows = len(champions) + 1
    kit = rng.normal(size=(rows, len(KIT_FAMILY_NAMES))).astype(np.float32)
    table = rng.normal(size=power_table_shape(rows)).astype(np.float32)
    kit[-1] = 0.0
    table[-1] = 0.0
    return KnowledgeTables(
        schema_version=KNOWLEDGE_SCHEMA_VERSION,
        feature_names=("z0",),
        champions=tuple(champions),
        z_matrix=np.zeros((rows, 1), dtype=np.float32),
        power=PowerTables(table, np.zeros(table.shape[:2], dtype=np.bool_)),
        kit=kit,
    )


def board(match_id: str, blue: list[str], red: list[str], day: str = "2025-06-01") -> DraftState:
    """One ten-pick board over ``ROLE_ORDER``, built without the conftest corpus."""
    from league_predictor.draft_grading.state import DraftState, PlayerSlot, TeamComp

    def comp(team: str, champions: list[str]) -> TeamComp:
        return TeamComp(
            team_id=team,
            slots=tuple(
                PlayerSlot(role=role, champion_id=champion, player_id=f"{team}_{role}")
                for role, champion in zip(ROLE_ORDER, champions, strict=True)
            ),
        )

    return DraftState(
        match_id=match_id,
        date=datetime.fromisoformat(f"{day}T00:00:00"),
        patch="16.1",
        region="LCK",
        blue=comp("blue", blue),
        red=comp("red", red),
    )


def boards(n: int, seed: int = 11) -> list[DraftState]:
    """``n`` seeded boards drawn from `POOL`, one day apart."""
    rng = np.random.default_rng(seed)
    return [
        board(
            f"g{i:04d}",
            [POOL[r][int(rng.integers(0, len(POOL[r])))] for r in range(N_ROLES)],
            [POOL[r][int(rng.integers(0, len(POOL[r])))] for r in range(N_ROLES)],
            day=f"2025-06-{(i % 28) + 1:02d}",
        )
        for i in range(n)
    ]


def _identity_curve():
    return EdgeCurve(np.linspace(0.0, 1.0, EDGE_CURVE_KNOTS))
