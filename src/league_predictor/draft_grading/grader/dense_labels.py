"""The set encoder's dense auxiliary labels, every one read from the facts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from operator import itemgetter
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.draft_grading.champion_attributes.classes import (
    CLASS_LABEL_TO_INDEX,
    UNKNOWN_CLASS,
    effective_class,
    ensure_loaded,
)
from league_predictor.draft_grading.grader.gd15_baseline import PlayerGd15Baseline
from league_predictor.draft_grading.grader.pro_excess import ordinal_or_none
from league_predictor.draft_grading.grader.slot_inputs import N_SLOTS
from league_predictor.vocab.roles import ROLE_ORDER, ROLE_RANK

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping, Sequence

#: The duration a game must exceed to carry the long-game label: 35 minutes, the bucket the
#: phase-0 duration read found the model equally confident and 6.3 pp less right in.
LONG_GAME_SECONDS = 2100

_SQL = """
SELECT gp.game_id, date(s.scheduled_ts) AS d, g.blue_team, g.red_team, g.duration_s,
       g.winner, gp.team_id, gp.role, gp.player_id, gp.champion_id,
       gp.golddiffat15, gp.dpm, gp.damageshare
FROM game_players gp
JOIN games g ON g.game_id = gp.game_id
JOIN series s ON s.series_id = g.series_id
WHERE g.blue_team IS NOT NULL AND s.scheduled_ts IS NOT NULL
  AND gp.role IN ('top', 'jng', 'mid', 'bot', 'sup')
ORDER BY d, gp.game_id
"""


@dataclass(frozen=True)
class DenseTargets:
    """Per-head labels and masks, row-aligned to the game ids handed in.

    Attributes:
        labels: head name -> ``(N,)`` or ``(N, 10)`` values; slot rows are blue then red in
            ``ROLE_ORDER``, so a mirrored board's labels are the half-swap.
        masks: head name -> the same shape in bool; False means the fact is absent and the
            head's loss skips the entry.
    """

    labels: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.labels["gd15"])

    def take(self, rows: slice) -> DenseTargets:
        """The same labels over a row slice — an OOF fold's prefix of the corpus."""
        return DenseTargets(
            {name: value[rows] for name, value in self.labels.items()},
            {name: mask[rows] for name, mask in self.masks.items()},
        )


@dataclass(frozen=True, slots=True)
class _Slot:
    """One game_players row reduced to what a dense label reads."""

    player_id: str
    champion_id: str
    gd15: float | None
    dpm: float | None
    damage_share: float | None


def _slots_of(rows: list, blue_team: str, red_team: str) -> dict[int, _Slot]:
    """The game's rows by slot index; a row on neither side of the pair never surfaces."""
    sides = {blue_team: 0, red_team: len(ROLE_ORDER)}
    slots: dict[int, _Slot] = {}
    for row in rows:
        offset = sides.get(row[6])
        if offset is None:
            continue
        slots[offset + ROLE_RANK[row[7]]] = _Slot(row[8], row[9], row[10], row[11], row[12])
    return slots


def dense_targets(db: sqlite3.Connection, game_ids: Sequence[str]) -> DenseTargets:
    """The five dense heads' labels for ``game_ids``, walk-forward and facts-only.

    Returns:
        Per-slot damage per minute, damage share and role class; the team gold diff at 15
        residualized by the player baseline read strictly before the game
        (`gd15_baseline.PlayerGd15Baseline`, the lane head's); and the blue-win label of
        games past ``LONG_GAME_SECONDS``, masked to them.

    Raises:
        RuntimeError: a requested game id carries no facts row — the corpus drifted under
            the fit, never a silently label-free game.
    """
    ensure_loaded()
    order = {game_id: i for i, game_id in enumerate(game_ids)}
    n = len(order)
    unknown_class = CLASS_LABEL_TO_INDEX[UNKNOWN_CLASS]
    labels: dict[str, np.ndarray] = {
        "dpm": np.zeros((n, N_SLOTS), dtype=np.float32),
        "damage_share": np.zeros((n, N_SLOTS), dtype=np.float32),
        "class": np.full((n, N_SLOTS), unknown_class, dtype=np.int64),
        "gd15": np.zeros(n, dtype=np.float32),
        "long_game": np.zeros(n, dtype=np.float32),
    }
    masks = {name: np.zeros(value.shape, dtype=np.bool_) for name, value in labels.items()}
    baselines = PlayerGd15Baseline()
    seen: set[str] = set()

    for game_id, group in groupby(db.execute(_SQL), key=itemgetter(0)):
        rows = list(group)
        ordinal = ordinal_or_none(rows[0][1])
        if ordinal is None:
            continue
        slots = _slots_of(rows, rows[0][2], rows[0][3])
        row = order.get(game_id)
        if row is not None:
            seen.add(game_id)
            _fill_game(
                labels,
                masks,
                row=row,
                slots=slots,
                head=rows[0],
                baselines=baselines,
                ordinal=ordinal,
            )
        # Baselines advance after the whole game is read: strictly-before semantics.
        for slot in slots.values():
            if slot.gd15 is not None:
                baselines.advance(slot.player_id, ordinal, float(slot.gd15))

    missing = [game_id for game_id in order if game_id not in seen]
    if missing:
        raise RuntimeError(
            f"dense labels: {len(missing)} of {n} games carry no facts row "
            f"(first: {missing[:5]}) — the corpus join drifted under the fit."
        )
    return DenseTargets(labels, masks)


def _fill_game(
    labels: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    *,
    row: int,
    slots: Mapping[int, _Slot],
    head: tuple,
    baselines: PlayerGd15Baseline,
    ordinal: int,
) -> None:
    """Write one game's labels; ``head`` is any of its rows, for the game-level columns."""
    for position, slot in slots.items():
        labels["class"][row, position] = CLASS_LABEL_TO_INDEX[
            effective_class(slot.champion_id, ROLE_ORDER[position % len(ROLE_ORDER)])
        ]
        for name, value in (("dpm", slot.dpm), ("damage_share", slot.damage_share)):
            if value is not None:
                labels[name][row, position] = value
                masks[name][row, position] = True
    masks["class"][row] = labels["class"][row] != CLASS_LABEL_TO_INDEX[UNKNOWN_CLASS]

    _, _, blue_team, _red, duration_s, winner = head[:6]
    if duration_s is not None and winner is not None and duration_s > LONG_GAME_SECONDS:
        labels["long_game"][row] = float(winner == blue_team)
        masks["long_game"][row] = True

    if len(slots) == N_SLOTS and all(slot.gd15 is not None for slot in slots.values()):
        blue = [slots[r] for r in range(len(ROLE_ORDER))]
        red = [slots[r + len(ROLE_ORDER)] for r in range(len(ROLE_ORDER))]
        baseline = sum(
            baselines.read(b.player_id, ordinal) - baselines.read(r.player_id, ordinal)
            for b, r in zip(blue, red, strict=True)
        )
        labels["gd15"][row] = sum(float(slot.gd15) for slot in blue) - baseline  # pyright: ignore[reportArgumentType]  every gd15 is non-None here
        masks["gd15"][row] = True
