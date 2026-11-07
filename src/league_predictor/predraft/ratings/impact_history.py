"""One-pass chronological Impact states for the complete historical grid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.decay import half_life_weight
from league_predictor.probability import sigmoid

if TYPE_CHECKING:
    from league_predictor.predraft.ratings.impact_corpus import ImpactCorpus


_LEARNING_RATE = 0.02
_HALF_LIFE_DAYS = 1460.0
_GRID_EPOCH = date(2017, 1, 1)


@dataclass(frozen=True, slots=True)
class ImpactHistoryRow:
    """One player's pre-cutoff online Impact state."""

    player_id: str
    asof_date: date
    impact_logodds: float


@dataclass(frozen=True, slots=True)
class ImpactHistoryBuild:
    """Complete history rows plus the eligible/influential corpus census."""

    rows: tuple[ImpactHistoryRow, ...]
    eligible_games: int
    influential_games: int
    first_game_date: date | None
    last_game_date: date | None


def _grid_cutoffs(first_game: date, through: date, grid_days: int, epoch: date) -> tuple[date, ...]:
    """Return epoch-aligned cutoffs after the first game plus one terminal tail cutoff."""
    if grid_days <= 0:
        raise ValueError(f"grid_days must be positive, got {grid_days}")
    epoch_day = epoch.toordinal()
    first_day = first_game.toordinal()
    if first_day < epoch_day:
        raise ValueError(f"first eligible game {first_game} precedes locked grid epoch {epoch}")
    first_step = ((first_day - epoch_day) // grid_days) + 1
    cutoff_day = epoch_day + first_step * grid_days
    terminal_day = through.toordinal() + 1
    cutoffs: list[date] = []
    while cutoff_day <= through.toordinal():
        cutoffs.append(date.fromordinal(cutoff_day))
        cutoff_day += grid_days
    terminal = date.fromordinal(terminal_day)
    if not cutoffs or cutoffs[-1] != terminal:
        cutoffs.append(terminal)
    return tuple(cutoffs)


def _eligible_games(corpus: ImpactCorpus, through_day: int) -> list[int]:
    """Indexes of the decided, two-sided games on or before ``through_day``, in play order."""
    eligible: list[int] = []
    for game_idx in range(len(corpus.series_ids)):
        day = int(corpus.scheduled_dates[game_idx])
        count_a, count_b = (int(value) for value in corpus.lineup_counts[game_idx])
        if day > through_day or not count_a or not count_b:
            continue
        if corpus.winners[game_idx] not in (corpus.team_a[game_idx], corpus.team_b[game_idx]):
            continue
        eligible.append(game_idx)
    eligible.sort(
        key=lambda idx: (
            int(corpus.scheduled_instants[idx]),
            corpus.series_ids[idx],
            int(corpus.game_nos[idx]),
        )
    )
    return eligible


def build_impact_history(
    corpus: ImpactCorpus,
    *,
    through: date,
    grid_days: int = 14,
) -> ImpactHistoryBuild:
    """Stream every eligible game once and emit strictly pre-cutoff player states."""
    eligible = _eligible_games(corpus, through.toordinal())
    if not eligible:
        return ImpactHistoryBuild((), 0, 0, None, None)

    first_day = int(corpus.scheduled_dates[eligible[0]])
    last_day = int(corpus.scheduled_dates[eligible[-1]])
    cutoffs = _grid_cutoffs(
        date.fromordinal(first_day),
        through,
        grid_days,
        _GRID_EPOCH,
    )

    n_players = len(corpus.player_ids)
    skill = np.zeros(n_players, dtype=np.float64)
    last_update = np.full(n_players, -1, dtype=np.int32)
    last_seen = np.full(n_players, -1, dtype=np.int32)
    rows: list[ImpactHistoryRow] = []
    cutoff_idx = 0
    influential_games = 0

    def emit(cutoff: date) -> None:
        cutoff_day = cutoff.toordinal()
        active = np.flatnonzero(last_seen >= 0)
        if not len(active):
            return
        decayed = skill[active] * half_life_weight(
            np.maximum(0, cutoff_day - last_update[active]), _HALF_LIFE_DAYS
        )
        rows.extend(
            ImpactHistoryRow(corpus.player_ids[int(player_idx)], cutoff, float(value))
            for player_idx, value in zip(active, decayed, strict=True)
        )

    for game_idx in eligible:
        day = int(corpus.scheduled_dates[game_idx])
        while cutoff_idx < len(cutoffs) and cutoffs[cutoff_idx].toordinal() <= day:
            emit(cutoffs[cutoff_idx])
            cutoff_idx += 1

        count_a, count_b = (int(value) for value in corpus.lineup_counts[game_idx])
        lineup_a = corpus.lineups[game_idx, 0, :count_a].astype(np.int64, copy=False)
        lineup_b = corpus.lineups[game_idx, 1, :count_b].astype(np.int64, copy=False)
        players = np.concatenate((lineup_a, lineup_b))
        unique, counts = np.unique(players, return_counts=True)
        if np.any(counts != 1):
            duplicate_idx = int(unique[np.flatnonzero(counts != 1)[0]])
            raise ValueError(
                f"game {corpus.series_ids[game_idx]} has duplicate player "
                f"{corpus.player_ids[duplicate_idx]!r}"
            )
        skill[players] *= half_life_weight(
            np.maximum(0, day - last_update[players]), _HALF_LIFE_DAYS
        )
        last_update[players] = day
        last_seen[players] = day
        margin = float(skill[lineup_a].sum() - skill[lineup_b].sum())
        probability_a = sigmoid(margin)
        outcome_a = 1.0 if corpus.winners[game_idx] == corpus.team_a[game_idx] else 0.0
        step = _LEARNING_RATE * (outcome_a - probability_a)
        skill[lineup_a] += step
        skill[lineup_b] -= step
        influential_games += 1

    while cutoff_idx < len(cutoffs):
        emit(cutoffs[cutoff_idx])
        cutoff_idx += 1

    return ImpactHistoryBuild(
        rows=tuple(rows),
        eligible_games=len(eligible),
        influential_games=influential_games,
        first_game_date=date.fromordinal(first_day),
        last_game_date=date.fromordinal(last_day),
    )
