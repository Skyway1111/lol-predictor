"""Decayed champion-role stats from chronological corpus prefixes."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.decay import half_life_rate
from league_predictor.draft_grading.grader.pro_excess import (
    HALF_LIFE_DAYS,
    PRO_EXCESS_BASELINES,
    SHRINKAGE_K,
    _baseline_map,
    ordinal_or_none,
)

if TYPE_CHECKING:
    import sqlite3


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _DecayedPrefix:
    """Chronological sufficient statistics for strict-before-date decay queries.

    Attributes:
        total: carried exponentially-decayed event count immediately after event ``i``.
        excess: the same carry for residuals; decaying it across the gap to a lookup date
            equals rebuilding ``exp(-rate * days_ago)`` over the full history.
    """

    ordinals: np.ndarray
    total: np.ndarray
    excess: np.ndarray
    decay_rate: float

    @classmethod
    def from_entries(
        cls,
        entries: list[tuple[int, int, float]],
        decay_rate: float,
    ) -> _DecayedPrefix:
        ordered = sorted(entries, key=lambda entry: entry[1])
        ordinals = np.fromiter((entry[1] for entry in ordered), dtype=np.int32)
        residuals = np.fromiter((entry[0] - entry[2] for entry in ordered), dtype=np.float64)
        return cls._build(ordinals, residuals, decay_rate)

    @classmethod
    def from_ordinals(cls, values: list[int], decay_rate: float) -> _DecayedPrefix:
        ordinals = np.sort(np.asarray(values, dtype=np.int32))
        return cls._build(ordinals, np.zeros(ordinals.size, dtype=np.float64), decay_rate)

    @classmethod
    def _build(
        cls,
        ordinals: np.ndarray,
        residuals: np.ndarray,
        decay_rate: float,
    ) -> _DecayedPrefix:
        total = np.empty(ordinals.size, dtype=np.float64)
        excess = np.empty(ordinals.size, dtype=np.float64)
        carried_total = 0.0
        carried_excess = 0.0
        previous = int(ordinals[0]) if ordinals.size else 0
        for i, (ordinal, residual) in enumerate(zip(ordinals, residuals, strict=True)):
            current = int(ordinal)
            factor = math.exp(-decay_rate * (current - previous))
            carried_total = carried_total * factor + 1.0
            carried_excess = carried_excess * factor + float(residual)
            total[i] = carried_total
            excess[i] = carried_excess
            previous = current
        return cls(ordinals, total, excess, decay_rate)

    def before(self, ref_ordinal: int) -> tuple[float, float]:
        """Return decayed ``(total, residual_sum)`` strictly before ``ref_ordinal``."""
        # Matching dtypes keeps searchsorted from copying the whole prefix for one date.
        index = int(self.ordinals.searchsorted(np.int32(ref_ordinal), side="left")) - 1
        if index < 0:
            return (0.0, 0.0)
        gap = ref_ordinal - int(self.ordinals[index])
        factor = math.exp(-self.decay_rate * gap)
        return (float(self.total[index] * factor), float(self.excess[index] * factor))


class BulkStatsCache:
    """Pre-loaded champion stats for batch feature computation, in-memory after a few queries.

    Attributes:
        baseline: the debias map; ``impact_bt`` raises via
            :func:`require_impact_history` when the grid is missing or empty.
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        half_life_days: float = HALF_LIFE_DAYS,
        baseline: str = "team_form",
    ):
        if baseline not in PRO_EXCESS_BASELINES:
            raise ValueError(
                f"unknown pro_excess_baseline {baseline!r}; known: {PRO_EXCESS_BASELINES}"
            )
        self.baseline = baseline
        self._decay_rate = half_life_rate(half_life_days)
        baseline_map = _baseline_map(db, baseline)

        logger.info("BulkStatsCache: loading game_players...")
        rows = db.execute(
            "SELECT gp.champion_id, gp.role, gp.result, gp.team_id, gp.game_id,"
            " date(s.scheduled_ts) "
            "FROM game_players gp JOIN games g ON g.game_id = gp.game_id "
            "JOIN series s ON s.series_id = g.series_id "
            "WHERE s.scheduled_ts IS NOT NULL"
        ).fetchall()
        logger.info("BulkStatsCache: loaded %d rows", len(rows))

        # Compile champion-role event lists into chronological decayed prefixes.
        champ_role: dict[tuple[str, str], list[tuple[int, int, float]]] = {}
        role_ordinals: dict[str, list[int]] = {}

        for champ_id, role, result, team_id, game_id, date_str in rows:
            ordinal = ordinal_or_none(date_str)
            if ordinal is None:
                continue
            form = baseline_map.get((team_id, game_id), 0.5)
            entry = (result, ordinal, form)

            champ_role.setdefault((champ_id, role), []).append(entry)
            role_ordinals.setdefault(role, []).append(ordinal)

        # Prefix states preserve strict-before decay with O(log n) lookups across training.
        self._champ_role_prefix = {
            key: _DecayedPrefix.from_entries(entries, self._decay_rate)
            for key, entries in champ_role.items()
        }
        self._role_prefix = {
            role: _DecayedPrefix.from_ordinals(ords, self._decay_rate)
            for role, ords in role_ordinals.items()
        }

        self._role_total_cache: dict[tuple[str, int], float] = {}
        self._pro_stats_cache: dict[tuple[str, str, int], tuple[float, float, float]] = {}
        logger.info(
            "BulkStatsCache: indexed %d champ-role pairs",
            len(self._champ_role_prefix),
        )

    @property
    def champions(self) -> frozenset[str]:
        """The champion spellings the facts carry, the keys every stat read resolves under."""
        return frozenset(champion for champion, _role in self._champ_role_prefix)

    def _excess(
        self,
        entry: _DecayedPrefix | None,
        ref_ordinal: int,
    ) -> tuple[float, float]:
        """``(excess, total_weight)`` for one champ(-role) entry; ``(0.0, 0.0)`` if empty."""
        if entry is None:
            return (0.0, 0.0)
        sum_w_total, sum_w_excess = entry.before(ref_ordinal)
        if sum_w_total == 0.0:
            return (0.0, 0.0)
        return (sum_w_excess / (sum_w_total + SHRINKAGE_K), sum_w_total)

    def _role_total(self, role: str, ref_ordinal: int) -> float:
        key = (role, ref_ordinal)
        if key in self._role_total_cache:
            return self._role_total_cache[key]

        prefix = self._role_prefix.get(role)
        if prefix is None:
            self._role_total_cache[key] = 0.0
            return 0.0

        total, _ = prefix.before(ref_ordinal)

        self._role_total_cache[key] = total
        return total

    def n_eff(self, champion: str, role: str, as_of: datetime) -> float:
        """Decayed evidence for one champion and role before the query date."""
        return float(self.pro_stats(champion, role, as_of)[2])

    def pro_stats(
        self, champion: str, role: str, as_of_date: str | datetime
    ) -> tuple[float, float, float]:
        """Decayed ``(excess, pick_rate, total_weight)`` for one champion and role."""
        ref_ordinal = self._ref_ordinal(as_of_date)
        key = (champion, role, ref_ordinal)
        cached = self._pro_stats_cache.get(key)
        if cached is not None:
            return cached
        excess, sum_w_total = self._excess(
            self._champ_role_prefix.get((champion, role)), ref_ordinal
        )
        if sum_w_total == 0.0:
            result = (0.0, 0.0, 0.0)
            self._pro_stats_cache[key] = result
            return result
        role_total = self._role_total(role, ref_ordinal)
        pick_rate = sum_w_total / role_total if role_total > 0 else 0.0
        result = (excess, pick_rate, sum_w_total)
        self._pro_stats_cache[key] = result
        return result

    @staticmethod
    def _ref_ordinal(as_of_date: str | datetime) -> int:
        if isinstance(as_of_date, datetime):
            return as_of_date.toordinal()
        return datetime.fromisoformat(as_of_date).toordinal()
