"""Walk-forward dataset assembly: a date window into the model's ``(X, y)`` matrix."""

from __future__ import annotations

import math
from collections.abc import Sequence
from collections.abc import Sequence as SequenceABC
from dataclasses import dataclass
from typing import TYPE_CHECKING

from league_predictor.asof import AsOf
from league_predictor.predraft.cleaning import (
    competitive_map_eligible,
    enriched_matches,
)

if TYPE_CHECKING:
    import sqlite3
    from datetime import date

    import numpy as np


@dataclass(frozen=True)
class SegmentRow:
    """One training-eligible map in reference orientation (sorted team id)."""

    match_id: str
    as_of: AsOf  # the map's own position: features are built at the map, never its day
    team_a: str  # sorted-canonical reference orientation (team_a <= team_b)
    team_b: str
    won: int  # 1 if team_a won this map
    tournament: str


def recency_weights(
    dates: Sequence[date], *, reference: date, tau: float, floor: float
) -> list[float]:
    """Per-row recency weight ``max(exp(-days_before_reference / tau), floor)``, tau in days."""
    return [max(math.exp(-max((reference - d).days, 0) / tau), floor) for d in dates]


def rows_in_window(conn: sqlite3.Connection, start: date, end: date) -> list[SegmentRow]:
    """Training-eligible maps in ``[start, end)``, reference orientation, label = team_a won."""
    rows: list[SegmentRow] = []
    for m in enriched_matches(conn, start=start.isoformat(), end=end.isoformat()):
        if not competitive_map_eligible(m):
            continue
        a, b = str(m["team1_id"]), str(m["team2_id"])
        team_a, team_b = (a, b) if a <= b else (b, a)
        rows.append(
            SegmentRow(
                match_id=str(m["match_id"]),
                as_of=AsOf.game(m["date"], m["series_id"], m["game_no"]),
                team_a=team_a,
                team_b=team_b,
                won=1 if str(m["winner"]) == team_a else 0,
                tournament=str(m["tournament"] or "unknown"),
            )
        )
    return rows


class ColumnarRows(SequenceABC):
    """Lazy dict-row facade over a float64 feature matrix; slices are views, not copies."""

    __slots__ = ("matrix", "names")

    def __init__(self, matrix: np.ndarray, names: Sequence[str]) -> None:
        self.matrix = matrix
        self.names = list(names)

    def __len__(self) -> int:
        return int(self.matrix.shape[0])

    def __getitem__(self, i):
        if isinstance(i, slice):
            return ColumnarRows(self.matrix[i], self.names)
        return {name: float(v) for name, v in zip(self.names, self.matrix[i], strict=True)}

    def take(self, indices: Sequence[int]) -> ColumnarRows:
        """The selected rows in ``indices`` order as a new ``ColumnarRows`` (fancy-index copy)."""
        return ColumnarRows(self.matrix[list(indices)], self.names)


@dataclass
class Dataset:
    """Orientation-augmented walk-forward matrix: every map appears twice, label flipped."""

    X: list[dict[str, float | str]] | ColumnarRows
    y: list[int]
    meta: list[dict]
    feature_names: list[str]
    categorical_names: list[str]


def window_queries(
    conn: sqlite3.Connection, start: date, end: date
) -> tuple[list[tuple[str, str, AsOf]], list[int], list[dict]]:
    """Orientation-augmented featurization inputs for eligible maps in ``[start, end)``.

    Returns:
        ``(queries, labels, meta)`` — one ``(team_a, team_b, as_of)`` query and label per
        (map, orientation); a map's two orientation rows share one meta dict, whose ``date``
        is the map's calendar day (the window and weight key) and ``as_of`` its position.
    """
    queries: list[tuple[str, str, AsOf]] = []
    y: list[int] = []
    meta: list[dict] = []
    for r in rows_in_window(conn, start, end):
        for team_a, team_b, label in (
            (r.team_a, r.team_b, r.won),
            (r.team_b, r.team_a, 1 - r.won),
        ):
            queries.append((team_a, team_b, r.as_of))
            y.append(label)
            meta.append(
                {
                    "date": r.as_of.day,
                    "as_of": r.as_of,
                    "match_id": r.match_id,
                    "tournament": r.tournament,
                    "team_a": team_a,
                    "team_b": team_b,
                }
            )
    return queries, y, meta


def validate_orientation_pairs(
    meta: Sequence[dict], labels: Sequence[int], weights: Sequence[float] | None = None
) -> None:
    """Assert exact adjacent swap pairs at the training verb boundary."""
    if len(meta) != len(labels) or len(meta) % 2:
        raise ValueError("orientation rows must have equal even metadata and label lengths")
    if weights is not None and len(weights) != len(meta):
        raise ValueError("orientation weights must align one-for-one with rows")
    for i in range(0, len(meta), 2):
        left, right = meta[i], meta[i + 1]
        if left["match_id"] != right["match_id"] or left["date"] != right["date"]:
            raise ValueError(f"orientation pair {i // 2} has mismatched match id/date")
        if left["team_a"] != right["team_b"] or left["team_b"] != right["team_a"]:
            raise ValueError(f"orientation pair {i // 2} does not reverse teams")
        if int(labels[i]) + int(labels[i + 1]) != 1:
            raise ValueError(f"orientation pair {i // 2} labels are not complementary")
        if weights is not None and float(weights[i]) != float(weights[i + 1]):
            raise ValueError(f"orientation pair {i // 2} does not preserve match weight")
