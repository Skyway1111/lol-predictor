"""The leak-free original pre-draft feature family: a persisted, extendable columnar index."""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING, ClassVar

import numpy as np

from league_predictor.asof import AsOf, MatchContext
from league_predictor.db.facts_prefix import main_db_path
from league_predictor.predraft.features.manifest import (
    SCHEMA_VERSION,
    diff_keys,
    feature_schema_fingerprint,
    full_feature_universe,
)
from league_predictor.predraft.features.snapshots import SnapshotStore, build_pre_draft_features
from league_predictor.predraft.pipeline import fit_corpus_store
from league_predictor.predraft.pipeline.dataset import rows_in_window

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

#: Bump when the index's row semantics change without a manifest change (the snapshot
#: stack's meaning). Facts drift is the store's digest; the manifest is the key's fingerprint.
_CACHE_VERSION = 5
_KIND = "original"

_NEUTRAL_CONTEXT = ("Bo1", "regular", 0, 0)

#: A persisted blob row: ``(team_a, team_b, as_of key)``, both orientations of every map.
_Row = tuple[str, str, str]


@dataclass(frozen=True)
class _IndexKey:
    """The schema half of the index's identity (``fit_corpus_store.StoreKey``)."""

    feature_names: tuple[str, ...]
    schema_fingerprint: str

    @classmethod
    def current(cls) -> _IndexKey:
        return cls(
            feature_names=tuple(sorted(full_feature_universe())),
            schema_fingerprint=f"{SCHEMA_VERSION}:{feature_schema_fingerprint()}",
        )

    def as_tuple(self) -> tuple:
        return (_CACHE_VERSION, self.feature_names, self.schema_fingerprint)

    @property
    def group(self) -> str:
        return fit_corpus_store.group_of(self.as_tuple())


def _match_context(conn: sqlite3.Connection) -> dict[str, tuple[str, str, int, int]]:
    """``game_id -> (format, stage, team_a maps won, team_b maps won)`` before that map.

    Returns:
        Bo1/regular for a NULL best_of/event_type; the wins count the series' earlier maps
        in the canonical ``team_a < team_b`` orientation.
    """
    rows = conn.execute(
        "SELECT g.game_id, 'Bo' || s.best_of, g.event_type,"
        " SUM(g.winner = s.team_a) OVER w, SUM(g.winner = s.team_b) OVER w"
        " FROM games g JOIN series s ON s.series_id = g.series_id"
        " WINDOW w AS (PARTITION BY g.series_id ORDER BY g.game_no"
        " ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING)"
    ).fetchall()
    return {
        str(r[0]): (str(r[1] or "Bo1"), str(r[2] or "regular"), int(r[3] or 0), int(r[4] or 0))
        for r in rows
    }


def _swap_built_features(
    store: SnapshotStore, forward: dict, *, team_a: str, team_b: str, as_of: AsOf
) -> dict:
    """Exact ``B vs A`` row from one fully built ``A vs B`` row (no lossy ``1-p`` shortcuts)."""
    swapped = dict(forward)
    for name in forward:
        if name.startswith("t1_") and f"t2_{name[3:]}" in forward:
            swapped[name] = forward[f"t2_{name[3:]}"]
        elif name.startswith("t2_") and f"t1_{name[3:]}" in forward:
            swapped[name] = forward[f"t1_{name[3:]}"]
    for base in diff_keys():
        swapped[f"diff_{base}"] = swapped.get(f"t1_{base}", 0.0) - swapped.get(f"t2_{base}", 0.0)
    swapped["diff_kda"] = max(-3.0, min(3.0, swapped["diff_kda"]))
    swapped.update(store.h2h(team_b, team_a, as_of))
    return swapped


def _featurize(
    conn: sqlite3.Connection, names: Sequence[str], *, start: date, through: date
) -> tuple[list[_Row], np.ndarray]:
    """Both orientations of every eligible map in ``[start, through)`` as rows of ``names``.

    Raises:
        RuntimeError: the composed row's columns differ from the locked manifest.
    """
    conn.row_factory = sqlite3.Row  # rows_in_window reads by column name
    rows = rows_in_window(conn, start, through)
    ctx = _match_context(conn)
    store = SnapshotStore(conn, teams={r.team_a for r in rows} | {r.team_b for r in rows})
    keys: list[_Row] = []
    matrix = np.empty((2 * len(rows), len(names)), dtype=np.float64)
    t0 = time.monotonic()
    for i, r in enumerate(rows):
        fmt, stage, wins_a, wins_b = ctx.get(r.match_id, _NEUTRAL_CONTEXT)
        forward = build_pre_draft_features(
            conn,
            r.team_a,
            r.team_b,
            fmt,
            stage,
            r.as_of,
            series_wins=(wins_a, wins_b),
            prune=False,
            store=store,
        )
        if i == 0 and set(forward) != set(names):
            raise RuntimeError(
                "original feature schema differs from the locked manifest: "
                f"missing={sorted(set(names) - set(forward))[:5]}, "
                f"unexpected={sorted(set(forward) - set(names))[:5]}"
            )
        swapped = _swap_built_features(
            store, forward, team_a=r.team_a, team_b=r.team_b, as_of=r.as_of
        )
        matrix[2 * i] = [forward[n] for n in names]
        matrix[2 * i + 1] = [swapped[n] for n in names]
        keys.append((r.team_a, r.team_b, r.as_of.key))
        keys.append((r.team_b, r.team_a, r.as_of.key))
    logger.info(
        "original index: %d maps in [%s, %s) featurized in %.0fs",
        len(rows),
        start,
        through,
        time.monotonic() - t0,
    )
    return keys, matrix


def build_original_index(conn: sqlite3.Connection, *, through: date) -> _OriginalFeaturesIndex:
    """The index over every eligible map before ``through``; persisted under ``data/cache/``.

    Returns:
        An exact hit loads the blob; a moved ``through`` featurizes only the maps after the
        newest still-valid blob's data-end and appends (rows are as-of their own map, so the
        result is row-identical to a from-scratch build); a historical correction before that
        data-end fails the blob's facts digest and rebuilds. A DB with no backing file builds
        in memory and persists nothing. COST: a cold build is minutes (benchmarks.md).
    """
    key = _IndexKey.current()
    names = list(key.feature_names)
    if not main_db_path(conn):
        keys, matrix = _featurize(conn, names, start=date.min, through=through)
        return _OriginalFeaturesIndex(matrix, keys, names, conn=conn)
    payload, digest = fit_corpus_store.find_extendable(conn, key, end=through, kind=_KIND)
    if payload is None:
        keys, matrix = _featurize(conn, names, start=date.min, through=through)
    else:
        keys, matrix = list(payload["rows"]), payload["matrix"]
        prior_end = date.fromisoformat(payload["end"])
        if prior_end == through:
            return _OriginalFeaturesIndex(matrix, keys, names, conn=conn)
        tail_keys, tail_matrix = _featurize(conn, names, start=prior_end, through=through)
        keys += tail_keys
        matrix = np.concatenate([matrix, tail_matrix])
    fit_corpus_store.store(key, end=through, digest=digest, kind=_KIND, matrix=matrix, rows=keys)
    return _OriginalFeaturesIndex(matrix, keys, names, conn=conn)


class _OriginalFeaturesIndex:
    """O(1) query read over the precomputed rows; absent feature keys default to 0.0.

    Raises:
        KeyError: A conn-less index queried for a matchup outside the precompute — the coverage
            tripwire. A conn-backed index instead computes it on demand at the query.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        rows: Sequence[_Row],
        feature_names: Sequence[str],
        conn: sqlite3.Connection | None = None,
        *,
        positions: dict[_Row, int] | None = None,
    ) -> None:
        self._matrix = matrix
        self._rows = list(rows)
        self._positions = positions or {row: i for i, row in enumerate(self._rows)}
        self.feature_names = list(feature_names)
        self._conn = conn

    def rebound(self, conn) -> _OriginalFeaturesIndex:
        """Same rows by reference, fallback rebound to ``conn`` (conns are thread-affine)."""
        return _OriginalFeaturesIndex(
            self._matrix, self._rows, self.feature_names, conn=conn, positions=self._positions
        )

    def _raw_features(self, team_a: str, team_b: str, as_of: AsOf) -> dict:
        """Return the stored full row, computing the sanctioned fallback when absent."""
        position = self._positions.get((team_a, team_b, as_of.key))
        if position is not None:
            return dict(zip(self.feature_names, self._matrix[position].tolist(), strict=True))
        if self._conn is None:
            raise KeyError(
                f"original-features index has no entry for ({team_a}, {team_b}, {as_of}); "
                "the query is past `through` or outside the precompute"
            )
        ctx = as_of.context or self._scheduled_match_context(self._conn, team_a, team_b, as_of)
        # ceiling: one two-team SnapshotStore per uncovered query, ~0.1 s; a per-connection
        # lazy store if a live surface ever prices hundreds of uncovered matchups at once.
        return build_pre_draft_features(
            self._conn,
            team_a,
            team_b,
            ctx.format,
            ctx.stage,
            as_of,
            series_wins=(ctx.wins_for(team_a), ctx.wins_for(team_b)),
            prune=False,
        )

    def _scheduled_match_context(
        self, conn: sqlite3.Connection, team_a: str, team_b: str, as_of: AsOf
    ) -> MatchContext:
        """The context of an on-demand matchup without one: its own scheduled match, else neutral.

        Returns:
            The pair's nearest unfinished ``upcoming_matches`` row (running, or not started on
            or after the query's day) supplies ``BoN``, its stage and its live score, so a
            playoff prediction prices ``is_playoffs=1`` and a running series its state; no
            row — or no schedule table (bare facts DBs) — reads as the neutral context.
        """
        lo, hi = (team_a, team_b) if team_a < team_b else (team_b, team_a)
        try:
            match = conn.execute(
                "SELECT best_of, stage, score_a, score_b FROM upcoming_matches"
                " WHERE team_a = ? AND team_b = ? AND ("
                "  ps_status = 'running'"
                "  OR (ps_status = 'not_started' AND date(scheduled_ts) >= ?))"
                " ORDER BY scheduled_ts LIMIT 1",
                (lo, hi, as_of.day.isoformat()),
            ).fetchone()
        except sqlite3.OperationalError:
            return MatchContext()
        if match is None:
            return MatchContext()
        fmt = f"Bo{int(match[0])}" if match[0] else "Bo1"
        wins = {lo: int(match[2] or 0), hi: int(match[3] or 0)}
        return MatchContext.of(fmt, str(match[1] or "regular"), wins)

    def features(self, team_a: str, team_b: str, as_of: AsOf) -> dict[str, float | str]:
        row = self._raw_features(team_a, team_b, as_of)
        return {name: float(row.get(name, 0.0)) for name in self.feature_names}

    def features_matrix(self, queries, feature_names):
        """Read selected columns straight into float64, bypassing one dict copy per row."""
        unexpected = sorted(set(feature_names) - set(self.feature_names))
        if unexpected:
            raise KeyError(f"unknown original feature columns: {unexpected}")
        positions = [
            self._positions.get((team_a, team_b, as_of.key), -1)
            for team_a, team_b, as_of in queries
        ]
        if all(position >= 0 for position in positions):
            columns = [self.feature_names.index(name) for name in feature_names]
            return self._matrix[positions][:, columns]

        def values():
            for team_a, team_b, as_of in queries:
                row = self._raw_features(team_a, team_b, as_of)
                for name in feature_names:
                    yield float(row.get(name, 0.0))

        return np.fromiter(
            values(), dtype=np.float64, count=len(queries) * len(feature_names)
        ).reshape(len(queries), len(feature_names))


class OriginalFeaturesFamily:
    """The full pre-pruning feature set as a single earnable family.

    Attributes:
        feature_names: From the locked manifest, so a persisted corpus is identifiable without
            deserializing the index. ``build`` requires the emitted set to equal it exactly; a
            missing or unexpected column is a hard schema error.
    """

    name = "original"
    feature_names: list[str] = sorted(full_feature_universe())
    categorical_names: ClassVar[list[str]] = []

    def build(
        self, conn: sqlite3.Connection, *, through: date, query_dates: Sequence[date]
    ) -> _OriginalFeaturesIndex:
        del query_dates
        return build_original_index(conn, through=through)

    def fallback_index(self, conn: sqlite3.Connection) -> _OriginalFeaturesIndex:
        """A conn-backed on-demand index for rare rows absent from a persisted fit corpus."""
        return _OriginalFeaturesIndex(
            np.empty((0, len(self.feature_names))), [], list(self.feature_names), conn=conn
        )
