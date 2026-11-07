"""The walk-forward fit seam — the index, window, corpus and month primitives every fit shares."""

from __future__ import annotations

import bisect
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, cast

from league_predictor.clock import trailing_months
from league_predictor.db.facts_prefix import main_db_path
from league_predictor.predraft.features import FEATURE_FAMILIES, FamilyIndex, FeatureFamily
from league_predictor.predraft.pipeline import fit_corpus_store
from league_predictor.predraft.pipeline.dataset import (
    ColumnarRows,
    Dataset,
    recency_weights,
    validate_orientation_pairs,
    window_queries,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from league_predictor.predraft.pipeline.chosen_model import _ServedRecipe
    from league_predictor.predraft.pipeline.lightgbm_bag import LightGBMBagBackend
    from league_predictor.predraft.pipeline.model_contract import FittedModel


@dataclass(frozen=True)
class WindowParams:
    """The fixed recipe's training-window and recency schedule.

    Attributes:
        tau: recency decay speed, in days.
        floor: flat minimum weight an old row keeps.
        cap_days: drop rows older than ``reference - cap_days``; ``None`` keeps all history.
    """

    tau: float
    floor: float
    cap_days: int | None


def build_indexes(
    conn: sqlite3.Connection,
    families: Sequence[str],
    *,
    through: date,
) -> tuple[list[FeatureFamily], dict[str, FamilyIndex]]:
    """Build each family's index once, excluding rows at or after ``through``."""
    fams = [FEATURE_FAMILIES[name]() for name in families]
    indexes = {f.name: f.build(conn, through=through, query_dates=[]) for f in fams}
    return fams, indexes


def _build_indexes_overlapping_original(
    conn: sqlite3.Connection, families: Sequence[str], *, through: date
):
    """``build_indexes`` with the ``original`` build overlapped onto a worker thread.

    Returns:
        ``(fams, indexes)`` identical to the serial build — same family objects in config
        order, same index contents. The worker reads through its own connection and writes
        only the index blob under ``data/cache/``; an in-memory DB or an ``original``-less
        family list falls back to the serial build.
    """
    db_path = main_db_path(conn)
    if "original" not in families or not db_path:
        return build_indexes(conn, families, through=through)

    fams = [FEATURE_FAMILIES[name]() for name in families]
    orig = next(f for f in fams if f.name == "original")

    def _build_original():
        worker_conn = sqlite3.connect(db_path)
        try:
            return orig.build(worker_conn, through=through, query_dates=[])
        finally:
            worker_conn.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_build_original)
        built = {
            f.name: f.build(conn, through=through, query_dates=[])
            for f in fams
            if f.name != "original"
        }
        built["original"] = future.result().rebound(conn)  # pyright: ignore[reportAttributeAccessIssue]  # original family's index carries rebound; not on the FamilyIndex protocol
    return fams, {f.name: built[f.name] for f in fams}  # deterministic family order


# FeaturizedCorpus featurizes in row chunks of this size: ~0.3 GB transient instead of the
# ~3.5 GB an all-at-once build holds. Chunking is value-preserving (same per-cell conversion).
_FEATURIZE_CHUNK_ROWS = 16_384


class FeaturizedCorpus:
    """A window span featurized once into one date-sorted float64 matrix plus ``y``/``meta``.

    Attributes:
        feature_names: the matrix's column order.
        categorical_names: always empty — float64 storage cannot hold a declared categorical,
            so a family set emitting one raises in ``__init__`` rather than being coerced.
        start: span start; ``end`` its exclusive span end. Rows are featurized as-of their own
            match date, never a window's ``train_end``, so any window is a bisected view.
    """

    def __init__(
        self,
        conn,
        *,
        start: date,
        end: date,
        families: Sequence[FeatureFamily],
        indexes: Mapping[str, FamilyIndex],
        drop_features: frozenset[str],
    ) -> None:
        import numpy as np

        from league_predictor.predraft.features import build_feature_matrix_batch

        categorical_names = [
            cn for f in families for cn in f.categorical_names if cn not in drop_features
        ]
        if categorical_names:
            raise ValueError(
                f"FeaturizedCorpus stores a float64 matrix and cannot hold declared "
                f"categorical columns {categorical_names}"
            )
        feature_names = [fn for f in families for fn in f.feature_names if fn not in drop_features]
        queries, y, meta = window_queries(conn, start, end)
        matrix = np.empty((len(queries), len(feature_names)), dtype=np.float64)
        for lo in range(0, len(queries), _FEATURIZE_CHUNK_ROWS):
            chunk = queries[lo : lo + _FEATURIZE_CHUNK_ROWS]
            matrix[lo : lo + len(chunk)] = build_feature_matrix_batch(
                chunk,
                indexes=indexes,
                families=families,
                drop_features=drop_features,
            )
        order = sorted(range(len(meta)), key=lambda i: meta[i]["date"])
        self._matrix = matrix[order]
        self._rows = ColumnarRows(self._matrix, feature_names)
        self._y = [y[i] for i in order]
        self._meta = [meta[i] for i in order]
        self._dates = [m["date"] for m in self._meta]
        self.feature_names = feature_names
        self.categorical_names = categorical_names
        self.drop_features = frozenset(drop_features)
        self.start = start
        self.end = end

    @classmethod
    def _from_parts(
        cls, *, matrix, y, meta, feature_names, drop_features, start: date, end: date
    ) -> FeaturizedCorpus:
        """A corpus from already-featurized, already date-sorted parts (snapshot load/extend)."""
        self = object.__new__(cls)
        self._matrix = matrix
        self._rows = ColumnarRows(matrix, list(feature_names))
        self._y = list(y)
        self._meta = list(meta)
        self._dates = [m["date"] for m in self._meta]
        self.feature_names = list(feature_names)
        self.categorical_names = []
        self.drop_features = frozenset(drop_features)
        self.start = start
        self.end = end
        return self

    def narrowed(self, drop_features: frozenset[str]) -> FeaturizedCorpus:
        """This corpus under a wider drop set: the kept columns, same rows, labels and span.

        Returns:
            ``self`` when the drop set is this corpus's own; a column-subset copy otherwise.

        Raises:
            ValueError: ``drop_features`` asks for a column this corpus never featurized.
        """
        drop_features = frozenset(drop_features)
        if drop_features == self.drop_features:
            return self
        if not drop_features >= self.drop_features:
            raise ValueError(
                "corpus was featurized with a different drop_features set: it lacks "
                f"{sorted(self.drop_features - drop_features)}"
            )
        kept = [i for i, name in enumerate(self.feature_names) if name not in drop_features]
        return FeaturizedCorpus._from_parts(
            matrix=self._matrix[:, kept],
            y=self._y,
            meta=self._meta,
            feature_names=[self.feature_names[i] for i in kept],
            drop_features=drop_features,
            start=self.start,
            end=self.end,
        )

    def window(self, start: date, end: date) -> Dataset:
        """The rows with ``start <= date < end`` as a no-copy view; raises outside the span."""
        if start < self.start or end > self.end:
            raise ValueError(
                f"corpus spans [{self.start}, {self.end}) but window [{start}, {end}) was requested"
            )
        lo = bisect.bisect_left(self._dates, start)
        hi = bisect.bisect_left(self._dates, end)
        return Dataset(
            X=cast("ColumnarRows", self._rows[lo:hi]),
            y=self._y[lo:hi],
            meta=self._meta[lo:hi],
            feature_names=self.feature_names,
            categorical_names=self.categorical_names,
        )


@dataclass(frozen=True)
class FitContext:
    """The retrain's shared fit state, assembled once per fit by ``build_fit_context``.

    Attributes:
        fams: the enabled family objects, in config order.
        indexes: their as-of indexes.
        drop: columns a fit must drop.
        corpus: the once-featurized corpus.
    """

    fams: Sequence[FeatureFamily]
    indexes: Mapping[str, FamilyIndex]
    drop: frozenset[str]
    corpus: FeaturizedCorpus


def build_fit_context(
    conn: sqlite3.Connection, cfg: _ServedRecipe, *, through: date, months: int
) -> FitContext:
    """Assemble the shared fit state: family indexes through ``through`` plus the featurized corpus.

    Args:
        months: how many trailing monthly windows the corpus must cover; the span reaches
            ``cfg.window.cap_days`` before the oldest month start so every ``fit_window``
            slice is covered.

    Every build carries every family index, so a ``FitContext`` has one shape; an exact
    persisted corpus snapshot only skips the featurize pass
    (``_featurized_corpus_incremental``'s exact hit).
    """
    drop = dropped_features([FEATURE_FAMILIES[name]() for name in cfg.families], cfg.features)
    train_ends = [through] + [start for start, _ in trailing_months(through, months)]
    span_start = (
        date.min
        if cfg.window.cap_days is None
        else min(train_ends) - timedelta(days=cfg.window.cap_days)
    )
    fams, idx = _build_indexes_overlapping_original(conn, cfg.families, through=through)
    corpus = _featurized_corpus_incremental(
        conn, fams=fams, indexes=idx, drop=drop, start=span_start, end=through
    )
    return FitContext(fams=fams, indexes=idx, drop=drop, corpus=corpus)


def _featurized_corpus_incremental(
    conn: sqlite3.Connection,
    *,
    fams: Sequence[FeatureFamily],
    indexes,
    drop: frozenset[str],
    start: date,
    end: date,
) -> FeaturizedCorpus:
    """Load the newest matching corpus snapshot, extend it through ``end``, and persist it.

    Returns:
        A corpus covering ``[start, end)``. A DB with no backing file skips the store entirely
        and gets a full ``FeaturizedCorpus`` build.
    """
    if not main_db_path(conn):
        return FeaturizedCorpus(
            conn, start=start, end=end, families=fams, indexes=indexes, drop_features=drop
        )
    key = fit_corpus_store.SnapshotKey.for_context(fams, drop=drop, start=start)
    payload, digest = fit_corpus_store.find_extendable(conn, key, end=end)
    if payload is None:
        corpus = FeaturizedCorpus(
            conn, start=start, end=end, families=fams, indexes=indexes, drop_features=drop
        )
    else:
        cached = FeaturizedCorpus._from_parts(
            matrix=payload["matrix"],
            y=payload["y"],
            meta=payload["meta"],
            feature_names=key.feature_names,
            drop_features=drop,
            start=start,
            end=date.fromisoformat(payload["end"]),
        )
        if cached.end == end:
            return cached  # exact hit: nothing to extend, already persisted
        corpus = _extend_featurized_corpus(
            conn, cached, through=end, families=fams, indexes=indexes
        )
    fit_corpus_store.store(
        key, end=end, digest=digest, matrix=corpus._matrix, y=corpus._y, meta=corpus._meta
    )
    return corpus


def _extend_featurized_corpus(
    conn: sqlite3.Connection,
    cached: FeaturizedCorpus,
    *,
    through: date,
    families: Sequence[FeatureFamily],
    indexes,
) -> FeaturizedCorpus:
    """Featurize only the maps after ``cached``'s data-end and append.

    Returns:
        A corpus row-identical to a from-scratch build over ``[cached.start, through)``:
        features are as-of each map's own date and the facts read order is (day, series,
        game_no), so splitting that stream at a date boundary concatenates back.
    """
    import numpy as np

    tail = FeaturizedCorpus(
        conn,
        start=cached.end,
        end=through,
        families=families,
        indexes=indexes,
        drop_features=cached.drop_features,
    )
    return FeaturizedCorpus._from_parts(
        matrix=np.concatenate([cached._matrix, tail._matrix]),
        y=cached._y + tail._y,
        meta=cached._meta + tail._meta,
        feature_names=cached.feature_names,
        drop_features=cached.drop_features,
        start=cached.start,
        end=through,
    )


def fit_window(
    backend: LightGBMBagBackend,
    *,
    train_end: date,
    window: WindowParams,
    drop_features: frozenset[str],
    corpus: FeaturizedCorpus,
) -> FittedModel | None:
    """Fit ``backend`` on eligible rows before ``train_end``, recency-weighted to ``train_end``.

    Args:
        corpus: answers the window by slice instead of rebuilding the matrix from the DB; its
            drop set must be exactly ``drop_features`` (``FeaturizedCorpus.narrowed`` first).
    Returns:
        The fitted model, or ``None`` when the window holds no rows.
    """
    start = date.min if window.cap_days is None else train_end - timedelta(days=window.cap_days)
    if corpus.drop_features != frozenset(drop_features):
        raise ValueError("corpus was featurized with a different drop_features set")
    ds = corpus.window(start, train_end)
    if not ds.X:
        return None
    weights = recency_weights(
        [m["date"] for m in ds.meta],
        reference=train_end,
        tau=window.tau,
        floor=window.floor,
    )
    validate_orientation_pairs(ds.meta, ds.y, weights)
    return backend.fit(
        ds.X,
        ds.y,
        sample_weight=weights,
        feature_names=ds.feature_names,
        categorical_names=ds.categorical_names,
    )


def dropped_features(families: Sequence[FeatureFamily], keep: Iterable[str]) -> frozenset[str]:
    """The emitted columns a fit must drop so it sees exactly ``keep``."""
    emitted = {fn for f in families for fn in f.feature_names}
    return frozenset(emitted - set(keep))


def month_start(label: str) -> date:
    """``"YYYY-MM"`` -> the first of that month -- a backtest window's train cutoff."""
    return date(int(label[:4]), int(label[5:7]), 1)
