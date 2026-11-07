"""Tests the walk-forward seam's once-featurized corpus answers every window like a fresh build."""

# file-length-ok: one module's contract mirror, split adds cross-file hops

from __future__ import annotations

from datetime import date

import pytest

from league_predictor import config
from league_predictor.predraft.pipeline.walk_forward import (
    FeaturizedCorpus,
    WindowParams,
    build_indexes,
    fit_window,
)
from tests.predraft.pipeline._dataset_reference import build_matrix


def _built(conn, *, start: date, end: date, drop: frozenset[str] = frozenset()):
    fams, idx = build_indexes(conn, ["recency"], through=end)
    corpus = FeaturizedCorpus(
        conn, start=start, end=end, families=fams, indexes=idx, drop_features=drop
    )
    return fams, idx, corpus


def test_corpus_window_equals_fresh_build_matrix(skill_db):
    """Slicing the corpus reproduces a fresh window build row-for-row, in the same order."""
    conn = skill_db(start=date(2026, 1, 1), end=date(2026, 4, 30), per_day=3)
    fams, idx, corpus = _built(conn, start=date(2026, 1, 1), end=date(2026, 5, 1))

    lo, hi = date(2026, 2, 1), date(2026, 4, 1)
    sliced = corpus.window(lo, hi)
    fresh = build_matrix(conn, start=lo, end=hi, families=fams, indexes=idx)

    assert list(sliced.X) == fresh.X
    assert sliced.y == fresh.y
    assert sliced.meta == fresh.meta
    assert sliced.feature_names == fresh.feature_names
    assert sliced.categorical_names == fresh.categorical_names
    assert len(sliced.X) > 0
    assert sliced.X.matrix.base is corpus._matrix  # a matrix VIEW, never a per-window copy


def test_chunked_featurization_is_byte_identical_to_all_at_once(skill_db, monkeypatch):
    """Chunk-streamed featurization is byte-identical to the all-at-once build."""
    import numpy as np

    from league_predictor.predraft.pipeline import walk_forward
    from league_predictor.predraft.pipeline.lightgbm_bag import to_frame

    conn = skill_db(start=date(2026, 1, 1), end=date(2026, 3, 31), per_day=3)
    lo, hi = date(2026, 1, 1), date(2026, 4, 1)
    fams, idx = build_indexes(conn, ["recency"], through=hi)

    # The all-at-once reference: one build_features_batch (via build_matrix), one to_frame
    # over every dict row, then the stable date sort.
    ds = build_matrix(conn, start=lo, end=hi, families=fams, indexes=idx)
    order = sorted(range(len(ds.meta)), key=lambda i: ds.meta[i]["date"])
    reference = to_frame(ds.X, feature_names=ds.feature_names, categorical_names=[]).to_numpy(
        dtype=np.float64
    )[order]

    monkeypatch.setattr(walk_forward, "_FEATURIZE_CHUNK_ROWS", 7)  # force many chunks
    corpus = FeaturizedCorpus(
        conn, start=lo, end=hi, families=fams, indexes=idx, drop_features=frozenset()
    )

    assert corpus._matrix.shape == reference.shape
    assert len(corpus._y) > 0
    assert corpus._matrix.dtype == reference.dtype == np.float64
    assert corpus._matrix.tobytes() == reference.tobytes()
    assert corpus._y == [ds.y[i] for i in order]
    assert corpus._meta == [ds.meta[i] for i in order]
    assert corpus.feature_names == ds.feature_names


# --------------------------------------------------------------------------- #
# build_fit_context -- the overlapped original-family build equals the serial one


def _seed_roster_db(path):
    """A file-backed facts DB of training-eligible maps carrying full 5-man rosters."""
    import itertools
    import sqlite3

    from league_predictor.db.schema import apply_current_schema
    from tests.predraft.conftest import ensure_event_type_column

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    ensure_event_type_column(conn)
    teams = ["t_aaa", "t_bbb", "t_ccc"]
    roles = ("top", "jng", "mid", "bot", "sup")
    n, day = 0, 1
    for _ in range(2):
        for a, b in itertools.combinations(teams, 2):
            mid = f"m{n}"
            winner = a if n % 2 == 0 else b
            n += 1
            ta, tb = sorted((a, b))
            conn.execute(
                "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
                "best_of, status) VALUES (?, 'LCK', ?, ?, ?, 1, 'completed')",
                (f"ser_{mid}", ta, tb, f"2026-03-{day:02d}T10:00:00+00:00"),
            )
            conn.execute(
                "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
                "VALUES (?, ?, 1, ?, 1800)",
                (mid, f"ser_{mid}", winner),
            )
            day += 1
            for tid in (a, b):
                for r in roles:
                    conn.execute(
                        "INSERT INTO game_players (game_id, player_id, team_id, role, "
                        "kills, deaths, assists) VALUES (?, ?, ?, ?, 5, 5, 10)",
                        (mid, f"{tid}_{r}", tid, r),
                    )
    conn.commit()
    return conn


def test_build_fit_context_overlapped_original_equals_serial(tmp_path, monkeypatch):
    """The thread-overlapped original-family build yields the serial build's FitContext exactly."""
    import threading
    from types import SimpleNamespace

    from league_predictor.predraft.features.families import original
    from league_predictor.predraft.pipeline import walk_forward

    conn = _seed_roster_db(tmp_path / "facts.db")
    cfg = SimpleNamespace(
        families=["original", "recency"],
        features=["recency_winrate20_diff"],
        window=WindowParams(tau=1e9, floor=0.0, cap_days=120),
    )
    through = date(2026, 4, 1)

    built_on: list[int] = []
    real_build = original.OriginalFeaturesFamily.build

    def spy_build(self, conn_, *, through, query_dates):
        built_on.append(threading.get_ident())
        return real_build(self, conn_, through=through, query_dates=query_dates)

    monkeypatch.setattr(original.OriginalFeaturesFamily, "build", spy_build)

    # Distinct store dirs per phase: the serial reference must FEATURIZE, not load the blobs
    # the overlapped build just persisted (that would compare it with itself).

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache_overlap")
    overlapped = walk_forward.build_fit_context(conn, cfg, through=through, months=2)
    assert built_on
    assert built_on[-1] != threading.get_ident()  # really built on the worker

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache_serial")
    monkeypatch.setattr(walk_forward, "main_db_path", lambda _conn: "")  # force the serial path
    serial = walk_forward.build_fit_context(conn, cfg, through=through, months=2)
    assert built_on[-1] == threading.get_ident()  # the serial path stayed on-thread

    assert [f.name for f in overlapped.fams] == [f.name for f in serial.fams]
    assert list(overlapped.indexes) == list(serial.indexes)  # deterministic family order
    assert overlapped.indexes["original"]._rows == serial.indexes["original"]._rows
    assert (
        overlapped.indexes["original"]._matrix.tobytes()
        == serial.indexes["original"]._matrix.tobytes()
    )
    assert overlapped.indexes["original"].feature_names == serial.indexes["original"].feature_names
    assert overlapped.indexes["original"]._conn is conn  # fallback rebound to caller's conn
    import numpy as np

    assert overlapped.drop == serial.drop
    assert np.array_equal(overlapped.corpus._matrix, serial.corpus._matrix, equal_nan=True)
    assert overlapped.corpus._y == serial.corpus._y
    assert overlapped.corpus._meta == serial.corpus._meta
    assert overlapped.corpus._matrix.shape[0] > 0
    conn.close()


def test_narrowed_corpus_is_the_column_subset_and_fits_the_wider_drop_set(skill_db):
    """A shared corpus answers an arm whose columns are a subset via ``narrowed`` (the A/B read)."""
    conn = skill_db(start=date(2026, 1, 1), end=date(2026, 2, 28), per_day=2)
    fams, idx, corpus = _built(conn, start=date(2026, 1, 1), end=date(2026, 3, 1))
    drop = frozenset({"recency_ema_diff"})

    narrowed = corpus.narrowed(drop)
    direct = FeaturizedCorpus(
        conn, start=corpus.start, end=corpus.end, families=fams, indexes=idx, drop_features=drop
    )
    assert narrowed.feature_names == direct.feature_names
    assert narrowed._matrix.tobytes() == direct._matrix.tobytes()
    assert narrowed._y == direct._y
    assert narrowed._meta == direct._meta
    assert corpus.narrowed(frozenset()) is corpus  # its own drop set: no copy
    with pytest.raises(ValueError, match="different drop_features"):
        narrowed.narrowed(frozenset())  # a narrower drop set asks for a column it lacks
    # fit_window takes the narrowed corpus under the wider drop set without a rebuild.
    fitted = fit_window(
        _one_seed_backend(),
        train_end=date(2026, 2, 15),
        window=WindowParams(tau=1e9, floor=0.0, cap_days=30),
        drop_features=drop,
        corpus=narrowed,
    )
    assert fitted is not None


def _one_seed_backend():
    from league_predictor.predraft.pipeline.lightgbm_bag import LightGBMBagBackend, LightGBMParams

    return LightGBMBagBackend(
        LightGBMParams(n_estimators=5, num_leaves=3, min_child_samples=2), seeds=(1,)
    )


def test_corpus_refuses_uncovered_window_and_foreign_drop_set(skill_db):
    conn = skill_db(start=date(2026, 1, 1), end=date(2026, 2, 28), per_day=2)
    _fams, _idx, corpus = _built(conn, start=date(2026, 1, 1), end=date(2026, 3, 1))

    with pytest.raises(ValueError, match="corpus spans"):
        corpus.window(date(2025, 12, 1), date(2026, 2, 1))  # reaches before the span

    with pytest.raises(ValueError, match="drop_features"):
        fit_window(
            None,
            train_end=date(2026, 2, 1),
            window=WindowParams(tau=1e9, floor=0.0, cap_days=30),
            drop_features=frozenset({"recency_winrate20_diff"}),
            corpus=corpus,
        )


# --------------------------------------------------------------------------- #
# build_fit_context -- persisted corpus snapshots (served incremental corpora)


def _file_skill_db(path, skill_db, *, start: date, end: date, per_day: int = 3):
    """A file-backed skill DB: seed in memory, ``.backup`` to disk, return the file connection."""
    import sqlite3

    mem = skill_db(start=start, end=end, per_day=per_day)
    disk = sqlite3.connect(str(path))
    mem.backup(disk)
    mem.close()
    disk.row_factory = sqlite3.Row
    return disk


def _snapshot_cfg(features=("recency_winrate20_diff",)):
    """A recency-only served config; the uncapped window puts the corpus start at date.min."""
    from types import SimpleNamespace

    return SimpleNamespace(
        families=["recency"],
        features=list(features),
        window=WindowParams(tau=1e9, floor=0.0, cap_days=None),
    )


@pytest.fixture
def featurized_spans(monkeypatch):
    """Record every (start, end) span ``FeaturizedCorpus`` featurizes."""
    from league_predictor.predraft.pipeline import walk_forward

    spans: list[tuple[date, date]] = []
    real = walk_forward.window_queries

    def spy(conn, start, end):
        spans.append((start, end))
        return real(conn, start, end)

    monkeypatch.setattr(walk_forward, "window_queries", spy)
    return spans


def _corpus_identical(a, b) -> None:
    """Assert two corpora are row-identical: same matrix bytes, labels, meta, columns, span."""
    assert a._matrix.shape == b._matrix.shape
    assert a._matrix.dtype == b._matrix.dtype
    assert a._matrix.tobytes() == b._matrix.tobytes()
    assert a._y == b._y
    assert a._meta == b._meta
    assert a.feature_names == b.feature_names
    assert a.drop_features == b.drop_features
    assert (a.start, a.end) == (b.start, b.end)


def test_build_fit_context_extends_the_persisted_corpus_snapshot(
    tmp_path, skill_db, featurized_spans, monkeypatch
):
    """A later ``through`` featurizes only the tail past the snapshot; an unchanged repeat loads."""
    from league_predictor.predraft.pipeline import walk_forward

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "fit-corpus")
    conn = _file_skill_db(
        tmp_path / "facts.db", skill_db, start=date(2026, 1, 1), end=date(2026, 2, 28)
    )
    cfg, t1, t2 = _snapshot_cfg(), date(2026, 3, 1), date(2026, 3, 21)

    ctx1 = walk_forward.build_fit_context(conn, cfg, through=t1, months=2)
    assert featurized_spans == [(date.min, t1)]  # first build: the full span
    assert len(list((tmp_path / "fit-corpus" / "cache").glob("fit_corpus_*.pkl"))) == 1

    from tests.predraft.conftest import _DEFAULT_SKILL, _generate

    _generate(
        conn,
        start=date(2026, 3, 1),
        end=date(2026, 3, 15),
        seed=7,
        skill=dict(_DEFAULT_SKILL),
        per_day=3,
    )
    featurized_spans.clear()
    ctx2 = walk_forward.build_fit_context(conn, cfg, through=t2, months=2)
    assert featurized_spans == [(t1, t2)]  # ONLY the tail after the snapshot's data-end
    assert len(ctx2.corpus._y) > len(ctx1.corpus._y)  # the tail really landed

    # Parity: row-identical to a from-scratch featurization over the same span.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "fresh")
    featurized_spans.clear()
    fresh = walk_forward.build_fit_context(conn, cfg, through=t2, months=2)
    assert featurized_spans == [(date.min, t2)]  # a genuine from-scratch reference
    _corpus_identical(ctx2.corpus, fresh.corpus)

    # Exact hit: an unchanged repeat at t2 loads the snapshot and featurizes nothing.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "fit-corpus")
    featurized_spans.clear()
    again = walk_forward.build_fit_context(conn, cfg, through=t2, months=2)
    assert featurized_spans == []
    _corpus_identical(again.corpus, fresh.corpus)
    conn.close()


def test_exact_hit_builds_the_same_indexes_as_a_fresh_context(
    tmp_path, skill_db, featurized_spans, monkeypatch
):
    """An exact snapshot hit skips only the featurize pass; the context shape never changes."""
    from league_predictor.predraft.pipeline import walk_forward

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "fit-corpus")
    conn = _file_skill_db(
        tmp_path / "facts.db", skill_db, start=date(2026, 1, 1), end=date(2026, 2, 28)
    )
    cfg, through = _snapshot_cfg(), date(2026, 3, 1)

    fresh = walk_forward.build_fit_context(conn, cfg, through=through, months=2)
    featurized_spans.clear()
    again = walk_forward.build_fit_context(conn, cfg, through=through, months=2)
    conn.close()

    assert featurized_spans == []  # the corpus loaded straight from the snapshot
    assert set(again.indexes) == set(fresh.indexes) == {"recency"}
    _corpus_identical(again.corpus, fresh.corpus)


def test_historical_correction_invalidates_the_corpus_snapshot(
    tmp_path, skill_db, featurized_spans, monkeypatch
):
    """A row-count-preserving history correction still invalidates the snapshot: full rebuild."""
    from league_predictor.predraft.pipeline import walk_forward

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "fit-corpus")
    conn = _file_skill_db(
        tmp_path / "facts.db", skill_db, start=date(2026, 1, 1), end=date(2026, 2, 28)
    )
    cfg, t1, t2 = _snapshot_cfg(), date(2026, 3, 1), date(2026, 3, 2)
    walk_forward.build_fit_context(conn, cfg, through=t1, months=2)

    # Count-preserving correction: flip one historical game's winner.
    gid, winner, team_a, team_b = conn.execute(
        "SELECT g.game_id, g.winner, s.team_a, s.team_b FROM games g "
        "JOIN series s ON s.series_id = g.series_id ORDER BY s.scheduled_ts LIMIT 1"
    ).fetchone()
    flipped = team_b if winner == team_a else team_a
    conn.execute("UPDATE games SET winner = ? WHERE game_id = ?", (flipped, gid))
    conn.commit()

    featurized_spans.clear()
    ctx = walk_forward.build_fit_context(conn, cfg, through=t2, months=2)
    assert featurized_spans == [(date.min, t2)]  # full rebuild, never an append onto stale rows

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "fresh")
    fresh = walk_forward.build_fit_context(conn, cfg, through=t2, months=2)
    _corpus_identical(ctx.corpus, fresh.corpus)
    conn.close()


def test_schema_change_misses_the_snapshot_and_memory_dbs_skip_the_store(
    tmp_path, skill_db, featurized_spans, monkeypatch
):
    """A changed column set mints a new snapshot group; an in-memory DB bypasses the store."""
    from league_predictor.predraft.pipeline import walk_forward

    store_dir = tmp_path / "fit-corpus"
    monkeypatch.setattr(config, "DATA_DIR", store_dir)
    conn = _file_skill_db(
        tmp_path / "facts.db", skill_db, start=date(2026, 1, 1), end=date(2026, 1, 31)
    )
    t1 = date(2026, 2, 1)
    walk_forward.build_fit_context(conn, _snapshot_cfg(), through=t1, months=1)

    featurized_spans.clear()
    wider = _snapshot_cfg(features=("recency_winrate20_diff", "recency_ema_diff"))
    walk_forward.build_fit_context(conn, wider, through=t1, months=1)
    assert featurized_spans == [(date.min, t1)]  # different columns: full featurization
    assert len(list((store_dir / "cache").glob("fit_corpus_*.pkl"))) == 2  # distinct groups coexist
    conn.close()

    mem = skill_db(start=date(2026, 1, 1), end=date(2026, 1, 31), per_day=2)
    before = sorted(p.name for p in (store_dir / "cache").glob("fit_corpus_*.pkl"))
    featurized_spans.clear()
    ctx = walk_forward.build_fit_context(mem, _snapshot_cfg(), through=t1, months=1)
    assert featurized_spans == [(date.min, t1)]
    assert len(ctx.corpus._y) > 0
    assert (
        sorted(p.name for p in (store_dir / "cache").glob("fit_corpus_*.pkl")) == before
    )  # nothing stored
    mem.close()


def test_fit_corpus_store_dir_follows_config(tmp_path, monkeypatch):
    from league_predictor.predraft.pipeline import fit_corpus_store

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    assert fit_corpus_store._cache_dir() == tmp_path / "data" / "cache"
