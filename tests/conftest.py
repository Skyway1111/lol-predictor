"""Shared test fixtures; the suite runs under xdist, so every fixture must be worker-safe."""

import ctypes
import shutil
import sqlite3
import sys

import pytest

#: One pytest session per login session: concurrent lanes sum their torch-loaded workers
#: past the box, so a second session waits on this named mutex (TRAPS.md, tooling).
_SUITE_MUTEX = "Local\\lol-predictor-pytest"
_WAIT_TIMEOUT = 0x102
_INFINITE = 0xFFFFFFFF


def pytest_xdist_auto_num_workers(config):  # noqa: ARG001 - mirrors the pytest-xdist hookspec, which pluggy calls by argument name
    """Cap xdist by memory and CPU through the production fan-out guard."""
    from league_predictor.resources import bounded_workers

    return bounded_workers(2.5, cap=8)


def pytest_sessionstart(session):
    """Hold the machine-wide suite mutex for the controller's lifetime."""
    if sys.platform != "win32" or hasattr(session.config, "workerinput"):
        return
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateMutexW(None, 0, _SUITE_MUTEX)
    if kernel32.WaitForSingleObject(handle, 0) == _WAIT_TIMEOUT:
        sys.stderr.write(f"waiting: another pytest session holds {_SUITE_MUTEX}\n")
        kernel32.WaitForSingleObject(handle, _INFINITE)
    session.config._suite_mutex = handle


def pytest_sessionfinish(session):
    handle = getattr(session.config, "_suite_mutex", None)
    if handle:
        ctypes.windll.kernel32.ReleaseMutex(handle)
        ctypes.windll.kernel32.CloseHandle(handle)


@pytest.fixture(autouse=True)
def _isolated_original_cache(request, tmp_path, monkeypatch):
    """Isolate memory caches and redirect disk caches to the test's temporary directory."""
    from league_predictor.predraft import feature_cache

    monkeypatch.setattr(feature_cache, "_FEATURE_CACHE", feature_cache.FeatureCache())
    # ``@pytest.mark.real_original_cache`` opts out: real-DB smokes share the configured cache
    # because a from-scratch precompute takes minutes.
    if request.node.get_closest_marker("real_original_cache"):
        yield
        return
    from league_predictor import config

    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    yield


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary SQLite database for testing."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture(scope="session")
def db_template_path(tmp_path_factory):
    """Canonical schema built ONCE per session into a template file."""
    # Fresh-DB fixtures should ``shutil.copy`` this (~2ms) instead of rebuilding (~0.2s each);
    # tests exercising the bootstrap itself must still call the real functions.
    from league_predictor.db import ensure_database_ready, get_connection

    path = tmp_path_factory.mktemp("db_template") / "template.db"
    conn = get_connection(path)
    ensure_database_ready(conn)
    conn.close()  # last close checkpoints the WAL, so the file is self-contained
    return path


@pytest.fixture
def full_db(tmp_path, db_template_path):
    """Create a DB with the full canonical schema, on the session template."""
    from league_predictor.db import get_connection

    db_path = tmp_path / "test.db"
    shutil.copy(db_template_path, db_path)
    conn = get_connection(db_path)
    yield conn
    conn.close()


@pytest.fixture
def rebuild_db():
    """In-memory connection carrying the current schema, with FKs enforced as in production."""
    from league_predictor.db.schema import apply_current_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    apply_current_schema(conn)
    yield conn
    conn.close()


def register_model(conn: sqlite3.Connection, model_id: str, kind: str = "rating") -> str:
    """Insert a minimal ``models`` row so FK-constrained writes can reference it."""
    conn.execute("INSERT INTO models (model_id, kind) VALUES (?, ?)", (model_id, kind))
    return model_id
