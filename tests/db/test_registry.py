"""Tests for the ``models`` registry table and the runtime code fingerprint."""

from __future__ import annotations

import subprocess

import pytest

from league_predictor.db.registry import (
    code_fingerprint,
    ensure_registered,
    get_model,
    prune_orphaned_models,
)
from league_predictor.db.rows import ModelRow
from league_predictor.lifecycle.draft import _PRUNE_REFS as _DRAFT_REFS


def test_ensure_registered_inserts_and_get_model_reads_back(rebuild_db):
    row = ModelRow(
        model_id="rating:abc123456789",
        kind="rating",
        artifact_sha="abc123456789deadbeef",
    )
    ensure_registered(rebuild_db, row)

    fetched = get_model(rebuild_db, "rating:abc123456789")
    assert fetched is not None
    assert fetched.model_id == "rating:abc123456789"
    assert fetched.kind == "rating"
    assert fetched.artifact_sha == "abc123456789deadbeef"


def test_get_model_returns_none_for_unknown_id(rebuild_db):
    assert get_model(rebuild_db, "rating:nope") is None


def test_ensure_registered_is_idempotent_for_same_sha(rebuild_db):
    row = ModelRow(model_id="served:v1", kind="served", artifact_sha="sha-1")
    ensure_registered(rebuild_db, row)
    ensure_registered(rebuild_db, row)  # no raise, no duplicate

    count = rebuild_db.execute("SELECT COUNT(*) FROM models").fetchone()[0]
    assert count == 1


def test_ensure_registered_raises_on_sha_collision(rebuild_db):
    ensure_registered(rebuild_db, ModelRow("served:v1", "served", artifact_sha="sha-1"))
    with pytest.raises(ValueError, match="refusing to re-register"):
        ensure_registered(rebuild_db, ModelRow("served:v1", "served", artifact_sha="sha-2"))


# --- prune_orphaned_models — draft -------------------------------------------------------


def test_the_fusion_reference_is_declared_in_production():
    """The pin below guards draft's REAL reference set, not a test-local copy."""
    assert ("postdraft_scores", "draft_model_id") in _DRAFT_REFS


def _seed_scoreable_game(conn) -> None:
    """A game + team + player the score tables can FK-reference."""
    for t in ("A", "B"):
        conn.execute("INSERT INTO teams (team_id) VALUES (?)", (t,))
    conn.execute("INSERT INTO players (player_id) VALUES ('p1')")
    conn.execute(
        "INSERT INTO series (series_id, team_a, team_b, status)"
        " VALUES ('s1', 'A', 'B', 'completed')"
    )
    conn.execute("INSERT INTO games (game_id, series_id, game_no) VALUES ('g1', 's1', 1)")


def _register_draft(conn, model_id) -> None:
    conn.execute("INSERT INTO models (model_id, kind) VALUES (?, 'draft')", (model_id,))


def test_prune_drops_only_unreferenced_draft_models(rebuild_db):
    """A draft: model referenced by neither draft_scores nor pick_scores is dropped."""
    _seed_scoreable_game(rebuild_db)
    _register_draft(rebuild_db, "draft:has_draft_score")
    _register_draft(rebuild_db, "draft:has_pick_score")
    _register_draft(rebuild_db, "draft:orphan")
    rebuild_db.execute(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red) "
        "VALUES ('g1', 'draft:has_draft_score', 0.6, 'B', 'C')"
    )
    rebuild_db.execute(
        "INSERT INTO pick_scores (game_id, side, role, model_id, player_id, percentile) "
        "VALUES ('g1', 'blue', 'mid', 'draft:has_pick_score', 'p1', 0.5)"
    )
    rebuild_db.commit()

    removed = prune_orphaned_models(rebuild_db, "draft", _DRAFT_REFS)
    rebuild_db.commit()

    assert removed == 1
    survivors = {
        r[0]
        for r in rebuild_db.execute("SELECT model_id FROM models WHERE model_id LIKE 'draft:%'")
    }
    assert survivors == {"draft:has_draft_score", "draft:has_pick_score"}


def test_prune_removes_the_orphan_a_rescore_leaves_behind(rebuild_db):
    """Once a rescore moves every score row onto the new id, the old registry row is pruned."""
    _seed_scoreable_game(rebuild_db)
    _register_draft(rebuild_db, "draft:old")
    _register_draft(rebuild_db, "draft:new")
    # The rescore's end state: every score row now carries draft:new.
    rebuild_db.execute(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red) "
        "VALUES ('g1', 'draft:new', 0.6, 'B', 'C')"
    )
    rebuild_db.commit()

    removed = prune_orphaned_models(rebuild_db, "draft", _DRAFT_REFS)
    rebuild_db.commit()

    assert removed == 1
    assert get_model(rebuild_db, "draft:old") is None
    assert get_model(rebuild_db, "draft:new") is not None


def test_prune_never_touches_non_draft_models(rebuild_db):
    """Only ``draft:`` ids are eligible; an orphaned served/rating row survives as provenance."""
    rebuild_db.execute(
        "INSERT INTO models (model_id, kind) VALUES ('served:v1_deadbeef', 'served')"
    )
    rebuild_db.execute("INSERT INTO models (model_id, kind) VALUES ('rating:abc123', 'rating')")
    _register_draft(rebuild_db, "draft:orphan")
    rebuild_db.commit()

    removed = prune_orphaned_models(rebuild_db, "draft", _DRAFT_REFS)
    rebuild_db.commit()

    assert removed == 1
    assert get_model(rebuild_db, "served:v1_deadbeef") is not None
    assert get_model(rebuild_db, "rating:abc123") is not None
    assert get_model(rebuild_db, "draft:orphan") is None


def test_a_postdraft_row_blocks_the_draft_models_prune(rebuild_db):
    """A ``postdraft_scores.draft_model_id`` reference keeps the draft models row alive."""
    # A fusion reference keeps its draft model alive until the fusion is rebuilt.
    _seed_scoreable_game(rebuild_db)
    _register_draft(rebuild_db, "draft:old")
    _register_draft(rebuild_db, "draft:new")
    rebuild_db.execute("INSERT INTO models (model_id, kind) VALUES ('served:v1_x', 'served')")
    # The rescore moved every draft_scores row to draft:new, but the fusion is not yet rebuilt.
    rebuild_db.execute(
        "INSERT INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red) "
        "VALUES ('g1', 'draft:new', 0.6, 'B', 'C')"
    )
    rebuild_db.execute(
        "INSERT INTO postdraft_scores (game_id, p_blue, served_model_id, draft_model_id) "
        "VALUES ('g1', 0.6, 'served:v1_x', 'draft:old')"
    )
    rebuild_db.commit()

    assert prune_orphaned_models(rebuild_db, "draft", _DRAFT_REFS) == 0
    assert get_model(rebuild_db, "draft:old") is not None

    # Once the fusion is rebuilt onto the new id, the old row is disposable residue.
    rebuild_db.execute("UPDATE postdraft_scores SET draft_model_id = 'draft:new'")
    assert prune_orphaned_models(rebuild_db, "draft", _DRAFT_REFS) == 1
    assert get_model(rebuild_db, "draft:old") is None


# --- prune_orphaned_models — impact ---------------------------------------------------------


def test_prune_impact_drops_only_unreferenced_impact_models(rebuild_db):
    """A impact: model with no player_impact_scores row is dropped; a referenced one survives."""
    rebuild_db.execute("INSERT INTO players (player_id) VALUES ('p1')")
    rebuild_db.execute("INSERT INTO models (model_id, kind) VALUES ('impact:referenced', 'impact')")
    rebuild_db.execute("INSERT INTO models (model_id, kind) VALUES ('impact:orphan', 'impact')")
    rebuild_db.execute(
        "INSERT INTO player_impact_scores (player_id, model_id, asof_date, role, impact) "
        "VALUES ('p1', 'impact:referenced', '2026-01-01', 'mid', 1.5)"
    )
    rebuild_db.commit()

    removed = prune_orphaned_models(rebuild_db, "impact", (("player_impact_scores", "model_id"),))
    rebuild_db.commit()

    assert removed == 1
    assert get_model(rebuild_db, "impact:referenced") is not None
    assert get_model(rebuild_db, "impact:orphan") is None


def test_prune_impact_never_touches_non_impact_models(rebuild_db):
    """Only ``impact:`` ids are eligible — an orphaned draft/served row must survive."""
    rebuild_db.execute("INSERT INTO models (model_id, kind) VALUES ('draft:xyz', 'draft')")
    rebuild_db.execute("INSERT INTO models (model_id, kind) VALUES ('impact:orphan', 'impact')")
    rebuild_db.commit()

    removed = prune_orphaned_models(rebuild_db, "impact", (("player_impact_scores", "model_id"),))
    rebuild_db.commit()

    assert removed == 1
    assert get_model(rebuild_db, "draft:xyz") is not None
    assert get_model(rebuild_db, "impact:orphan") is None


# --- code_fingerprint: the one runtime source-tree hash ----------------------------------


def _git(root, *args):
    subprocess.run(  # noqa: S603 - argv is a fixed list, no shell
        ["git", "-C", str(root), *args],  # noqa: S607 - git resolves from PATH by design
        check=True,
        capture_output=True,
    )


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    """A throwaway repo with ``_repo_root`` pointed at it; yields the package dir it covers."""
    import league_predictor.db.registry as registry_module

    root = tmp_path / "repo"
    pkg = root / "src" / "league_predictor"
    (pkg / "sub").mkdir(parents=True)
    (pkg / "a.py").write_text("A = 1\n", encoding="utf-8")
    (pkg / "sub" / "b.py").write_text("B = 2\n", encoding="utf-8")
    (root / "README.md").write_text("outside the runtime tree\n", encoding="utf-8")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    monkeypatch.setattr(registry_module, "_repo_root", lambda: root)
    return pkg


@pytest.mark.usefixtures("fake_repo")
def test_code_fingerprint_is_stable_hex():
    a, b = code_fingerprint(), code_fingerprint()
    assert a == b
    assert len(a) == 64
    int(a, 16)  # 64-hex sha256, or this raises


def test_code_fingerprint_follows_content_not_staging_state(fake_repo):
    """The fingerprint keys on file content, never on stat or git index state."""
    clean = code_fingerprint()
    (fake_repo / "a.py").write_text("A = 2\n", encoding="utf-8")
    dirty = code_fingerprint()
    assert dirty != clean
    _git(fake_repo.parents[1], "add", "-A")
    assert code_fingerprint() == dirty


def test_code_fingerprint_ignores_a_stat_only_touch(fake_repo):
    import os
    import time

    before = code_fingerprint()
    stamp = time.time() + 60
    os.utime(fake_repo / "a.py", (stamp, stamp))
    assert code_fingerprint() == before


def test_code_fingerprint_include_dirty_false_reads_the_index_alone(fake_repo):
    clean = code_fingerprint(include_dirty=False)
    (fake_repo / "a.py").write_text("A = 3\n", encoding="utf-8")
    assert code_fingerprint(include_dirty=False) == clean
    assert code_fingerprint(include_dirty=True) != clean


def test_code_fingerprint_counts_untracked_runtime_files_but_not_caches(fake_repo):
    base = code_fingerprint()
    cache = fake_repo / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-314.pyc").write_bytes(b"\x00bytecode")
    (fake_repo / "sub" / "b.pyc").write_bytes(b"\x00stray")
    assert code_fingerprint() == base  # bytecode caches never affect the imported package
    (fake_repo / "new_module.py").write_text("NEW = True\n", encoding="utf-8")
    assert code_fingerprint() != base


def test_code_fingerprint_sees_a_deleted_tracked_file(fake_repo):
    base = code_fingerprint()
    (fake_repo / "sub" / "b.py").unlink()
    assert code_fingerprint() != base


def test_code_fingerprint_ignores_files_outside_the_runtime_tree(fake_repo):
    root = fake_repo.parents[1]
    base = code_fingerprint()
    (root / "README.md").write_text("changed\n", encoding="utf-8")
    (root / "scratch.py").write_text("x = 1\n", encoding="utf-8")
    assert code_fingerprint() == base
