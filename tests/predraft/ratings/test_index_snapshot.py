"""Persisted skill-filter serving-index snapshots."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from league_predictor.db import facts_prefix
from league_predictor.predraft.ratings.skill_filter import index_snapshot
from league_predictor.predraft.ratings.skill_filter.index_snapshot import (
    load_index_snapshot,
    save_index_snapshot,
    snapshot_path,
)
from league_predictor.predraft.ratings.skill_filter.serving_bag import BaggedSkillFilterIndex


class _Member:
    pass


def _index() -> BaggedSkillFilterIndex:
    index = BaggedSkillFilterIndex([_Member()])
    index.source_signature = "facts-v1"
    index.through = date(2026, 7, 11)
    return index


def test_snapshot_round_trips_only_for_exact_artifact_and_corpus(tmp_path):
    artifact = tmp_path / "rating.pt"
    artifact.write_bytes(b"rating")
    index = _index()

    save_index_snapshot(
        index, artifact_path=artifact, artifact_sha="abc", source_identity="digest-v1"
    )

    loaded = load_index_snapshot(
        artifact_path=artifact,
        artifact_sha="abc",
        source_signature="facts-v1",
        source_identity="digest-v1",
        through=date(2026, 7, 10),
    )
    assert isinstance(loaded, BaggedSkillFilterIndex)
    assert loaded.source_signature == "facts-v1"
    assert snapshot_path(artifact).name == "rating.index.pkl"


def test_snapshot_fails_closed_on_stale_identity_or_uncovered_date(tmp_path):
    artifact = tmp_path / "rating.pt"
    artifact.write_bytes(b"rating")
    save_index_snapshot(
        _index(), artifact_path=artifact, artifact_sha="abc", source_identity="digest-v1"
    )

    common = {
        "artifact_path": artifact,
        "artifact_sha": "abc",
        "source_signature": "facts-v1",
        "source_identity": "digest-v1",
    }
    assert (
        load_index_snapshot(**(common | {"artifact_sha": "other"}), through=date(2026, 7, 10))
        is None
    )
    assert (
        load_index_snapshot(**(common | {"source_signature": "changed"}), through=date(2026, 7, 10))
        is None
    )
    assert (
        load_index_snapshot(**(common | {"source_identity": "changed"}), through=date(2026, 7, 10))
        is None
    )
    assert load_index_snapshot(**common, through=date(2026, 7, 12)) is None


def test_snapshot_accepts_only_a_same_vocabulary_corpus_prefix(tmp_path):
    artifact = tmp_path / "rating.pt"
    artifact.write_bytes(b"rating")
    index = _index()
    index.source_signature = (100, "2026-07-11", "vocab")
    save_index_snapshot(
        index, artifact_path=artifact, artifact_sha="abc", source_identity="digest-v1"
    )

    common = {
        "artifact_path": artifact,
        "artifact_sha": "abc",
        "source_identity": "digest-v1",
        "through": date(2026, 7, 10),
    }
    assert load_index_snapshot(**common, source_signature=(95, "2026-07-10", "vocab")) is not None
    assert load_index_snapshot(**common, source_signature=(101, "2026-07-11", "vocab")) is None
    assert load_index_snapshot(**common, source_signature=(95, "2026-07-10", "new-vocab")) is None


def test_corrupt_snapshot_is_a_cache_miss(tmp_path):
    artifact = tmp_path / "rating.pt"
    artifact.write_bytes(b"rating")
    snapshot_path(artifact).write_bytes(b"not pickle")

    assert (
        load_index_snapshot(
            artifact_path=artifact,
            artifact_sha="abc",
            source_signature="facts-v1",
            source_identity="digest-v1",
            through=date(2026, 7, 10),
        )
        is None
    )


def test_snapshot_from_removed_module_is_a_cache_miss(tmp_path):
    """A pre-rename pickle must trigger a replay, not break live serving."""
    artifact = tmp_path / "rating.pt"
    artifact.write_bytes(b"rating")
    snapshot_path(artifact).write_bytes(
        b"cleague_predictor.predraft.ratings.learned.serving\nLearnedIndex\n."
    )

    assert (
        load_index_snapshot(
            artifact_path=artifact,
            artifact_sha="abc",
            source_signature="facts-v1",
            source_identity="digest-v1",
            through=date(2026, 7, 10),
        )
        is None
    )


def test_facts_identity_is_stable_across_an_unrelated_open_transaction(tmp_path):
    from league_predictor.predraft.ratings.skill_filter.index_snapshot import facts_identity

    path = tmp_path / "facts.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE series (series_id TEXT, league_id TEXT, team_a TEXT, team_b TEXT, "
        "scheduled_ts TEXT, best_of INTEGER)"
    )
    conn.execute("CREATE TABLE games (game_id TEXT, series_id TEXT, winner TEXT)")
    conn.execute("CREATE TABLE unrelated (value TEXT)")
    conn.execute("INSERT INTO series VALUES ('s','LCK','A','B','2026-01-01',3)")
    conn.execute("INSERT INTO games VALUES ('g','s','A')")
    conn.commit()
    through = date(2026, 2, 1)
    outside = facts_identity(conn, through=through)

    conn.execute("INSERT INTO unrelated VALUES ('pending')")
    assert conn.in_transaction
    inside = facts_identity(conn, through=through)
    conn.rollback()
    conn.close()

    assert outside == inside
    assert outside.startswith("facts:")


def test_file_backed_facts_identity_refuses_a_slow_digest_fallback(tmp_path, monkeypatch):
    path = tmp_path / "facts.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE series (series_id TEXT)")
    conn.commit()
    monkeypatch.setattr(
        facts_prefix,
        "native_prefix_digests",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("extension unavailable")),
    )

    try:
        with pytest.raises(RuntimeError, match="extension unavailable"):
            index_snapshot.facts_identity(conn, through=date(2026, 2, 1))
    finally:
        conn.close()


def test_rating_identity_declares_rating_inputs_only(tmp_path):
    """Draft products cannot invalidate a rating replay, while rating facts must."""
    from league_predictor.predraft.ratings.skill_filter.index_snapshot import facts_identity

    path = tmp_path / "rating-inputs.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE series (
            series_id TEXT, league_id TEXT, team_a TEXT, team_b TEXT,
            scheduled_ts TEXT, best_of INTEGER
        );
        CREATE TABLE games (game_id TEXT, series_id TEXT, winner TEXT);
        CREATE TABLE game_players (game_id TEXT, player_id TEXT);
        CREATE TABLE game_team_stats (game_id TEXT, team_id TEXT, gold REAL);
        CREATE TABLE draft_scores (game_id TEXT, model_id TEXT, score REAL);
        INSERT INTO series VALUES ('s', 'LCK', 'A', 'B', '2026-01-01', 3);
        INSERT INTO games VALUES ('g', 's', 'A');
        INSERT INTO game_players VALUES ('g', 'p');
        INSERT INTO game_team_stats VALUES ('g', 'A', 100.0);
        """
    )
    conn.commit()
    through = date(2026, 2, 1)
    original = facts_identity(conn, through=through)

    conn.execute("INSERT INTO draft_scores VALUES ('g', 'draft-v2', 0.75)")
    conn.commit()
    assert facts_identity(conn, through=through) == original

    conn.execute("UPDATE game_team_stats SET gold = 101.0")
    conn.commit()
    assert facts_identity(conn, through=through) == original

    conn.execute("UPDATE game_players SET player_id = 'p2'")
    conn.commit()
    assert facts_identity(conn, through=through) != original
    conn.close()
