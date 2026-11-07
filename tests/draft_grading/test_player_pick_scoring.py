"""The schema-independent pieces of the per-pick scorer: classifier and role pools."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import cast

import pytest

from league_predictor.draft_grading import player_pick_scoring
from league_predictor.draft_grading.grader.features import build_role_pools
from league_predictor.draft_grading.player_pick_scoring import (
    PickScope,
    _classify_blind_or_counter,
    _SeqStep,
)


@dataclass(frozen=True)
class _Pick:
    """Local stand-in for a draft_actions pick step; the classifier reads step/side/role."""

    step: int
    side: str
    role: str


# ---- _classify_blind_or_counter (pure) ---------------------------------


def test_blind_classification_matches_pick_sequence() -> None:
    """A blue top picked at step 1 with red top at step 4 → blind. Reversed → counter."""
    seq = cast(
        "list[_SeqStep]",
        [
            _Pick(step=1, side="blue", role="top"),
            _Pick(step=2, side="red", role="top"),
            _Pick(step=3, side="red", role="mid"),
            _Pick(step=4, side="blue", role="mid"),
        ],
    )
    is_blind = _classify_blind_or_counter(seq, side="blue", role="top")
    assert is_blind is True
    # Red top picked second → counter (focal step 2 > blue top step 1).
    is_blind = _classify_blind_or_counter(seq, side="red", role="top")
    assert is_blind is False


def test_blind_classification_handles_same_phase_role_pick_correctly() -> None:
    """With both teams picking a role in adjacent steps, the later picker is the counter."""
    seq = cast(
        "list[_SeqStep]",
        [
            _Pick(step=3, side="blue", role="mid"),
            _Pick(step=4, side="red", role="mid"),
        ],
    )
    is_blind_blue = _classify_blind_or_counter(seq, side="blue", role="mid")
    is_blind_red = _classify_blind_or_counter(seq, side="red", role="mid")
    assert is_blind_blue is True  # picks first
    assert is_blind_red is False  # responds


def test_classify_raises_when_focal_missing() -> None:
    """If the focal slot isn't in the sequence at all, raise rather than guess."""
    seq = cast("list[_SeqStep]", [_Pick(step=1, side="blue", role="top")])
    with pytest.raises(ValueError, match="not in sequence"):
        _classify_blind_or_counter(seq, side="red", role="sup")


def test_classify_treats_missing_enemy_as_blind() -> None:
    """An enemy that never picks the focal role classifies as blind."""
    seq = cast("list[_SeqStep]", [_Pick(step=1, side="blue", role="top")])
    is_blind = _classify_blind_or_counter(seq, side="blue", role="top")
    assert is_blind is True


# ---- build_role_pools --------------------------------------------------


def _seed_game_players() -> sqlite3.Connection:
    """In-memory DB with the one table build_role_pools reads."""
    db = sqlite3.connect(":memory:")
    db.executescript(
        """
        CREATE TABLE game_players (
            game_id TEXT NOT NULL,
            player_id TEXT NOT NULL,
            champion_id TEXT,
            role TEXT
        );
        """
    )
    return db


def test_build_role_pools_respects_min_games() -> None:
    """Champions below ``min_role_games`` at a role are excluded from that role's pool."""
    db = _seed_game_players()
    for i in range(20):
        db.execute(
            "INSERT INTO game_players (game_id, player_id, champion_id, role) "
            "VALUES (?, 'p', 'Aatrox', 'top')",
            (f"m{i}",),
        )
    db.execute(
        "INSERT INTO game_players (game_id, player_id, champion_id, role) "
        "VALUES ('mY', 'p', 'Yuumi', 'top')",
    )
    db.commit()
    pools = build_role_pools(db, min_role_games=15)
    assert "Aatrox" in pools["top"]
    assert "Yuumi" not in pools["top"]


def test_build_role_pools_separates_roles() -> None:
    """A champ appears in a role's pool only where it meets the games floor at that role."""
    db = _seed_game_players()
    for i in range(20):
        db.execute(
            "INSERT INTO game_players (game_id, player_id, champion_id, role) "
            "VALUES (?, 'p', 'Aatrox', 'top')",
            (f"t{i}",),
        )
    for i in range(5):
        db.execute(
            "INSERT INTO game_players (game_id, player_id, champion_id, role) "
            "VALUES (?, 'p', 'Aatrox', 'mid')",
            (f"m{i}",),
        )
    db.commit()
    pools = build_role_pools(db, min_role_games=15)
    assert "Aatrox" in pools["top"]
    assert "Aatrox" not in pools["mid"]


def test_build_role_pools_can_bound_to_most_played_candidates() -> None:
    db = _seed_game_players()
    for champion, count in (("A", 30), ("B", 20), ("C", 10)):
        for i in range(count):
            db.execute(
                "INSERT INTO game_players (game_id, player_id, champion_id, role) "
                "VALUES (?, 'p', ?, 'top')",
                (f"{champion}{i}", champion),
            )
    db.commit()

    pools = build_role_pools(db, min_role_games=1, max_champions=2)

    assert pools["top"] == {"A", "B"}


def test_parallel_worker_loads_through_the_dispatching_loader(tmp_path, monkeypatch):
    """The worker routes through live.load_grader, so a format-3 family loads as a blend (D6)."""
    from pathlib import Path

    from league_predictor.db.schema import apply_current_schema
    from tests.conftest import register_model
    from tests.draft_grading.conftest import (
        ROLES as CONF_ROLES,
    )
    from tests.draft_grading.conftest import MockPredictor, seed_game
    from tests.draft_grading.conftest import (
        role_pools as conf_role_pools,
    )

    db_path = tmp_path / "facts.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    conn.executemany("INSERT INTO teams (team_id) VALUES (?)", [("T_A",), ("T_B",)])
    players = [f"bp_{r}" for r in CONF_ROLES] + [f"rp_{r}" for r in CONF_ROLES]
    conn.executemany("INSERT INTO players (player_id) VALUES (?)", [(p,) for p in players])
    conn.commit()
    register_model(conn, "draft:v1", kind="draft")
    seed_game(conn, "g1")
    scope = PickScope(year_start=date(2026, 1, 1), role_pools=conf_role_pools())
    slots = player_pick_scoring._reconcile_slots(conn, "draft:v1", scope, force=True)
    conn.close()

    seen: dict[str, Path] = {}

    def fake_load_grader(_conn, *, artifact: Path):
        seen["artifact"] = artifact
        return MockPredictor()

    monkeypatch.setattr(player_pick_scoring, "load_grader", fake_load_grader)
    monkeypatch.setattr(
        player_pick_scoring,
        "BulkStatsCache",
        lambda _conn, baseline: None,  # noqa: ARG005 - BulkStatsCache(conn, baseline=...) binds baseline by keyword
    )

    rows, counts = player_pick_scoring._parallel_pick_worker(
        str(db_path),
        "models/draft/draft.json",
        model_id="draft:v1",
        slots=slots,
        scope=scope,
    )

    assert seen["artifact"] == Path("models/draft/draft.json")
    assert counts["scored"] == 10
    assert len(rows) == 10


def test_parallel_scoring_uses_direct_worker_when_memory_allows_one(tmp_path, monkeypatch):
    """A memory-starved box must not pay process-pool overhead for width one."""
    conn = sqlite3.connect(tmp_path / "pick-scores.db")
    conn.execute(
        "CREATE TABLE pick_scores (game_id, side, role, model_id, player_id, actual_score, "
        "alt_count, best_score, worst_score, percentile)"
    )
    seen: list[list[dict]] = []
    monkeypatch.setattr(player_pick_scoring.resources, "bounded_workers", lambda *_a, **_k: 1)
    monkeypatch.setattr(
        player_pick_scoring,
        "_parallel_pick_worker",
        lambda _db, _model, slots, **_kw: (
            seen.append(slots) or [],
            {
                "scored": 0,
                "skipped_no_draft": 0,
                "skipped_predict_error": 0,
                "skipped_degenerate": 0,
            },
        ),
    )

    class _ForbiddenPool:
        def __init__(self, **_kwargs):
            raise AssertionError("single-worker scoring must not create a process pool")

    monkeypatch.setattr(player_pick_scoring, "ProcessPoolExecutor", _ForbiddenPool)
    scope = PickScope(year_start=date(2026, 1, 1), role_pools={})
    result = player_pick_scoring._dispatch_parallel(
        conn,
        "draft:test",
        [{"game_id": "g"}],
        scope,
        tmp_path / "draft.json",
        delete_stale=False,
    ).finish(conn)
    assert seen == [[{"game_id": "g"}]]
    assert result["scored"] == 0
    conn.close()
