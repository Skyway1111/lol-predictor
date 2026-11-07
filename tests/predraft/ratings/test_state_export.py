"""Tests for the team_state export from the serving timelines (skill_filter/state_export)."""

from __future__ import annotations

import sqlite3
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from league_predictor import clock
from league_predictor.db.asof import team_state_asof, team_state_coverage
from league_predictor.db.schema import apply_current_schema
from league_predictor.db.writes import replace_team_state
from league_predictor.predraft.ratings.skill_filter import state_export as se
from league_predictor.predraft.ratings.skill_filter.state_export import (
    _appearance_queries,
    _assemble_rows,
    export_team_state,
    team_state_rows,
)


def _frozen(records_by_org: dict[int, list[date]]):
    """A _FrozenTimelines stand-in carrying only the team_key/team_offset bands."""
    import torch

    from league_predictor.predraft.ratings.skill_filter.serving_frozen import (
        _KEY_STRIDE,
    )

    n_orgs = (max(records_by_org) + 1) if records_by_org else 0
    keys, offsets = [], [0]
    for oi in range(n_orgs):
        keys += [oi * _KEY_STRIDE + d.toordinal() for d in records_by_org.get(oi, [])]
        offsets.append(len(keys))
    return SimpleNamespace(
        team_key=torch.tensor(keys, dtype=torch.int64),
        team_offset=torch.tensor(offsets, dtype=torch.int64),
    )


# --- _appearance_queries: appearance-day sparsity ------------------------------------- #


def test_appearance_queries_are_sparse_and_per_day():
    fr = _frozen(
        {
            0: [date(2026, 1, 5), date(2026, 2, 1), date(2026, 3, 9)],
            1: [date(2026, 1, 5), date(2026, 2, 1)],
        }
    )
    teams, days = _appearance_queries(fr, {"AA": 0, "BB": 1})
    assert list(zip(teams, days, strict=True)) == [
        ("AA", date(2026, 1, 5)),
        ("AA", date(2026, 2, 1)),
        ("AA", date(2026, 3, 9)),
        ("BB", date(2026, 1, 5)),
        ("BB", date(2026, 2, 1)),
    ]


def test_appearance_queries_skip_orgs_without_a_team_id():
    """An org index with no team_id in the vocab inverse contributes no rows."""
    fr = _frozen({0: [date(2026, 1, 5)], 7: [date(2026, 1, 5)]})
    teams, days = _appearance_queries(fr, {"AA": 0})  # oi 7 unmapped
    assert teams == ["AA"]
    assert days == [date(2026, 1, 5)]


# --- _assemble_rows: stamping + strength ---------------------------------------------- #


def _ms(strength) -> float:
    return strength


def test_assemble_rows_stamps_asof_and_strength():
    day = date(2026, 3, 10)
    rows = _assemble_rows(["AA"], [day], [_ms(2.0)], "rating:m")
    (row,) = rows
    assert row.team_id == "AA"
    assert row.model_id == "rating:m"
    assert row.asof_date == "2026-03-10"  # asof = the appearance day itself
    assert row.strength == 2.0


# --- team_state_rows: rep-member-only ------------------------------------------------ #


class _Trap:
    """Any attribute access is a non-representative-member read — a regression."""

    def __getattr__(self, name):
        raise AssertionError(f"non-representative bag member was read (.{name})")


def testteam_state_rows_reads_only_the_representative_member(monkeypatch):
    seen = []

    def fake_member_states(member, teams, _query_dates):
        seen.append(member)
        return [_ms(1.0) for _ in teams]

    monkeypatch.setattr(se, "_member_states", fake_member_states)
    member0 = SimpleNamespace(_frozen_tl=_frozen({0: [date(2026, 1, 5)]}), _org_index={"AA": 0})
    index = SimpleNamespace(members=[member0, _Trap(), _Trap()])

    rows = team_state_rows(index, "rating:m")
    assert [r.team_id for r in rows] == ["AA"]
    # _member_states was called once and ONLY with member0.
    assert seen
    assert all(m is member0 for m in seen)
    assert len(seen) == 1


def testteam_state_rows_empty_when_no_appearances(monkeypatch):
    monkeypatch.setattr(
        se, "_member_states", lambda *_: pytest.fail("no readout when there are no appearances")
    )
    index = SimpleNamespace(members=[SimpleNamespace(_frozen_tl=_frozen({}), _org_index={})])
    assert team_state_rows(index, "rating:m") == []


# --- replace_team_state -> team_state_asof round-trip (the temporal rule) -------------- #


def test_row_dated_d_is_served_for_d_plus_one_not_d():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    apply_current_schema(conn)
    conn.execute("INSERT INTO models (model_id, kind) VALUES ('rating:m', 'rating')")

    rows = _assemble_rows(["AA"], [date(2026, 3, 10)], [_ms(2.0)], "rating:m")
    replace_team_state(conn, "rating:m", rows)

    assert team_state_asof(conn, ["AA"], "rating:m", date(2026, 3, 10)) == {}  # same day excluded
    served = team_state_asof(conn, ["AA"], "rating:m", date(2026, 3, 11))  # D+1 served
    assert served["AA"].strength == 2.0
    assert served["AA"].asof_date == "2026-03-10"
    count, max_asof = team_state_coverage(conn, "rating:m")
    assert count == 1
    assert max_asof == "2026-03-10"
    conn.close()


# --- tiny 1-member REAL index: readout == derived_features_batch (fast) ---------------- #


def _tiny_bag_backend(tmp_path, conn):
    """A 1-member BaggedSkillFilterBackend over the synthetic corpus, exercising the bag path."""
    from league_predictor.predraft.ratings.skill_filter import BaggedSkillFilterBackend
    from league_predictor.predraft.ratings.skill_filter.artifact import save_bag_artifact
    from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
    from league_predictor.predraft.ratings.skill_filter.model import FilterConfig, SkillFilter

    cfg = FilterConfig(d=2, d_org=4, seed=3)
    corpus = build_corpus(conn, through=date(2026, 4, 1), norm_through=date(2026, 4, 1))
    model = SkillFilter(cfg, n_leagues=max(1, corpus.vocab.n_leagues))
    path = tmp_path / "bag.pt"
    sha = save_bag_artifact(
        path,
        [model],
        cfg,
        corpus.norm,
        corpus.vocab.league_index,
        signature=corpus.signature,
        notes={},
    )
    return BaggedSkillFilterBackend(path, expected_sha=sha)


def test_readout_matches_derived_features_batch(tmp_path):
    """Stored strength equals the same member's self-matchup read at D+1."""
    from tests.predraft.ratings.conftest import _conn

    conn = _conn()
    backend = _tiny_bag_backend(tmp_path, conn)
    index = backend.build_index(conn, through=date(2026, 4, 1), query_dates=[])
    rows = team_state_rows(index, "rating:test")
    assert rows

    member = index.members[0]
    row = rows[-1]
    q = date.fromisoformat(row.asof_date) + timedelta(days=1)  # export queries at D+1
    feats = member.derived_features_batch([(row.team_id, row.team_id, q)])[0]
    assert row.strength == pytest.approx(feats["rating_strength_a"], abs=1e-5)


def test_export_team_state_wrapper_is_deterministic(tmp_path):
    """export_team_state's fresh internal build reproduces the explicit-build rows."""
    from tests.predraft.ratings.conftest import _conn

    conn = _conn()
    backend = _tiny_bag_backend(tmp_path, conn)
    rows = export_team_state(conn, backend, "rating:test")
    assert rows
    explicit = team_state_rows(
        backend.build_index(conn, through=clock.utc_today(), query_dates=[]), "rating:test"
    )
    # export uses clock.utc_today() internally; match that build.
    assert len(rows) == len(explicit)
    assert rows[0].team_id == explicit[0].team_id
    assert rows[0].strength == pytest.approx(explicit[0].strength, abs=1e-6)


# --- slow smokes: the write half (self-contained) + the full real-corpus half --------- #


@pytest.mark.slow
def test_export_smoke_write_half(tmp_path):
    """Exported rows land in a fresh rebuild-schema target and read back through coverage."""
    from league_predictor.db import get_connection
    from league_predictor.db.registry import ensure_registered
    from league_predictor.db.rows import ModelRow
    from tests.predraft.ratings.conftest import _conn

    corpus = _conn()
    backend = _tiny_bag_backend(tmp_path, corpus)
    rows = export_team_state(corpus, backend, "rating:test")
    assert rows

    target_path = tmp_path / "rebuild_smoke.db"
    target = get_connection(target_path)
    try:
        apply_current_schema(target)
        ensure_registered(target, ModelRow(model_id="rating:test", kind="rating"))
        target.commit()
        replace_team_state(target, "rating:test", rows)
        target.commit()
        count, max_asof = team_state_coverage(target, "rating:test")
        assert count == len(rows)
        assert max_asof == max(r.asof_date for r in rows)
    finally:
        target.close()


@pytest.mark.slow
def test_export_smoke_against_live_db(capsys):
    """The pinned bag replayed over the live DB into the REBUILD_SMOKE_DB target (skips unset)."""
    import os
    import time

    from league_predictor.config import DB_PATH
    from league_predictor.db import get_connection, registry
    from league_predictor.predraft.pipeline.chosen_skill_filter import chosen_skill_filter_backend
    from league_predictor.predraft.pipeline.model_identity import (
        current_rating_model_id,
        model_row,
    )

    smoke_target = os.environ.get("REBUILD_SMOKE_DB")
    if not smoke_target:
        pytest.skip(
            "REBUILD_SMOKE_DB not set -- point it at a populated current-schema "
            "DB to run the full export smoke"
        )
    if not Path(DB_PATH).exists() or not Path(smoke_target).exists():
        pytest.skip("live DB corpus or the REBUILD_SMOKE_DB target not present")

    corpus = get_connection(DB_PATH)
    rating_id = current_rating_model_id()
    t0 = time.time()
    index = chosen_skill_filter_backend().build_index(
        corpus, through=clock.utc_today(), query_dates=[]
    )
    build_s = time.time() - t0
    t1 = time.time()
    rows = team_state_rows(index, rating_id)
    export_s = time.time() - t1
    with capsys.disabled():
        print(  # noqa: T201 - smoke diagnostics for the operator
            f"\n[smoke] index build {build_s:.1f}s | readout {export_s:.1f}s | {len(rows)} rows"
        )

    assert len(rows) > 100_000, f"only {len(rows)} team_state rows"

    # Spot-check: the stored strength equals a direct self-matchup query at the same D+1.
    row = rows[len(rows) // 2]
    q = date.fromisoformat(row.asof_date) + timedelta(days=1)
    feats = index.members[0].derived_features_batch([(row.team_id, row.team_id, q)])[0]
    assert row.strength == pytest.approx(feats["rating_strength_a"], abs=1e-4)

    target = get_connection(Path(smoke_target))
    try:
        registry.ensure_registered(target, model_row("rating"))
        target.commit()
        replace_team_state(target, rating_id, rows)
        target.commit()
        count, max_asof = team_state_coverage(target, rating_id)
        with capsys.disabled():
            print(f"[smoke] wrote {count} rows, max asof {max_asof}")  # noqa: T201 - smoke line
        assert count == len(rows)
    finally:
        target.close()
    corpus.close()
