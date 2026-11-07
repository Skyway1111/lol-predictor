"""The vectorized featurization is correct: bulk == single-query, plus pinned golden values."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

import pytest

from league_predictor.predraft.ratings.skill_filter import (
    BaggedSkillFilterBackend,
    BaggedSkillFilterIndex,
    derived_feature_names,
)
from league_predictor.predraft.ratings.skill_filter.artifact import save_bag_artifact
from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
from league_predictor.predraft.ratings.skill_filter.model import FilterConfig, SkillFilter
from tests.predraft.conftest import facts_db_from_legacy
from tests.predraft.ratings.conftest import _ROLES

if TYPE_CHECKING:
    import sqlite3

_THROUGH = date(2026, 6, 1)


# Six teams over ~5 months: multi-snapshot timelines, a 60-day trend lookback, a late
# debut (FF), and a mid-stream roster churn (EE) -- one row per gather branch.
_ROWS = (
    ("g01", "2026-01-05", "AA", "BB", "AA"),
    ("g02", "2026-01-12", "CC", "DD", "DD"),
    ("g03", "2026-01-20", "EE", "AA", "EE"),
    ("g04", "2026-02-02", "BB", "CC", "BB"),
    ("g05", "2026-02-15", "DD", "EE", "DD"),
    ("g06", "2026-02-28", "AA", "CC", "AA"),
    ("g07", "2026-03-10", "BB", "DD", "DD"),
    ("g08", "2026-03-22", "EE", "CC", "CC"),  # EE churns one starter here (see _pg_rows)
    ("g09", "2026-04-05", "AA", "DD", "AA"),
    ("g10", "2026-04-18", "BB", "EE", "EE"),
    ("g11", "2026-05-01", "CC", "AA", "CC"),
    ("g12", "2026-05-14", "FF", "DD", "FF"),  # FF debuts; never seen before this day
    ("g13", "2026-05-22", "EE", "FF", "EE"),
)


def _pg_rows(mid: str, tid: str) -> list[tuple]:
    """Five role rows for one team in one map; EE swaps its SUP from g08 on."""
    rows = []
    for r in _ROLES:
        pid = f"{tid.lower()}_{r}"
        if tid == "EE" and r == "sup" and mid in ("g08", "g13"):
            pid = "ee_sup2"  # churned-in support
        rows.append(
            (
                mid,
                tid,
                pid,
                r,
                3,
                2,
                5,
                60.0,
                6.0,
                90.0,
                110.0,
                11.0,
                130.0,
                0.21,
                1.2,
                0.21,
                12,
                8,
                1,
                1,
                0,
                "Blue",
            )
        )
    return rows


def _conn() -> sqlite3.Connection:
    ms, pgs = [], []
    for mid, day, t1, t2, w in _ROWS:
        ms.append((mid, f"{day} 10:00:00", t1, t2, w, "LCK", 1800, 1, "14.01", t1, "regular"))
        pgs += _pg_rows(mid, t1) + _pg_rows(mid, t2)
    return facts_db_from_legacy(ms, pgs)


def _index(tmp_path, conn, cfg):
    corpus = build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
    model = SkillFilter(cfg, n_leagues=max(1, corpus.vocab.n_leagues))
    path = tmp_path / "rating.pt"
    sha = save_bag_artifact(
        path,
        [model],
        cfg,
        corpus.norm,
        corpus.vocab.league_index,
        signature=corpus.signature,
        notes={},
    )
    return BaggedSkillFilterBackend(path, expected_sha=sha).build_index(
        conn, through=_THROUGH, query_dates=[]
    )


def _bag(tmp_path, conn, cfg, k=3):
    """A K-member BaggedSkillFilterIndex of independently-seeded indexes, no bag artifact."""
    members = []
    for i in range(k):
        sub = tmp_path / f"m{i}"
        sub.mkdir()
        members.append(_index(sub, conn, FilterConfig(**{**cfg.__dict__, "seed": i})))
    return BaggedSkillFilterIndex(members)


# A spread of matchups (both orientations), some same-day series-mates (dedup path),
# a never-seen team (FF early), and dates that straddle real meetings + a 60-day trend.
_QUERIES = [
    ("AA", "BB", date(2026, 3, 1)),
    ("BB", "AA", date(2026, 3, 1)),  # swap of the previous (dedup-distinct, swap-invariant)
    ("AA", "BB", date(2026, 3, 1)),  # exact dup -> dedup must scatter back correctly
    ("CC", "DD", date(2026, 4, 1)),
    ("EE", "AA", date(2026, 4, 20)),  # EE after its churn
    ("FF", "DD", date(2026, 3, 1)),  # FF never seen before this date -> pure prior
    ("FF", "DD", date(2026, 5, 25)),  # FF now has one appearance
    ("AA", "CC", date(2026, 5, 20)),
    ("DD", "EE", date(2026, 5, 30)),
    ("BB", "FF", date(2026, 4, 10)),
]

_CONFIGS = [
    pytest.param(FilterConfig(d=2, d_org=4, seed=3), id="deployed"),
]

_TOL = {"abs": 2e-5, "rel": 2e-5}  # bulk vs single-query: head GEMM reassociation only

# Pinned to the fixture's deterministic init under the current torch; regenerate when model
# init or gather/forward logic changes. abs=1e-4 absorbs cross-machine GEMM skew.
_GOLDEN_CFG = FilterConfig(d=2, d_org=4, seed=3)
_GOLDEN_RAW = {
    ("AA", "BB", date(2026, 3, 1)): 0.15747490525245667,
    ("EE", "AA", date(2026, 4, 20)): 0.11564621329307556,
    ("FF", "DD", date(2026, 3, 1)): -0.5517390370368958,  # FF never seen -> pure prior
}
_GOLDEN_DERIVED_AB = {
    "rating_strength_a": 0.019854925572872162,
    "rating_strength_b": -0.1376199722290039,
    "rating_logit_diff": 0.15747490525245667,  # IS the AA-vs-BB raw_score
}


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_raw_score_batch_matches_single_query(tmp_path, cfg):
    idx = _index(tmp_path, _conn(), cfg)
    batched = idx.raw_score_batch(_QUERIES)
    assert len(batched) == len(_QUERIES)
    for (a, b, d), got in zip(_QUERIES, batched, strict=True):
        assert got == pytest.approx(idx.raw_score(a, b, d), **_TOL), (a, b, d)


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_derived_features_batch_matches_single_query(tmp_path, cfg):
    idx = _index(tmp_path, _conn(), cfg)
    batched = idx.derived_features_batch(_QUERIES)
    assert len(batched) == len(_QUERIES)
    names = set(derived_feature_names(cfg.d, cfg.d_org))
    for (a, b, d), got in zip(_QUERIES, batched, strict=True):
        ref = idx.derived_features(a, b, d)
        assert set(got) == names, (a, b, d)
        for col in names:
            assert got[col] == pytest.approx(ref[col], **_TOL), (a, b, d, col)


def test_batched_empty_queries_returns_empty(tmp_path):
    idx = _index(tmp_path, _conn(), _CONFIGS[0].values[0])
    assert idx.raw_score_batch([]) == []
    assert idx.derived_features_batch([]) == []


def test_batched_unknown_team_reads_prior(tmp_path):
    """A team absent from the corpus reads the pure-prior state; bulk and single agree."""
    idx = _index(tmp_path, _conn(), FilterConfig(d=2, d_org=4, seed=3))
    qs = [
        ("ZZ", "AA", date(2026, 4, 1)),
        ("AA", "ZZ", date(2026, 4, 1)),
        ("ZZ", "YY", date(2026, 4, 1)),
    ]
    batched = idx.derived_features_batch(qs)
    raws = idx.raw_score_batch(qs)
    for (a, b, d), feat, raw in zip(qs, batched, raws, strict=True):
        ref = idx.derived_features(a, b, d)
        assert raw == pytest.approx(idx.raw_score(a, b, d), **_TOL), (a, b, d)
        for col in ref:
            assert feat[col] == pytest.approx(ref[col], **_TOL), (a, b, d, col)


def test_batched_dedup_preserves_order_and_duplicates(tmp_path):
    """Dedup/scatter maps a repeated query and its swap each back to their own row."""
    idx = _index(tmp_path, _conn(), _CONFIGS[0].values[0])
    qs = [
        ("AA", "BB", date(2026, 3, 1)),
        ("BB", "AA", date(2026, 3, 1)),
        ("AA", "BB", date(2026, 3, 1)),
    ]
    got = idx.raw_score_batch(qs)
    assert got[0] == pytest.approx(got[2], **_TOL)  # identical dup -> identical value
    assert got[0] == pytest.approx(-got[1], **_TOL)  # swap negates the logit


# Worst measured bulk-vs-single drift is ~1.2e-7 on CPU (head-GEMM reassociation only);
# the ceiling sits ~17x above it so a 10x creep still fails.
_DRIFT_CEILING = 2e-6


@pytest.mark.parametrize("cfg", _CONFIGS)
def test_bagged_batch_matches_single_query(tmp_path, cfg):
    bag = _bag(tmp_path, _conn(), cfg)
    raws = bag.raw_score_batch(_QUERIES)
    feats = bag.derived_features_batch(_QUERIES)
    for (a, b, d), raw, feat in zip(_QUERIES, raws, feats, strict=True):
        assert raw == pytest.approx(bag.raw_score(a, b, d), **_TOL), (a, b, d)
        ref = bag.derived_features(a, b, d)
        for col in ref:
            assert feat[col] == pytest.approx(ref[col], **_TOL), (a, b, d, col)


def test_golden_values_pin_the_featurization(tmp_path):
    """Exact pinned outputs for the fixture model: a gather/forward bug moves these values."""
    idx = _index(tmp_path, _conn(), _GOLDEN_CFG)
    raws = idx.raw_score_batch(list(_GOLDEN_RAW))
    for (a, b, d), got in zip(_GOLDEN_RAW, raws, strict=True):
        assert got == pytest.approx(_GOLDEN_RAW[(a, b, d)], abs=1e-4), (a, b, d)
    feat = idx.derived_features_batch([("AA", "BB", date(2026, 3, 1))])[0]
    for col, want in _GOLDEN_DERIVED_AB.items():
        assert feat[col] == pytest.approx(want, abs=1e-4), col
