"""Tests for ``LatentStrengthFamily``, the skill-filter-derived family in the served rating slot."""

from __future__ import annotations

import sqlite3
from datetime import date

import pytest

from league_predictor.asof import AsOf
from league_predictor.predraft.features import (
    build_features,
    build_features_batch,
)
from league_predictor.predraft.features.latent_strength import LatentStrengthFamily
from league_predictor.predraft.ratings.skill_filter import (
    BaggedSkillFilterBackend,
    derived_feature_names,
)
from league_predictor.predraft.ratings.skill_filter.artifact import save_bag_artifact
from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
from league_predictor.predraft.ratings.skill_filter.model import FilterConfig, SkillFilter
from tests.predraft.conftest import facts_db_from_legacy

_ROLES = ("top", "jng", "mid", "bot", "sup")
_CFG = FilterConfig(d=2, d_org=4, seed=1)
_THROUGH = date(2026, 4, 1)
_AS_OF = date(2026, 3, 25)


def _conn() -> sqlite3.Connection:
    ms, pgs = [], []
    for mid, day, t1, t2, w in (
        ("m1", "2026-01-01", "AA", "BB", "BB"),
        ("m2", "2026-02-05", "AA", "BB", "AA"),
        ("m3", "2026-03-12", "AA", "BB", "AA"),
    ):
        ms.append((mid, f"{day} 10:00:00", t1, t2, w, "LCK", 1800, 1, "14.01", t1, "regular"))
        for tid in (t1, t2):
            pgs += [
                (
                    mid,
                    tid,
                    f"{tid.lower()}_{r}",
                    r,
                    2,
                    1,
                    3,
                    50.0,
                    5.0,
                    80.0,
                    100.0,
                    10.0,
                    120.0,
                    0.2,
                    1.1,
                    0.2,
                    10,
                    7,
                    1,
                    1,
                    0,
                    "Blue",
                )
                for r in _ROLES
            ]
    return facts_db_from_legacy(ms, pgs)


def _backend(tmp_path) -> BaggedSkillFilterBackend:
    conn = _conn()
    corpus = build_corpus(conn, through=_THROUGH, norm_through=_THROUGH)
    model = SkillFilter(_CFG, n_leagues=max(1, corpus.vocab.n_leagues))
    path = tmp_path / "rating.pt"
    sha = save_bag_artifact(
        path,
        [model],
        _CFG,
        corpus.norm,
        corpus.vocab.league_index,
        signature=corpus.signature,
        notes={},
    )
    return BaggedSkillFilterBackend(path, expected_sha=sha)


def test_family_feature_names_are_the_declared_derived_set():
    fam = LatentStrengthFamily(_CFG)
    assert fam.name == "learned_rating"
    assert fam.feature_names == derived_feature_names(_CFG.d, _CFG.d_org)
    assert fam.categorical_names == []


def test_family_build_emits_exactly_its_feature_names(tmp_path):
    fam = LatentStrengthFamily(_CFG, backend=_backend(tmp_path))
    index = fam.build(_conn(), through=_THROUGH, query_dates=[])
    assert set(index.features("AA", "BB", AsOf.of(_AS_OF))) == set(fam.feature_names)


def test_family_is_antisymmetric_through_the_assembler(tmp_path):
    index = LatentStrengthFamily(_CFG, backend=_backend(tmp_path)).build(
        _conn(), through=_THROUGH, query_dates=[]
    )
    kw = {"indexes": {"learned_rating": index}, "families": ["learned_rating"]}
    ab = build_features("AA", "BB", _AS_OF, **kw)
    ba = build_features("BB", "AA", _AS_OF, **kw)
    assert ab["rating_logit_diff"] == pytest.approx(-ba["rating_logit_diff"], abs=1e-6)
    assert ab["rating_rd"] == pytest.approx(ba["rating_rd"], abs=1e-6)  # uncertainty is symmetric
    assert ab["rating_strength_a"] == pytest.approx(ba["rating_strength_b"], abs=1e-6)


class _StubIndex:
    """A family index with only the per-row ``features`` contract, forcing the batch fallback."""

    def features(self, team_a: str, team_b: str, _as_of: date) -> dict[str, float]:
        return {"stub_sign": 1.0 if team_a < team_b else -1.0}


def test_family_features_batch_matches_per_row(tmp_path):
    """``features_batch`` is float-equivalent to per-row ``features`` for every query."""
    index = LatentStrengthFamily(_CFG, backend=_backend(tmp_path)).build(
        _conn(), through=_THROUGH, query_dates=[]
    )
    q = AsOf.of(_AS_OF)
    queries = [("AA", "BB", q), ("BB", "AA", q), ("AA", "BB", AsOf.of(date(2026, 2, 20)))]
    batched = index.features_batch(queries)
    for (a, b, d), got in zip(queries, batched, strict=True):
        ref = index.features(a, b, d)
        assert set(got) == set(ref)
        for col in ref:
            assert got[col] == pytest.approx(ref[col], abs=2e-5), (a, b, d, col)


def test_build_features_batch_equals_per_row_and_unions_fallback_families(tmp_path):
    """A batched family plus a per-row-only family union into the same ``build_features`` rows."""
    rating_idx = LatentStrengthFamily(_CFG, backend=_backend(tmp_path)).build(
        _conn(), through=_THROUGH, query_dates=[]
    )
    indexes = {"learned_rating": rating_idx, "stub": _StubIndex()}
    families = ["learned_rating", "stub"]
    queries = [("AA", "BB", _AS_OF), ("BB", "AA", _AS_OF)]
    rows = build_features_batch(queries, indexes=indexes, families=families)
    for (a, b, d), got in zip(queries, rows, strict=True):
        ref = build_features(a, b, d, indexes=indexes, families=families)
        assert set(got) == set(ref)
        for col in ref:
            assert got[col] == pytest.approx(ref[col], abs=2e-5), (a, b, d, col)
    assert rows[0]["stub_sign"] == 1.0
    assert rows[1]["stub_sign"] == -1.0  # fallback ran


def test_config_only_family_refuses_to_build_its_own_index():
    """The per-window judge path supplies the index externally; a backend-less family must raise."""
    with pytest.raises(ValueError, match="backend"):
        LatentStrengthFamily(_CFG).build(
            sqlite3.connect(":memory:"), through=_THROUGH, query_dates=[]
        )
