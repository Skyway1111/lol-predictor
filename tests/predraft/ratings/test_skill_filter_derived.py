"""Tests for the skill filter serving index's derived matchup features as a feature family."""

from __future__ import annotations

from datetime import date

import pytest

from league_predictor.predraft.ratings.skill_filter import (
    BaggedSkillFilterBackend,
    derived_feature_names,
)
from league_predictor.predraft.ratings.skill_filter.artifact import save_bag_artifact
from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
from league_predictor.predraft.ratings.skill_filter.model import FilterConfig, SkillFilter
from tests.predraft.ratings.conftest import _conn


def _index(tmp_path, conn, cfg):
    corpus = build_corpus(conn, through=date(2026, 4, 1), norm_through=date(2026, 4, 1))
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
        conn, through=date(2026, 4, 1), query_dates=[]
    )


# derived_feature_names -- the one source of truth for the column set


def test_derived_feature_names_cover_the_six_concepts():
    names = derived_feature_names(d=2, d_org=4)
    assert len(names) == len(set(names)) == 20
    assert {"rating_strength_a", "rating_strength_b", "rating_logit_diff"} <= set(names)  # head
    assert {"rating_rd", "rating_rd_diff"} <= set(names)  # uncertainty axis
    assert {"rating_spread_a", "rating_spread_b"} <= set(names)  # dispersion
    assert {"rating_trend_a", "rating_trend_b", "rating_trend_diff"} <= set(names)  # trend
    assert {
        "rating_latent_a_c0",
        "rating_latent_a_c1",
        "rating_latent_b_c0",  # composite
        "rating_latent_b_c1",
        "rating_latent_diff_c0",
        "rating_latent_diff_c1",
    } <= set(names)
    assert {
        "rating_org_diff_c0",
        "rating_org_diff_c1",  # org residual
        "rating_org_diff_c2",
        "rating_org_diff_c3",
    } <= set(names)


def test_derived_feature_names_scale_with_the_state_dims():
    # 10 scalars + 3 cols per latent channel (a/b/diff) + 1 org-diff col per org channel.
    assert len(derived_feature_names(d=3, d_org=2)) == 10 + 3 * 3 + 2


# derived_features -- the matchup family the index emits

_CFG = FilterConfig(d=2, d_org=4, seed=3)
_AS_OF = date(2026, 3, 25)


def test_derived_features_emit_exactly_the_declared_names(tmp_path):
    feats = _index(tmp_path, _conn(), _CFG).derived_features("AA", "BB", _AS_OF)
    assert set(feats) == set(derived_feature_names(_CFG.d, _CFG.d_org))
    assert all(isinstance(v, float) for v in feats.values())


def test_rating_logit_diff_is_the_raw_score_head_across_dates(tmp_path):
    """rating_logit_diff equals raw_score at every query date, match days included."""
    idx = _index(tmp_path, _conn(), _CFG)
    for day in (date(2026, 1, 20), date(2026, 2, 5), date(2026, 3, 12), _AS_OF):
        head = idx.derived_features("AA", "BB", day)["rating_logit_diff"]
        assert head == pytest.approx(idx.raw_score("AA", "BB", day), abs=1e-6), day


def test_training_and_serving_hist_match_for_a_stable_roster(tmp_path):
    """The training replay's history signal equals the serving index's, for a stable roster."""
    import torch

    from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
    from league_predictor.predraft.ratings.skill_filter.model import hist_features

    conn = _conn()
    corpus = build_corpus(conn, through=date(2026, 4, 1), norm_through=date(2026, 4, 1))
    idx = _index(tmp_path, conn, _CFG)  # same data; rebuilds the same corpus internally
    m3 = next(db for db in corpus.days if db.date == date(2026, 3, 12))

    train_hist = hist_features(m3.roster_cont, m3.debut)  # what the replay feeds
    a = idx.members[0]._batched_team_state(torch, ["AA"], [date(2026, 3, 12)])
    b = idx.members[0]._batched_team_state(torch, ["BB"], [date(2026, 3, 12)])
    serve_hist = idx.members[0]._batched_pair_strength(torch, a, b)[1]  # what the index builds
    assert torch.allclose(train_hist[0], serve_hist[0, :, :2], atol=1e-6)


def test_single_query_equals_bulk_featurization(tmp_path):
    """Single-query featurization agrees with the bulk path over the same queries."""
    idx = _index(tmp_path, _conn(), _CFG)
    queries = [
        (a, b, day)
        for a, b in (("AA", "BB"), ("BB", "AA"), ("CC", "DD"))
        for day in (date(2026, 1, 20), date(2026, 3, 12), _AS_OF)
    ]
    bulk_feats = idx.derived_features_batch(queries)
    bulk_raw = idx.raw_score_batch(queries)
    for (a, b, day), feat, raw in zip(queries, bulk_feats, bulk_raw, strict=True):
        single = idx.derived_features(a, b, day)
        for col in single:
            assert single[col] == pytest.approx(feat[col], abs=2e-5), (a, b, day, col)
        assert idx.raw_score(a, b, day) == pytest.approx(raw, abs=2e-5), (a, b, day)


def test_derived_features_obey_the_swap_transform_invariant(tmp_path):
    """Per-team values exchange, _diff columns negate, the symmetric confidence is unchanged."""
    idx = _index(tmp_path, _conn(), _CFG)
    ab = idx.derived_features("AA", "BB", _AS_OF)
    ba = idx.derived_features("BB", "AA", _AS_OF)

    for col in (n for n in ab if "_diff" in n):  # incl. rating_rd_diff (per-side imbalance)
        assert ab[col] == pytest.approx(-ba[col], abs=1e-6), col
    assert ab["rating_rd"] == pytest.approx(ba["rating_rd"], abs=1e-6)  # uncertainty is symmetric
    assert ab["rating_rd"] > 0.0
    for a_col in (
        "rating_strength_a",
        "rating_spread_a",
        "rating_trend_a",
        "rating_latent_a_c0",
        "rating_latent_a_c1",
    ):
        b_col = a_col.replace("_a", "_b", 1)
        assert ab[a_col] == pytest.approx(ba[b_col], abs=1e-6), a_col
        assert ab[b_col] == pytest.approx(ba[a_col], abs=1e-6), b_col


# A facts game reads its own five and context, the inputs the training replay scored it with.


def _game_conn():
    """AA fields a substitute, then plays MSI game 2 on red; a tier-2 team meets AA at MSI."""
    from tests.predraft.conftest import facts_db_from_legacy
    from tests.predraft.ratings.conftest import _ROLES, _match

    matches = (
        _match("g1", "2026-01-01", "AA", "BB", "BB"),
        _match("g2", "2026-02-05", "AA", "BB", "AA"),
        _match("g3", "2026-03-12", "AA", "BB", "AA", tournament="MSI", gn=2, blue="BB"),
        _match("g4", "2026-03-20", "CC", "DD", "CC", tournament="VCS", patch="14.02"),
        _match("g5", "2026-03-25", "AA", "DD", "DD", tournament="MSI", patch="14.02"),
    )
    stats = (2, 1, 3, 50.0, 5.0, 80.0, 100.0, 10.0, 120.0, 0.2, 1.1, 0.2, 10, 7, 1, 1, 0, "Blue")
    players = [
        (
            m[0],
            t,
            "aa_sub" if (m[0], t, r) == ("g2", "AA", "top") else f"{t.lower()}_{r}",
            r,
            *stats,
        )
        for m in matches
        for t in (m[2], m[3])
        for r in _ROLES
    ]
    return facts_db_from_legacy(list(matches), players)


def _game_index(tmp_path, conn, model, corpus, recipe):
    path = tmp_path / f"rating_{recipe is not None}.pt"
    sha = save_bag_artifact(
        path,
        [model],
        _CFG,
        corpus.norm,
        corpus.vocab.league_index,
        signature=corpus.signature,
        notes={},
        recipe=recipe,
    )
    return BaggedSkillFilterBackend(path, expected_sha=sha).build_index(
        conn, through=date(2026, 4, 1), query_dates=[]
    )


def test_a_facts_game_reads_the_inputs_the_replay_scored_it_with(tmp_path):
    import torch

    from league_predictor.asof import AsOf, MatchContext
    from league_predictor.predraft.ratings.skill_filter.recipe import RATING_RECIPE
    from tests.predraft.ratings._replay_reference import replay_for_training

    conn = _game_conn()
    corpus = build_corpus(conn, through=date(2026, 4, 1), norm_through=date(2026, 4, 1))
    model = SkillFilter(_CFG, n_leagues=max(1, corpus.vocab.n_leagues))
    torch.manual_seed(0)
    with torch.no_grad():  # move the zero-initialized heads so states and context matter
        for p in model.parameters():
            p.add_(0.3 * torch.randn_like(p))
    index = _game_index(tmp_path, conn, model, corpus, RATING_RECIPE.to_dict())
    inv = {i: t for t, i in corpus.vocab.org_index.items()}
    games = [
        (inv[a], inv[b], AsOf.game(db.date.isoformat(), *pos))
        for db in corpus.days
        for (a, b), pos in zip(db.orgs.tolist(), db.positions, strict=True)
    ]
    replay = replay_for_training(corpus, model).logits

    got = torch.tensor(index.raw_score_batch(games))
    day = torch.tensor(index.raw_score_batch([(a, b, q.day) for a, b, q in games]))
    assert torch.allclose(got, replay, atol=1e-5)
    assert (day - replay).abs().max() > 1e-2  # the substitute and the context matter here
    swapped = torch.tensor(index.raw_score_batch([(b, a, q) for a, b, q in games]))
    assert torch.allclose(swapped, -got, atol=1e-6)

    a, b, q = games[-1]
    scheduled = AsOf(q.ts, context=MatchContext.of("Bo1", "regular"))
    assert index.raw_score(a, b, scheduled) == pytest.approx(index.raw_score(a, b, q.day))
    with pytest.raises(ValueError, match="queried as"):
        index.raw_score("AA", "CC", q)
    unstamped = _game_index(tmp_path, conn, model, corpus, None)
    assert torch.allclose(torch.tensor(unstamped.raw_score_batch(games)), day, atol=1e-6)
