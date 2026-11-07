"""Walk-forward, leakage, and batched-equivalence guards for Rating replay."""

from __future__ import annotations

import dataclasses
from datetime import date

import torch

from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
from league_predictor.predraft.ratings.skill_filter.model import FilterConfig, SkillFilter
from league_predictor.predraft.ratings.skill_filter.replay import (
    replay_for_serving,
)
from league_predictor.predraft.ratings.skill_filter.replay_bag import replay_bag_for_serving
from league_predictor.predraft.ratings.skill_filter.replay_inputs import precompute_replay_statics
from tests.predraft.conftest import facts_db_from_legacy
from tests.predraft.ratings._replay_reference import replay_for_training
from tests.predraft.ratings.conftest import _five, _match


def _toy_conn(*, last_winner="AA", last_gd15=100.0):
    matches = [
        _match("m1", "2026-01-01", "AA", "BB", "BB"),
        _match("m2", "2026-01-05", "AA", "CC", "AA"),
        _match("m3", "2026-01-05", "BB", "CC", "BB"),
        _match("m4", "2026-01-05", "AA", "BB", "AA", gn=2),
        _match("m5", "2026-01-20", "AA", "BB", last_winner),
    ]
    games = []
    for mid, a, b in (
        ("m1", "AA", "BB"),
        ("m2", "AA", "CC"),
        ("m3", "BB", "CC"),
        ("m4", "AA", "BB"),
        ("m5", "AA", "BB"),
    ):
        gd = last_gd15 if mid == "m5" else 100.0
        games += _five(mid, a, a.lower(), gd15=gd)
        games += _five(mid, b, b.lower(), gd15=-gd)
    return facts_db_from_legacy(matches, games)


def _corpus(conn):
    return build_corpus(conn, through=date(2026, 2, 1), norm_through=date(2026, 1, 10))


def _model(corpus, seed=3):
    return SkillFilter(FilterConfig(d=4, seed=seed), max(1, corpus.vocab.n_leagues))


def test_future_mutation_leaves_past_predictions_unchanged():
    base_corpus = _corpus(_toy_conn())
    changed_corpus = _corpus(_toy_conn(last_winner="BB", last_gd15=900.0))
    model = _model(base_corpus)
    base = replay_for_training(base_corpus, model)
    changed = replay_for_training(changed_corpus, model)
    assert torch.equal(base.logits[:4], changed.logits[:4])
    assert not torch.equal(base.labels[4:], changed.labels[4:])


def test_games_own_outcome_and_box_score_cannot_move_its_prediction():
    corpus = _corpus(_toy_conn())
    model = _model(corpus)
    base = replay_for_training(corpus, model)
    last = corpus.days[-1]
    poisoned = dataclasses.replace(
        corpus,
        days=[
            *corpus.days[:-1],
            dataclasses.replace(
                last,
                perf=last.perf + last.perf_mask * 1000000.0,
                team_feat=last.team_feat + last.team_mask * 1000000.0,
                a_won=1.0 - last.a_won,
            ),
        ],
    )
    assert torch.equal(base.logits, replay_for_training(poisoned, model).logits)


def test_replay_is_deterministic_and_differentiable():
    corpus = _corpus(_toy_conn())
    a = replay_for_training(corpus, _model(corpus))
    model = _model(corpus)
    b = replay_for_training(corpus, model)
    assert torch.equal(a.logits, b.logits)
    torch.nn.functional.binary_cross_entropy_with_logits(b.logits, b.labels).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_margin_supervision_is_aligned_and_leak_safe():
    corpus = _corpus(_toy_conn())
    model = _model(corpus)
    base = replay_for_training(corpus, model, with_margin=True)
    last = corpus.days[-1]
    changed = dataclasses.replace(
        corpus,
        days=[
            *corpus.days[:-1],
            dataclasses.replace(last, team_feat=last.team_feat + last.team_mask * 1000000.0),
        ],
    )
    poisoned = replay_for_training(changed, model, with_margin=True)
    assert base.margin_pred is not None
    assert base.margin_target is not None
    assert torch.equal(base.margin_pred, poisoned.margin_pred)
    assert not torch.equal(base.margin_target[-1:], poisoned.margin_target[-1:])


def test_precomputed_statics_are_bit_identical():
    corpus = _corpus(_toy_conn())
    model = _model(corpus, seed=5)
    statics = precompute_replay_statics(corpus, norm=corpus.norm)
    direct = replay_for_training(corpus, model, norm=corpus.norm, with_margin=True)
    cached = replay_for_training(corpus, model, norm=corpus.norm, with_margin=True, statics=statics)
    assert torch.equal(direct.logits, cached.logits)
    assert torch.equal(direct.margin_pred, cached.margin_pred)


def test_batched_serving_replay_matches_each_member():
    corpus = _corpus(_toy_conn())
    models = [_model(corpus, seed) for seed in range(3)]
    batched = replay_bag_for_serving(corpus, models, norm=corpus.norm)
    for member, model in enumerate(models):
        solo = replay_for_serving(corpus, model, norm=corpus.norm)
        actual = batched.member(member)
        assert actual.player_dates == solo.player_dates
        assert actual.org_dates == solo.org_dates
        assert actual.team_records == solo.team_records
        for pid, states in solo.player_states.items():
            for expected, got in zip(states, actual.player_states[pid], strict=True):
                assert torch.allclose(expected, got, atol=1e-6)
        for org, states in solo.org_states.items():
            for expected, got in zip(states, actual.org_states[org], strict=True):
                assert torch.allclose(expected, got, atol=1e-6)
