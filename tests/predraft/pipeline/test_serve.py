"""Tests for the side-agnostic serve path (``pipeline/serve.py``) and its batched twins."""

from __future__ import annotations

import itertools
import math
from datetime import date

import pytest

from league_predictor.asof import AsOf
from league_predictor.predraft.features.families.recency import RecencyFamily
from league_predictor.predraft.pipeline.chosen_model import (
    SERVED_TEMPERATURE,
    ServedModel,
)
from league_predictor.predraft.pipeline.lightgbm_bag import LightGBMBagBackend
from league_predictor.predraft.pipeline.model_contract import predict_symmetric
from league_predictor.predraft.pipeline.serve import (
    adjust_map_prob,
    base_score,
    base_score_batch,
    blend_toward_rating,
    blended_base,
    feature_computation,
    game_prob,
    predict,
    predict_batch,
    rating_logit,
    series_prob,
)
from league_predictor.predraft.pipeline.staleness import dampen_if_stale
from league_predictor.predraft.pipeline.temperature import apply_temperature
from league_predictor.predraft.pipeline.walk_forward import FeaturizedCorpus, build_indexes
from league_predictor.predraft.ratings.region_rating import RegionRatingTable


def _fit_and_index(conn):
    fam = RecencyFamily()
    idx = fam.build(conn, through=date(2026, 4, 15), query_dates=[])
    ds = FeaturizedCorpus(
        conn,
        start=date.min,
        end=date(2026, 4, 15),
        families=[fam],
        indexes={"recency": idx},
        drop_features=frozenset(),
    ).window(date.min, date(2026, 4, 15))
    fitted = LightGBMBagBackend().fit(
        ds.X,
        ds.y,
        sample_weight=None,
        feature_names=ds.feature_names,
        categorical_names=ds.categorical_names,
    )
    return fitted, {"recency": idx}


def _served(conn):
    """A fitted recency-only ServedModel + its as-of indexes."""
    fitted, idx = _fit_and_index(conn)
    model = ServedModel(
        fitted,
        families=["recency"],
        feature_names=list(fitted._feature_names),
    )
    return model, idx


def test_game_prob_is_side_agnostic_and_correctly_signed(skill_db):
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    fitted, idx = _fit_and_index(conn)
    p = game_prob(fitted, idx, "AAA", "DDD", date(2026, 4, 14), families=["recency"])
    q = game_prob(fitted, idx, "DDD", "AAA", date(2026, 4, 14), families=["recency"])
    assert 0.0 < p < 1.0
    assert p > 0.5  # AAA is the strong team
    assert abs(p + q - 1.0) < 1e-9  # exactly side-agnostic


def test_series_prob_bo1_passthrough_and_bo3_amplifies_symmetrically():
    assert series_prob(0.7, "Bo1") == 0.7  # single map: series == map
    assert series_prob(0.7, "Bo3") > 0.7  # best-of amplifies the favourite
    assert abs(series_prob(0.7, "Bo3") + series_prob(0.3, "Bo3") - 1.0) < 1e-9  # swap-symmetric


# --- the rating-blend primitive: rating_logit + blend_toward_rating ---------------------------


def test_rating_logit_symmetrizes_the_two_orientations():
    """z_rating = 0.5 * (ab - ba) of rating_logit_diff, exactly antisymmetric under a team swap."""
    ab = {"rating_logit_diff": 0.80}
    ba = {"rating_logit_diff": -0.82}  # not exactly -0.80: batched drift
    assert rating_logit(ab, ba) == pytest.approx(0.5 * (0.80 - (-0.82)))
    assert rating_logit(ab, ba) == pytest.approx(-rating_logit(ba, ab), abs=1e-15)


def test_blend_is_bit_exact_identity_at_zero():
    """w == 0.0 returns the input prob object itself, never a logit round-trip of it."""
    for p in (0.0, 1e-12, 0.5, 0.7333333333333333, 1.0 - 1e-9, 1.0):
        assert blend_toward_rating(p, 3.7, 0.0) is p


def test_blend_math_is_sigmoid_of_the_logit_mix():
    """At w = 0.7 the blend is exactly sigmoid((1-w)*logit(p) + w*z_rating)."""
    p, z, w = 0.6, 1.2, 0.7
    expected = 1.0 / (1.0 + math.exp(-((1.0 - w) * math.log(0.6 / 0.4) + w * z)))
    assert blend_toward_rating(p, z, w) == pytest.approx(expected, abs=1e-15)


def test_blend_is_antisymmetric():
    """blend(1 - p, -z, w) == 1 - blend(p, z, w), so the blended base stays side-agnostic."""
    w = 0.7
    for p, z in [(0.62, 1.1), (0.30, -0.4), (0.50, 0.0), (0.91, 2.0)]:
        assert blend_toward_rating(1.0 - p, -z, w) == pytest.approx(
            1.0 - blend_toward_rating(p, z, w), abs=1e-15
        )


def test_blend_is_monotone_in_p_and_pulls_toward_the_rating():
    """Blending is monotone in p (w < 1), and a positive rating logit lifts a coin-flip base."""
    z, w = 0.5, 0.7
    probs = [blend_toward_rating(p, z, w) for p in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert probs == sorted(probs)
    assert all(a < b for a, b in itertools.pairwise(probs))
    assert blend_toward_rating(0.5, 0.5, 0.7) > 0.5  # z > 0 lifts the flip
    assert blend_toward_rating(0.5, -0.5, 0.7) < 0.5


# --- the ServedModel chain: feature_computation -> base_score -> predict ---


def test_feature_computation_assembles_the_chosen_subset_antisymmetrically(skill_db):
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)
    feats = feature_computation(idx, "AAA", "DDD", date(2026, 4, 14), features=model.feature_names)
    rev = feature_computation(idx, "DDD", "AAA", date(2026, 4, 14), features=model.feature_names)
    assert set(feats) == set(model.feature_names)
    assert all(abs(feats[name] + rev[name]) < 1e-9 for name in model.feature_names)


def test_base_score_is_side_agnostic_and_signed(skill_db):
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)
    ab = feature_computation(idx, "AAA", "DDD", date(2026, 4, 14), features=model.feature_names)
    ba = feature_computation(idx, "DDD", "AAA", date(2026, 4, 14), features=model.feature_names)
    p = base_score(model, ab, ba)
    assert 0.0 < p < 1.0
    assert p > 0.5  # AAA favoured
    assert abs(p + base_score(model, ba, ab) - 1.0) < 1e-9


def _feats_with_rating(model, idx):
    """AAA-vs-DDD recency feats both ways, with an antisymmetric rating logit injected."""
    ab = feature_computation(idx, "AAA", "DDD", date(2026, 4, 14), features=model.feature_names)
    ba = feature_computation(idx, "DDD", "AAA", date(2026, 4, 14), features=model.feature_names)
    return {**ab, "rating_logit_diff": 0.8}, {**ba, "rating_logit_diff": -0.8}


def test_base_score_blends_toward_the_rating_when_weighted(skill_db):
    """rating_blend > 0 gives the blend_toward_rating image of the bag; 0 gives the bare bag."""
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)
    ab, ba = _feats_with_rating(model, idx)
    bare = predict_symmetric(model.bag, [ab], [ba])[0]

    model.rating_blend = 0.0
    assert base_score(model, ab, ba) == bare  # off -> byte-exact bare bag

    model.rating_blend = 0.7
    z_r = rating_logit(ab, ba)
    assert z_r == pytest.approx(0.8)  # symmetrized rating logit
    assert base_score(model, ab, ba) == blend_toward_rating(bare, z_r, 0.7)


def test_base_score_blend_is_side_agnostic(skill_db):
    """The blended base still sums to 1 across a team swap."""
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)
    ab, ba = _feats_with_rating(model, idx)
    model.rating_blend = 0.7
    assert abs(base_score(model, ab, ba) + base_score(model, ba, ab) - 1.0) < 1e-12


def test_base_score_requires_rating_blend(skill_db):
    """A model missing ``rating_blend`` raises rather than silently scoring the bare bag."""
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)
    ab, ba = _feats_with_rating(model, idx)
    del model.__dict__["rating_blend"]
    with pytest.raises(AttributeError):
        base_score(model, ab, ba)


def _served_with_lineup(conn):
    """A recency-scored ServedModel whose indexes also carry the lineup family (no league layer)."""
    fitted, _ = _fit_and_index(conn)
    model = ServedModel(
        fitted,
        families=["recency", "lineup"],
        feature_names=list(fitted._feature_names),
    )
    _, idx = build_indexes(conn, ["recency", "lineup"], through=date(2026, 4, 15))
    return model, idx


def test_predict_dampens_a_stale_matchup(skill_db):
    """Two idle teams pull the prob toward 0.5; a fresh matchup is byte-identical to the base."""
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served_with_lineup(conn)

    fresh = date(2026, 4, 18)  # 3 days after the last generated game -> not stale
    ab = feature_computation(idx, "AAA", "DDD", fresh, features=model.feature_names)
    ba = feature_computation(idx, "DDD", "AAA", fresh, features=model.feature_names)
    assert predict(model, idx, "AAA", "DDD", as_of=fresh) == apply_temperature(
        base_score(model, ab, ba), SERVED_TEMPERATURE
    )

    stale = date(2026, 6, 20)  # ~66 days idle -> stale
    sab = feature_computation(idx, "AAA", "DDD", stale, features=model.feature_names)
    sba = feature_computation(idx, "DDD", "AAA", stale, features=model.feature_names)
    base = base_score(model, sab, sba)
    p = predict(model, idx, "AAA", "DDD", as_of=stale)
    assert p != base
    assert abs(p - 0.5) < abs(base - 0.5)  # humbler than the bare base
    assert (p - 0.5) * (base - 0.5) > 0  # but still favours the same side


def test_predict_equals_base_and_is_side_agnostic(skill_db):
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)  # no league layer -> the chain is the bare base
    ab = feature_computation(idx, "AAA", "DDD", date(2026, 4, 14), features=model.feature_names)
    ba = feature_computation(idx, "DDD", "AAA", date(2026, 4, 14), features=model.feature_names)
    p = predict(model, idx, "AAA", "DDD", as_of=date(2026, 4, 14))
    assert p == apply_temperature(base_score(model, ab, ba), SERVED_TEMPERATURE)
    assert abs(p + predict(model, idx, "DDD", "AAA", as_of=date(2026, 4, 14)) - 1.0) < 1e-9


def test_predict_applies_the_region_residual_only_where_its_beta_fires(skill_db):
    """``predict`` boosts the stronger pool cross-tier; zero-beta majors and same pool hold."""
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)  # league is None -> base chain
    d = date(2026, 4, 14)
    p_base = predict(model, idx, "AAA", "DDD", as_of=d)
    tiers = {1: 0.0, 2: -1.0, 3: -2.0}

    # LCK (tier1) vs LFL (tier2): cross-tier -> the residual fires and lifts the major's side
    model.league = RegionRatingTable(
        r={"LCK": 0.4, "LFL": -0.9},
        tier_effect=tiers,
        home={"AAA|2026": "LCK", "DDD|2026": "LFL"},
        beta_cross_tier=0.5,
    )
    assert predict(model, idx, "AAA", "DDD", as_of=d) > p_base  # AAA's tier outranks

    # LEC (tier1) vs LCK (tier1): the served recipe holds within-tier-1 at beta 0 -> identity
    model.league = RegionRatingTable(
        r={"LEC": -0.3, "LCK": 0.4},
        tier_effect=tiers,
        home={"AAA|2026": "LEC", "DDD|2026": "LCK"},
        beta_cross_tier=0.5,
    )
    assert predict(model, idx, "AAA", "DDD", as_of=d) == p_base  # majors -> identity

    model.league = RegionRatingTable(
        r={"LCK": 0.4},
        tier_effect=tiers,
        home={"AAA|2026": "LCK", "DDD|2026": "LCK"},
        beta_cross_tier=0.5,
        beta_within_tier1=0.5,
    )
    assert predict(model, idx, "AAA", "DDD", as_of=d) == p_base  # same pool -> identity


class _StaleLineup:
    """Duck-typed lineup index: every team idle 999 days -> the dampener always fires."""

    def days_since_last_game(self, _team, _as_of):
        return 999.0


class _CurrentLineup:
    """Duck-typed lineup index: every team current -> the dampener is identity."""

    def days_since_last_game(self, _team, _as_of):
        return 1.0


def test_adjust_map_prob_applies_temperature_before_the_dampener():
    """Serve order is base -> cross-region -> temperature -> dampener: the dampener sits on top."""
    p = 0.72
    idx = {"lineup": _StaleLineup()}
    result = adjust_map_prob(
        p,
        None,
        idx,
        "AAA",
        "DDD",
        date(2026, 6, 20),
        temperature=SERVED_TEMPERATURE,
        damp_facts_games=False,
    )
    assert result == dampen_if_stale(apply_temperature(p, SERVED_TEMPERATURE), 999.0, 999.0)
    assert result != dampen_if_stale(p, 999.0, 999.0)  # temperature actually fired


def test_adjust_map_prob_damps_a_facts_game_only_when_its_recipe_does():
    """A facts game's rating reads its fielded fives, so a current recipe leaves it undamped."""
    p, idx = 0.72, {"lineup": _StaleLineup()}
    game = AsOf.game("2026-06-20T10:00:00+00:00", "ser_1", 1)
    tempered = apply_temperature(p, SERVED_TEMPERATURE)
    for damp, expected in ((False, tempered), (True, dampen_if_stale(tempered, 999.0, 999.0))):
        got = adjust_map_prob(
            p, None, idx, "AAA", "DDD", game, temperature=SERVED_TEMPERATURE, damp_facts_games=damp
        )
        assert got == expected


def test_adjust_map_prob_temperature_only_when_current():
    """A current matchup (dampener identity) returns exactly the temperature-scaled prob."""
    p = 0.72
    idx = {"lineup": _CurrentLineup()}
    assert adjust_map_prob(
        p,
        None,
        idx,
        "AAA",
        "DDD",
        date(2026, 6, 20),
        temperature=SERVED_TEMPERATURE,
        damp_facts_games=False,
    ) == apply_temperature(p, SERVED_TEMPERATURE)


def test_predict_uses_the_fixed_production_temperature(skill_db):
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)  # no league, no lineup -> the chain is base -> temperature
    ab = feature_computation(idx, "AAA", "DDD", date(2026, 4, 14), features=model.feature_names)
    ba = feature_computation(idx, "DDD", "AAA", date(2026, 4, 14), features=model.feature_names)
    base = base_score(model, ab, ba)

    assert predict(model, idx, "AAA", "DDD", as_of=date(2026, 4, 14)) == apply_temperature(
        base, SERVED_TEMPERATURE
    )


# --- the batched serve chain: base_score_batch / predict_batch (the in-sample-backfill scorer) ---


def test_base_score_batch_matches_per_row_blend(skill_db):
    """base_score_batch blends each row toward its own rating logit, float-identical to per-row."""
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)
    model.rating_blend = 0.7
    d = date(2026, 4, 14)
    feats_ab, feats_ba = [], []
    for k, (a, b) in enumerate([("AAA", "DDD"), ("BBB", "CCC"), ("DDD", "AAA"), ("CCC", "BBB")]):
        ab = feature_computation(idx, a, b, d, features=model.feature_names)
        ba = feature_computation(idx, b, a, d, features=model.feature_names)
        z = 0.3 * (k + 1)  # distinct, antisymmetric target
        feats_ab.append({**ab, "rating_logit_diff": z})
        feats_ba.append({**ba, "rating_logit_diff": -z})

    batched = base_score_batch(model, feats_ab, feats_ba)
    per_row = [base_score(model, ab, ba) for ab, ba in zip(feats_ab, feats_ba, strict=True)]
    assert batched == per_row  # bit-for-bit, blend included
    assert any(p != 0.5 for p in batched)  # non-vacuous


def test_predict_batch_equals_predict_per_game(skill_db):
    """predict_batch reproduces per-game predict across fresh and stale matchups, both ways."""
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served_with_lineup(conn)
    model.league = RegionRatingTable(
        r={"LCK": 0.4, "LFL": -0.9},
        tier_effect={1: 0.0, 2: -1.0, 3: -2.0},
        home={
            f"{t}|2026": ("LCK" if t in ("AAA", "BBB") else "LFL")
            for t in ("AAA", "BBB", "CCC", "DDD")
        },
        beta_cross_tier=0.5,
    )

    teams = ["AAA", "BBB", "CCC", "DDD"]
    matchups = [
        (a, b, d)
        for d in (date(2026, 4, 14), date(2026, 6, 20))  # fresh, then ~66d stale
        for a in teams
        for b in teams
        if a != b
    ]
    batched = predict_batch(model, idx, matchups)
    one_by_one = [predict(model, idx, a, b, as_of=d) for a, b, d in matchups]

    assert len(batched) == len(matchups) > 0
    assert all(abs(x - y) <= 1e-12 for x, y in zip(batched, one_by_one, strict=True))
    assert predict_batch(model, idx, []) == []  # empty batch is a clean no-op


def test_served_backfill_rows_match_serve_predict_per_game(skill_db):
    """The batched backfill scorer matches canonical per-game serving exactly."""
    from league_predictor.predraft.pipeline import operations as served_ops

    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served_with_lineup(conn)
    model.league = RegionRatingTable(
        r={"LCK": 0.4, "LFL": -0.9},
        tier_effect={1: 0.0, 2: -1.0, 3: -2.0},
        home={
            f"{t}|2026": ("LCK" if t in ("AAA", "BBB") else "LFL")
            for t in ("AAA", "BBB", "CCC", "DDD")
        },
        beta_cross_tier=0.5,
    )

    scores = served_ops.score_backfill(conn, model, indexes=idx)
    games = served_ops.dated_games(conn)
    assert len(scores) == len(games) > 0
    for score, g in zip(scores, games, strict=True):
        at = AsOf.game(g["scheduled_ts"], g["series_id"], g["game_no"])
        ref = predict(model, idx, g["team_a"], g["team_b"], as_of=at)
        assert score.game_id == g["game_id"]
        assert abs(score.p_a - ref) <= 1e-12


def test_predict_is_fixed_temperature_image_of_game_prob(skill_db):
    conn = skill_db(start=date(2025, 6, 1), end=date(2026, 4, 15))
    model, idx = _served(conn)  # no league layer -> predict is the bare base
    p_predict = predict(model, idx, "AAA", "DDD", as_of=date(2026, 4, 14))
    p_gameprob = game_prob(model.bag, idx, "AAA", "DDD", date(2026, 4, 14), families=model.families)
    assert p_predict == apply_temperature(p_gameprob, SERVED_TEMPERATURE)


def test_the_series_lead_shifts_the_blended_base_by_its_weight_in_logits():
    """The base adds ``weight * lead`` after the rating blend; a zero weight is bit-exact."""
    from league_predictor.probability import logit

    ab = {"rating_logit_diff": 0.4, "diff_series_wins": 1.0}
    ba = {"rating_logit_diff": -0.4, "diff_series_wins": -1.0}
    blended = blend_toward_rating(0.6, rating_logit(ab, ba), 0.36)
    assert blended_base(0.6, ab, ba, 0.36) == blended
    shifted = blended_base(0.6, ab, ba, 0.36, 0.16)
    assert math.isclose(logit(shifted) - logit(blended), 0.16, abs_tol=1e-9)
    assert math.isclose(blended_base(0.4, ba, ab, 0.36, 0.16), 1.0 - shifted, abs_tol=1e-12)
