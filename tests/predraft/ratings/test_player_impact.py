"""Preparation and fit pins for the roster-change Impact estimator (player_impact/)."""

from __future__ import annotations

import numpy as np
import pytest

from league_predictor.predraft.ratings.player_impact import ROSTER_SIZE, game_eta
from league_predictor.predraft.ratings.player_impact.design import (
    _ALPHA_KEY,
    _BETA_KEY,
    _GAMMA_KEY,
)
from league_predictor.predraft.ratings.player_impact.results import _value_to_pp
from tests.predraft.ratings._impact_helpers import _facts_conn, _seed_game, _zero_priors


def _seeded_conn():
    """One retired 2023 pairing plus 25 recent AA-vs-BB games."""
    from datetime import date, timedelta

    conn = _facts_conn()
    _seed_game(conn, gid="old", day="2023-01-01", team1="OLD", team2="GONE", winner="OLD")
    for index in range(25):
        day = (date(2026, 4, 1) + timedelta(days=index)).isoformat()
        _seed_game(
            conn,
            gid=f"recent-{index}",
            day=day,
            team1="AA",
            team2="BB",
            winner="AA" if index % 2 == 0 else "BB",
        )
    return conn


def test_full_history_fit_preserves_recent_qualification_and_every_row_influences(monkeypatch):
    """The current recipe uses all prior games without publishing retired-only players."""
    from datetime import date

    from league_predictor.predraft.ratings.impact_corpus import build_impact_corpus
    from league_predictor.predraft.ratings.player_impact import fit as impact_fit
    from league_predictor.predraft.ratings.player_impact import prepare_impact_fit

    conn = _seeded_conn()
    as_of = date(2026, 5, 1)
    corpus = build_impact_corpus(conn)
    monkeypatch.setattr(impact_fit, "_stat_context_priors", _zero_priors)
    full = prepare_impact_fit(conn, as_of, corpus=corpus)

    assert len(full.games) == 26
    assert not any(player.startswith(("OLD_", "GONE_")) for player in full.qualified)
    assert any(key.startswith("__history_") for key in full.col_index)
    # Blue side enters as the fitted beta column, never a constant side-bias column.
    assert "__side_bias__" not in full.col_index
    assert {_GAMMA_KEY, _ALPHA_KEY, _BETA_KEY} <= set(full.col_index)
    # All three fit columns are exempt: shrinking the prior scale, the region weight, or the
    # side weight would bias calibration toward zero.
    for key in (_GAMMA_KEY, _ALPHA_KEY, _BETA_KEY):
        assert full.penalty_mask[full.col_index[key]] == 0.0
    assert full.penalty_mask[full.col_index["AA_mid"]] == 1.0
    assert np.all(full.sample_weight > 0.0)
    assert all(full.X.getrow(row).nnz > 0 for row in range(len(full.games)))

    _seed_game(conn, gid="future", day="2027-01-01", team1="NEW", team2="LATER", winner="NEW")
    future_corpus = build_impact_corpus(conn)
    after_future_mutation = prepare_impact_fit(conn, as_of, corpus=future_corpus)
    assert after_future_mutation.qualified == full.qualified
    assert after_future_mutation.col_index == full.col_index
    assert np.array_equal(after_future_mutation.X.toarray(), full.X.toarray())
    assert np.array_equal(after_future_mutation.y, full.y)
    assert np.array_equal(after_future_mutation.sample_weight, full.sample_weight)

    conn.close()


def _varied_priors(_conn, _as_of, player_ids, **_):
    """Deterministic NON-ZERO stat priors, so the fitted gamma column carries real signal."""
    return {pid: (sum(ord(c) for c in pid) % 7 - 3.5) / 10.0 for pid in player_ids}


#: Two teams per league so every domestic pairing is unambiguous; ``NACL`` normalizes to no
#: region, so its players are the fixture's tier-anchored (unresolved) side.
_PARITY_LEAGUES = (
    ("LCK", "AA", "A2"),
    ("LEC", "BB", "B2"),
    ("PCS", "CC", "C2"),
    ("NACL", "DD", "D2"),
)


def _parity_conn():
    """A production-shaped corpus: four leagues, cross-region play, replacements, both sides.

    Returns:
        Connection whose 26 domestic games per league fix each player's region, whose eight
        international games (excluded from region resolution) are the cross-region rows that
        identify alpha, whose team ``ZZ`` plays twice so replacement columns exist, and whose
        blue side alternates so beta is identified in both directions.
    """
    from datetime import date, timedelta

    conn = _facts_conn()
    for index in range(26):
        day = (date(2026, 1, 5) + timedelta(days=index * 4)).isoformat()
        for league, home, away in _PARITY_LEAGUES:
            _seed_game(
                conn,
                gid=f"{league}-{index}",
                day=day,
                team1=home,
                team2=away,
                winner=home if index % 3 else away,
                league=league,
                blue=home if index % 2 else away,
            )
    for index, (home, away) in enumerate(
        (
            ("AA", "BB"),
            ("AA", "CC"),
            ("BB", "DD"),
            ("CC", "D2"),
            ("A2", "B2"),
            ("A2", "C2"),
            ("B2", "DD"),
            ("CC", "AA"),
        )
    ):
        _seed_game(
            conn,
            gid=f"WLDs-{index}",
            day=(date(2026, 4, 20) + timedelta(days=index)).isoformat(),
            team1=home,
            team2=away,
            winner=home if index % 2 else away,
            league="WLDs",
            blue=away if index % 2 else home,
        )
    for index in range(26):  # GG fields three players — the narrow lineups the facts carry
        _seed_game(
            conn,
            gid=f"short-{index}",
            day=(date(2026, 1, 7) + timedelta(days=index * 4)).isoformat(),
            team1="FF",
            team2="GG",
            winner="FF" if index % 3 else "GG",
            league="LEC",
            blue="GG" if index % 2 else "FF",
            roles={"GG": ("top", "jng", "mid")},
        )
    for index in range(2):  # ZZ is sub-threshold: its five players share the replacement columns
        _seed_game(
            conn,
            gid=f"sub-{index}",
            day=(date(2026, 2, 1) + timedelta(days=index)).isoformat(),
            team1="AA",
            team2="ZZ",
            winner="AA",
            league="LCK",
        )
    return conn


def _parity_fit(monkeypatch):
    """Prepare and fit the production recipe over ``_parity_conn``; returns ``(context, fit)``."""
    from datetime import date

    from league_predictor.predraft.ratings.player_impact import fit as impact_fit
    from league_predictor.predraft.ratings.player_impact import (
        fit_prepared_impact,
        prepare_impact_fit,
    )

    conn = _parity_conn()
    monkeypatch.setattr(impact_fit, "_stat_context_priors", _varied_priors)
    context = prepare_impact_fit(conn, date(2026, 5, 15))
    conn.close()
    return context, fit_prepared_impact(context, lam=1.0)


def test_the_parity_fixture_exercises_every_term_of_the_predictor(monkeypatch):
    """Guard the parity fixture itself: a degenerate corpus would pass any scorer."""
    context, fit = _parity_fit(monkeypatch)

    assert abs(fit.gamma) > 1e-6
    assert abs(fit.alpha) > 1e-6
    assert abs(fit.beta) > 1e-6
    assert fit.values.replacement_value, "no sub-threshold player, so no replacement column"
    assert {g.blue_sign for g in context.games} == {-1, 1}
    regions = {context.player_region.get(p, "") for p in context.qualified}
    assert len({r for r in regions if r}) >= 2
    assert "" in regions
    assert any(context.region.gap(g.team1_players, g.team2_players) for g in context.games)
    widths = {len(g.team1_players) for g in context.games} | {
        len(g.team2_players) for g in context.games
    }
    assert widths - {ROSTER_SIZE}, "no sub-five lineup, so the share denominator is untested"
    assert all(context.skill_priors.get(p, 0.0) for p in context.qualified)


def test_scorer_eta_equals_the_fit_design_row(monkeypatch):
    """THE criterion: every design row's eta is exactly what ``game_eta`` returns for it."""
    from league_predictor.predraft.ratings.player_impact.design import _fit_masked_ridge

    context, fit = _parity_fit(monkeypatch)
    theta = _fit_masked_ridge(
        context.X, context.y, context.sample_weight, lam=1.0, penalty_mask=context.penalty_mask
    )
    design_eta = context.X @ theta

    for index, game in enumerate(context.games):
        scored = game_eta(fit.values, game.team1_players, game.team2_players, game.blue_sign)
        assert scored == pytest.approx(float(design_eta[index]), abs=1e-12), (
            f"game {index}: scorer {scored} != design row {design_eta[index]}"
        )


def test_published_values_re_sum_to_the_eta_the_predictor_gives_the_game(monkeypatch):
    """The published decomposition reconstructs eta: no candidate-vs-realized gap can open."""
    context, fit = _parity_fit(monkeypatch)
    published = {pid: estimate.value_logodds for pid, estimate in fit.scores.items()}
    checked = 0

    short = 0
    for game in context.games:
        roster_a, roster_b = game.team1_players, game.team2_players
        if not all(p in published for p in roster_a + roster_b):
            continue  # a roster with a replacement player publishes no value for them
        short += len(roster_a) < ROSTER_SIZE or len(roster_b) < ROSTER_SIZE
        region_term = fit.alpha * context.region.gap(roster_a, roster_b)
        shares_a = sum(fit.alpha * context.region.share(p) for p in roster_a)
        shares_b = sum(fit.alpha * context.region.share(p) for p in roster_b)
        assert shares_a - shares_b == pytest.approx(region_term, abs=1e-12)

        eta = game_eta(fit.values, roster_a, roster_b, game.blue_sign)
        published_gap = sum(published[p] for p in roster_a) - sum(published[p] for p in roster_b)
        assert published_gap == pytest.approx(eta - fit.beta * game.blue_sign, abs=1e-12)
        checked += 1
    assert checked >= 100, f"only {checked} games exercised the re-sum property"
    assert short >= 4, f"only {short} sub-five lineups checked; the denominator is untested"


def test_evaluator_reconstructs_a_persisted_recipe(monkeypatch):
    """The evaluator-private path uses persisted windows instead of source defaults."""
    from datetime import date

    from league_predictor.predraft.ratings.impact_corpus import build_impact_corpus
    from league_predictor.predraft.ratings.player_impact import ImpactRecipe
    from league_predictor.predraft.ratings.player_impact import fit as impact_fit
    from league_predictor.predraft.ratings.player_impact.fit import (
        _prepare_impact_fit_for_recipe,
    )

    conn = _seeded_conn()
    corpus = build_impact_corpus(conn)
    recipe = ImpactRecipe(
        version=9,
        influence_window_months=24,
        qualification_window_months=None,
        half_life_days=300.0,
        min_games=25,
        history_buckets_per_role=64,
    )
    monkeypatch.setattr(impact_fit, "_stat_context_priors", _zero_priors)

    context = _prepare_impact_fit_for_recipe(conn, date(2026, 5, 1), recipe, corpus=corpus)

    assert len(context.games) == 25  # the persisted 24-month window, not the source default
    assert not any(player.startswith(("OLD_", "GONE_")) for player in context.qualified)
    assert not any(key.startswith("__history_") for key in context.col_index)
    assert context.sample_weight[0] == pytest.approx(
        0.5 ** (context.games[0].days_before_asof / 300.0)
    )
    conn.close()


# ── DB-facing smoke test ─────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.real_original_cache  # live fit reads the configured, persisted stat-context weights
def test_fit_player_impact_smoke_and_roundtrip():
    """fit_player_impact runs read-only on the live facts DB (or ``REBUILD_SMOKE_DB``)."""
    import json
    import os
    import sqlite3
    from datetime import date
    from pathlib import Path

    from league_predictor.config import DB_PATH
    from league_predictor.predraft.ratings.player_impact import fit_player_impact

    smoke = os.environ.get("REBUILD_SMOKE_DB")
    if smoke and Path(smoke).exists():
        target = Path(smoke)
    elif Path(DB_PATH).exists():
        target = Path(DB_PATH)
    else:
        pytest.skip("no REBUILD_SMOKE_DB and no live DB available")

    conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"games", "game_players", "series"}.issubset(have):
        conn.close()
        pytest.skip("target DB missing the facts tables")
    fit = fit_player_impact(conn, date(2026, 5, 1))

    assert fit.scores, "expected at least one qualified player"
    r = next(iter(fit.scores.values()))
    # V_p = gamma * prior + delta + the player's fifth of the team region term.
    share = fit.alpha * fit.values.region.share(r.player_id)
    assert abs(r.value_logodds - (fit.gamma * r.prior_logodds + r.delta + share)) < 1e-6
    assert 0.0 <= r.percentile <= 100.0
    assert r.grade in {"S", "A", "B", "C", "D"}
    assert r.results_pp == pytest.approx(r.impact_pp - r.stats_pp)

    # Impact reads as lift over the average premier starter of the role, so each role's
    # premier mean sits at exactly 0 pp and both signs occur.
    for role, mean in fit.premier_role_means.items():
        assert _value_to_pp(mean, mean, fit.premier_mean_logodds) == pytest.approx(0.0)
        role_pp = [e.impact_pp for e in fit.scores.values() if e.role == role]
        assert min(role_pp) < 0.0 < max(role_pp), f"{role} should straddle its reference"

    # to_artifact serializes cleanly and preserves the qualified-player set.
    payload = json.loads(json.dumps(fit.to_artifact()))
    assert set(payload["players"]) == set(fit.scores)
    assert payload["players"][r.player_id]["impact_pp"] == pytest.approx(round(r.impact_pp, 2))
    conn.close()
