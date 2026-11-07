"""Tests for the one context-adjusted stat home (stat_context/), term by term."""

# file-length-ok: one home's contract mirror — every term of one grid, probed against one
# fixture family; splitting it scatters the shared seeders across cross-file hops.

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest

from league_predictor.predraft.ratings.stat_context import (
    GRID_EPOCH,
    GRID_STEP_DAYS,
    PILLARS,
    STATS,
    ContextZ,
    aggregate_prior,
    build_game_context_z,
    game_kda,
    walk_forward_pillar_aggregates,
)

_THROUGH = date(2026, 7, 1)
_PROBE_DAY = date(2026, 6, 15)
_WINDOW_START = date(2026, 1, 1)
#: Start of the probe's grid period — the earliest same-period day the era lock must cover.
_PROBE_PERIOD_START = _PROBE_DAY - timedelta(days=(_PROBE_DAY - GRID_EPOCH).days % GRID_STEP_DAYS)


def _conn():
    import sqlite3

    from league_predictor.db.schema import apply_current_schema

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    return conn


def _seed_team_state(conn, team, day, strength):
    conn.execute("INSERT OR IGNORE INTO models (model_id, kind) VALUES ('rating:test', 'rating')")
    conn.execute(
        "INSERT OR REPLACE INTO team_state (team_id, model_id, asof_date, strength) "
        "VALUES (?, 'rating:test', ?, ?)",
        (team, day.isoformat(), strength),
    )


_STAT_COLUMNS = (
    "golddiffat15",
    "csdiffat15",
    "xpdiffat15",
    "golddiffat10",
    "csdiffat10",
    "xpdiffat10",
    "cspm",
    "dpm",
    "killparticipation",
    "kills",
    "deaths",
    "assists",
    "firstblood",
    "vspm",
    "earnedgoldshare",
)


def _seed_game(
    conn,
    gid,
    day,
    *,
    league="LCK",
    team1="AAA",
    team2="BBB",
    winner=None,
    blue=None,
    rows=(),
    blind_roles=(),
):
    """One one-game series; ``rows`` are (player_id, team, role, {stat: value}).

    Args:
        blind_roles: (side, role) pairs whose pick lands in the game's first two
            ``draft_actions`` pick slots, marking that appearance as a blind pick.
    """
    team_a, team_b = sorted((team1, team2))
    winner = winner or team1
    day_iso = day.isoformat() if isinstance(day, date) else day
    conn.execute("INSERT OR IGNORE INTO leagues (league_id, name) VALUES (?, ?)", (league, league))
    for t in (team_a, team_b):
        conn.execute("INSERT OR IGNORE INTO teams (team_id) VALUES (?)", (t,))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
        "best_of, winner, status) VALUES (?, ?, ?, ?, ?, 1, ?, 'completed')",
        (f"ser_{gid}", league, team_a, team_b, f"{day_iso}T10:00:00+00:00", winner),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner, blue_team) VALUES (?, ?, 1, ?, ?)",
        (gid, f"ser_{gid}", winner, blue),
    )
    for player_id, team, role, stats in rows:
        conn.execute("INSERT OR IGNORE INTO players (player_id) VALUES (?)", (player_id,))
        champion = stats.get("champion_id")
        columns = {name: stats.get(name) for name in _STAT_COLUMNS}
        conn.execute(
            "INSERT INTO game_players (game_id, team_id, player_id, role, result, champion_id, "
            + ", ".join(columns)
            + ") VALUES (?, ?, ?, ?, ?, ?, "
            + ", ".join("?" * len(columns))
            + ")",
            (gid, team, player_id, role, 1 if team == winner else 0, champion, *columns.values()),
        )
    if blind_roles:
        ordered = list(blind_roles)
        for side in ("blue", "red"):
            ordered.extend((side, role) for role in ("top", "jng", "mid", "bot", "sup"))
        seen: set[tuple[str, str]] = set()
        for seq, (side, role) in enumerate(
            (pair for pair in ordered if not (pair in seen or seen.add(pair))), start=1
        ):
            conn.execute(
                "INSERT INTO draft_actions (game_id, seq, side, kind, champion_id, role) "
                "VALUES (?, ?, ?, 'pick', ?, ?)",
                (gid, seq, side, f"champ{seq}", role),
            )
    conn.commit()


def _base_conn():
    """A connection whose team_state gate is satisfied with two neutral teams."""
    conn = _conn()
    _seed_team_state(conn, "AAA", date(2025, 12, 1), 0.0)
    _seed_team_state(conn, "BBB", date(2025, 12, 1), 0.0)
    return conn


def _window_days(n):
    """n distinct days between the window start and the probe's grid period."""
    span = (date(2026, 4, 30) - _WINDOW_START).days
    return [_WINDOW_START + timedelta(days=(i * span) // max(n - 1, 1)) for i in range(n)]


def _seed_mid_window(conn, value_of, n=40, *, league="LCK", extra=None):
    """n window games, one mid per side: the winner row gets ``value_of('win', i)`` dpm."""
    for i, day in enumerate(_window_days(n)):
        rows = [
            (f"w{league}{i}", "AAA", "mid", {"dpm": value_of("win", i), **(extra or {})}),
            (f"l{league}{i}", "BBB", "mid", {"dpm": value_of("loss", i), **(extra or {})}),
        ]
        _seed_game(conn, f"g_{league}_{i}", day, league=league, winner="AAA", rows=rows)


def _probe(conn, stats, *, role="mid", league="LCK", winner=True, blue=None, blind=False):
    """Seed one probe game on the probe day; the probe player sits on team AAA."""
    rows = [("probe", "AAA", role, stats), ("probe_opp", "BBB", role, {})]
    _seed_game(
        conn,
        f"probe_{role}",
        _PROBE_DAY,
        league=league,
        winner="AAA" if winner else "BBB",
        blue=blue,
        rows=rows,
        blind_roles=((("blue" if blue == "AAA" else "red"), role),) if blind else (),
    )
    return "probe"


def _appearance_index(ctx: ContextZ, player: str, day: date) -> int:
    pidx = ctx.player_ids.index(player)
    mask = (ctx.appearance_players == pidx) & (ctx.appearance_dates == day.toordinal())
    (idx,) = np.flatnonzero(mask)
    return int(idx)


def _z(ctx: ContextZ, player: str, stat: str, day: date = _PROBE_DAY) -> float:
    return float(ctx.stat_z[_appearance_index(ctx, player, day), STATS.index(stat)])


def _pillar(ctx: ContextZ, player: str, pillar: str, day: date = _PROBE_DAY) -> float:
    return float(ctx.pillar_z[_appearance_index(ctx, player, day), PILLARS.index(pillar)])


def test_exports_and_through_bound():
    """Appearances strictly before ``through`` enter; later ones do not."""
    conn = _base_conn()
    _seed_mid_window(conn, lambda *_: 500.0)
    _seed_game(conn, "late", _THROUGH, rows=[("late_p", "AAA", "mid", {"dpm": 500.0})])
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()

    n = len(ctx.appearance_players)
    assert len(ctx.appearance_dates) == len(ctx.appearance_roles) == n
    assert ctx.stat_z.shape == (n, len(STATS))
    assert ctx.pillar_z.shape == (n, len(PILLARS))
    if "late_p" in ctx.player_ids:
        assert not (ctx.appearance_players == ctx.player_ids.index("late_p")).any()


def test_outcome_stratified_mu_prices_deviation_from_the_outcome_mean():
    """A winner is measured against typical winners, and z winsorizes at ±4."""

    def values(outcome, i):
        return (590.0, 610.0)[i % 2] if outcome == "win" else (390.0, 410.0)[i % 2]

    conn = _base_conn()
    _seed_mid_window(conn, values)
    _probe(conn, {"dpm": 605.0}, winner=True)
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()
    assert _z(ctx, "probe", "dpm") == pytest.approx(0.5, abs=0.02)

    conn = _base_conn()
    _seed_mid_window(conn, values)
    _probe(conn, {"dpm": 400.0}, winner=True)
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()
    assert _z(ctx, "probe", "dpm") == -4.0  # (400-600)/10 = -20, winsorized


def test_covariate_variant_pools_the_outcome_gap_across_tiers():
    """Stratified mu is per-tier; the covariate variant prices one pooled outcome gap."""

    def tier1(outcome, i):
        return (590.0, 610.0)[i % 2] if outcome == "win" else (390.0, 410.0)[i % 2]

    conn = _base_conn()
    _seed_mid_window(conn, tier1)
    _seed_mid_window(conn, lambda _o, i: (490.0, 510.0)[i % 2], league="TCL")
    _probe(conn, {"dpm": 600.0}, winner=True)
    covariate = build_game_context_z(conn, _THROUGH)
    conn.close()

    # Pooled gap b = 100: E = 500 + 100*(1 - 0.5) = 550; residual sigma(t1 mid) = sqrt(2600).
    assert _z(covariate, "probe", "dpm") == pytest.approx(50.0 / np.sqrt(2600.0), abs=0.02)


def test_side_effect_is_removed_before_z():
    """A blue-side laning diff is measured against the blue-side expectation."""
    conn = _base_conn()
    for i, day in enumerate(_window_days(40)):
        blue_gd = 100.0 + (-10.0, 10.0)[(i // 2) % 2]
        rows = [
            (f"b{i}", "AAA", "mid", {"golddiffat15": blue_gd}),
            (f"r{i}", "BBB", "mid", {"golddiffat15": -blue_gd}),
        ]
        _seed_game(conn, f"side_{i}", day, winner="AAA" if i % 2 else "BBB", blue="AAA", rows=rows)
    _probe(conn, {"golddiffat15": 90.0}, blue="AAA")
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()

    # The blue expectation is +100 with residual sigma 10: 90 on blue reads -1, not +0.9.
    assert _z(ctx, "probe", "golddiffat15") == pytest.approx(-1.0, abs=0.05)


def test_blind_pick_deficit_is_priced():
    """A blind pick is measured against the blind expectation, not the pooled mean."""
    conn = _base_conn()
    for i, day in enumerate(_window_days(40)):
        blind_side = "blue" if i % 4 < 2 else "red"
        jitter = (-10.0, 10.0)[(i // 4) % 2]
        blue_gd = (-50.0 if blind_side == "blue" else 50.0) + jitter
        rows = [
            (f"b{i}", "AAA", "mid", {"golddiffat15": blue_gd}),
            (f"r{i}", "BBB", "mid", {"golddiffat15": -blue_gd + 2 * jitter}),
        ]
        _seed_game(
            conn,
            f"blind_{i}",
            day,
            winner="AAA" if i % 2 else "BBB",
            blue="AAA",
            rows=rows,
            blind_roles=((blind_side, "mid"),),
        )
    _probe(conn, {"golddiffat15": -40.0}, blue="AAA", blind=True)
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()

    # The blind expectation is about -50 after the side pass; -40 reads near +1 sigma.
    assert _z(ctx, "probe", "golddiffat15") == pytest.approx(1.0, abs=0.35)


def _seed_opponent_history(conn):
    """64 window games where the AAA mid's gd15 tracks opponent strength (alpha = -50)."""
    _seed_team_state(conn, "AAA", date(2025, 12, 1), 0.0)
    _seed_team_state(conn, "STRONG", date(2025, 12, 1), 1.0)
    _seed_team_state(conn, "WEAK", date(2025, 12, 1), -1.0)
    for i, day in enumerate(_window_days(64)):
        opp = "STRONG" if i % 2 else "WEAK"
        gd = (-50.0 if opp == "STRONG" else 50.0) + (-5.0, 5.0)[(i // 4) % 2]
        winner = "AAA" if i % 4 in (1, 2) else opp
        rows = [
            (f"m{i}", "AAA", "mid", {"golddiffat15": gd}),
            (f"o{i}", opp, "mid", {}),
        ]
        _seed_game(conn, f"opp_{i}", day, team2=opp, winner=winner, rows=rows)
    _seed_game(
        conn,
        "probe_game",
        _PROBE_DAY,
        team2="STRONG",
        winner="AAA",
        rows=[("probe", "AAA", "mid", {"golddiffat15": -45.0}), ("po", "STRONG", "mid", {})],
    )


def test_opponent_strength_credit():
    """Facing a strong opponent raises the expectation-adjusted z, per the fitted slope."""
    conn = _conn()
    _seed_opponent_history(conn)
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()

    # alpha = -50 per unit of strength; vs +1.0 the expectation is -50, so -45 reads ~+1 sigma.
    assert _z(ctx, "probe", "golddiffat15") == pytest.approx(1.0, abs=0.35)


def test_team_state_reads_strictly_before_the_game_day():
    """A team_state row stamped ON the game day incorporates that day — it must not be read."""
    conn = _conn()
    _seed_opponent_history(conn)
    baseline = _z(build_game_context_z(conn, _THROUGH), "probe", "golddiffat15")

    _seed_team_state(conn, "STRONG", _PROBE_DAY, -50.0)  # absurd same-day value
    same_day = _z(build_game_context_z(conn, _THROUGH), "probe", "golddiffat15")
    conn.close()

    assert same_day == baseline


def test_at10_fallback_scales_by_the_locked_constant():
    """A NULL @15 diff falls back to the @10 diff scaled by 1.33."""
    conn = _base_conn()
    for i, day in enumerate(_window_days(40)):
        gd = 50.0 + (-10.0, 10.0)[(i // 2) % 2]
        rows = [
            (f"w{i}", "AAA", "mid", {"golddiffat15": gd}),
            (f"l{i}", "BBB", "mid", {"golddiffat15": -gd}),
        ]
        _seed_game(conn, f"at10_{i}", day, winner="AAA", rows=rows)
    _probe(conn, {"golddiffat15": None, "golddiffat10": 30.0}, winner=True)
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()

    # E(win) = +50, sigma = 10; the probe prices 1.33 * 30 = 39.9 -> z ~ -1.01.
    assert _z(ctx, "probe", "golddiffat15") == pytest.approx((1.33 * 30.0 - 50.0) / 10.0, abs=0.05)


def _seed_lane_bt_history(conn, role="mid"):
    """40 window games: one ``role`` star S beats one-off fodder by +50, W loses by -50."""
    for i, day in enumerate(_window_days(40)):
        jitter = (-10.0, 10.0)[(i // 4) % 2]
        star, gd = ("S", 50.0 + jitter) if i % 2 else ("W", -50.0 + jitter)
        rows = [
            (star, "AAA", role, {"golddiffat15": gd}),
            (f"fodder{i}", "BBB", role, {"golddiffat15": -gd}),
        ]
        _seed_game(conn, f"bt_{i}", day, winner="AAA" if (i // 2) % 2 else "BBB", rows=rows)


def _seed_lane_bt_probes(conn, role="mid"):
    for team, player, opp in (("P1T", "P1", "S"), ("P2T", "P2", "W")):
        _seed_game(
            conn,
            f"probe_vs_{opp}",
            _PROBE_DAY,
            team1=team,
            team2="AAA",
            winner=team,
            rows=[
                (player, team, role, {"golddiffat15": 0.0}),
                (opp, "AAA", role, {"golddiffat15": 0.0}),
            ],
        )


def test_lane_bt_credits_beating_a_strong_lane_opponent():
    """v8: the same even lane reads higher against a proven strong laner."""
    conn = _base_conn()
    _seed_lane_bt_history(conn)
    _seed_lane_bt_probes(conn)
    with_bt = build_game_context_z(conn, _THROUGH, lane_bt=True)
    without = build_game_context_z(conn, _THROUGH, lane_bt=False)
    conn.close()

    assert _pillar(with_bt, "P1", "laning") > 0.25
    assert _pillar(with_bt, "P2", "laning") < -0.25
    assert abs(_pillar(without, "P1", "laning") - _pillar(without, "P2", "laning")) < 0.1


def test_lane_bt_prices_the_sup_lane_pillar_too():
    """The sup 'lane' pillar rides the same opponent-adjusted margins as 'laning'."""
    conn = _base_conn()
    _seed_lane_bt_history(conn, role="sup")
    _seed_lane_bt_probes(conn, role="sup")
    with_bt = build_game_context_z(conn, _THROUGH, lane_bt=True)
    conn.close()

    assert _pillar(with_bt, "P1", "lane") > 0.25
    assert _pillar(with_bt, "P2", "lane") < -0.25


def test_empty_corpus_paths_return_empty_not_crash():
    conn = _conn()
    _seed_team_state(conn, "AAA", date(2025, 12, 1), 0.0)
    ctx = build_game_context_z(conn, _THROUGH, lane_bt=True)
    conn.close()

    assert len(ctx.appearance_players) == 0
    assert walk_forward_pillar_aggregates(ctx).shape == (0, len(PILLARS))
    assert aggregate_prior(ctx, _THROUGH) == {}


def test_a_half_life_that_would_overflow_is_refused():
    ctx = _hand_ctx(
        [
            ("p", date(2018, 1, 1), "mid", {"combat": 1.0}),
            ("p", date(2026, 1, 1), "mid", {"combat": -1.0}),
        ]
    )
    with pytest.raises(ValueError, match="half_life_days"):
        walk_forward_pillar_aggregates(ctx, half_life_days=1.0)


def test_lane_bt_reads_only_true_lane_diffs():
    """v8: cspm alone produces no laning value; the diffs are the lane."""
    conn = _base_conn()
    for i, day in enumerate(_window_days(40)):
        gd = (-50.0, 50.0)[i % 2]
        cspm = (4.0, 6.0)[(i // 2) % 2]
        rows = [
            (f"w{i}", "AAA", "mid", {"golddiffat15": gd, "cspm": cspm}),
            (f"l{i}", "BBB", "mid", {"golddiffat15": -gd, "cspm": cspm}),
        ]
        _seed_game(conn, f"csonly_{i}", day, winner="AAA", rows=rows)
    _probe(conn, {"cspm": 5.0})
    with_bt = build_game_context_z(conn, _THROUGH, lane_bt=True)
    without = build_game_context_z(conn, _THROUGH, lane_bt=False)
    conn.close()

    assert np.isnan(_pillar(with_bt, "probe", "laning"))
    assert not np.isnan(_pillar(without, "probe", "laning"))


def test_kda_per_game_is_bounded_on_zero_death_games():
    assert game_kda(10, 0, 5) == pytest.approx(15.0)
    assert game_kda(2, 4, 6) == pytest.approx(1.6)


def test_z_is_walk_forward_and_era_locked():
    """Same-period and later games never move an appearance's z."""

    def build(extra_games):
        conn = _base_conn()
        _seed_mid_window(
            conn, lambda o, i: (590.0, 610.0)[i % 2] if o == "win" else (390.0, 410.0)[i % 2]
        )
        _probe(conn, {"dpm": 605.0}, winner=True)
        for gid, day in extra_games:
            _seed_game(conn, gid, day, rows=[("outlier", "AAA", "mid", {"dpm": 9000.0})])
        ctx = build_game_context_z(conn, _THROUGH)
        conn.close()
        return _z(ctx, "probe", "dpm")

    base = build(())
    later = build((("after", _PROBE_DAY + timedelta(days=10)),))
    same_period = build((("sameper", _PROBE_PERIOD_START),))

    assert later == base
    assert same_period == base


def test_thin_cells_leave_the_appearance_absent():
    """A cell (and its role fallback) below the support floor produces no z at all."""
    conn = _base_conn()
    _seed_mid_window(conn, lambda _o, i: (490.0, 510.0)[i % 2], n=8)
    _probe(conn, {"dpm": 500.0})
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()

    assert np.isnan(_z(ctx, "probe", "dpm"))
    assert np.isnan(_pillar(ctx, "probe", "combat"))


def test_missing_stats_leave_other_pillar_stats_working():
    """A NULL stat is absent, and the pillar re-weights over the present stats."""
    conn = _base_conn()
    for i, day in enumerate(_window_days(40)):
        gd = (-50.0, 50.0)[i % 2]
        cspm = (4.0, 6.0)[(i // 2) % 2]
        rows = [
            (
                f"w{i}",
                "AAA",
                "mid",
                {
                    "dpm": (590.0, 610.0)[i % 2],
                    "golddiffat15": gd,
                    "csdiffat15": gd,
                    "xpdiffat15": gd,
                    "cspm": cspm,
                },
            ),
            (
                f"l{i}",
                "BBB",
                "mid",
                {
                    "dpm": (390.0, 410.0)[i % 2],
                    "golddiffat15": -gd,
                    "csdiffat15": -gd,
                    "xpdiffat15": -gd,
                    "cspm": cspm,
                },
            ),
        ]
        _seed_game(conn, f"mix_{i}", day, winner="AAA", rows=rows)
    _probe(conn, {"dpm": 605.0, "golddiffat15": 0.0, "csdiffat15": 0.0, "cspm": None})
    ctx = build_game_context_z(conn, _THROUGH)
    conn.close()

    assert np.isnan(_z(ctx, "probe", "cspm"))
    assert np.isnan(_z(ctx, "probe", "xpdiffat15"))
    assert not np.isnan(_pillar(ctx, "probe", "laning"))
    assert not np.isnan(_pillar(ctx, "probe", "combat"))


def test_team_state_is_required():
    conn = _conn()
    _seed_game(conn, "g", date(2026, 1, 1), rows=[("p", "AAA", "mid", {"dpm": 500.0})])
    with pytest.raises(RuntimeError, match="backfill rating"):
        build_game_context_z(conn, _THROUGH)
    conn.close()


def _hand_ctx(entries):
    """A ContextZ built directly from (player, day, role, {pillar: z}) rows."""
    players = sorted({entry[0] for entry in entries})
    stat_z = np.full((len(entries), len(STATS)), np.nan)
    pillar_z = np.full((len(entries), len(PILLARS)), np.nan)
    app_players, app_dates, app_roles = [], [], []
    for i, (player, day, role, pillars) in enumerate(entries):
        app_players.append(players.index(player))
        app_dates.append(day.toordinal())
        app_roles.append(("top", "jng", "mid", "bot", "sup").index(role))
        for name, value in pillars.items():
            pillar_z[i, PILLARS.index(name)] = value
    return ContextZ(
        player_ids=tuple(players),
        appearance_players=np.array(app_players, dtype=np.int32),
        appearance_dates=np.array(app_dates, dtype=np.int64),
        appearance_roles=np.array(app_roles, dtype=np.int16),
        stat_z=stat_z,
        pillar_z=pillar_z,
        through=_THROUGH,
    )


def test_aggregate_prior_recency_weighting_and_shrinkage():
    """z-bar is the 60d-half-life weighted mean, shrunk by n/(n+20)."""
    as_of = date(2026, 7, 1)
    ctx = _hand_ctx(
        [
            ("p", as_of - timedelta(days=120), "mid", {"combat": 1.0}),
            ("p", as_of - timedelta(days=60), "mid", {"combat": -1.0}),
        ]
    )
    priors = aggregate_prior(ctx, as_of)

    raw = (0.25 * 1.0 + 0.5 * -1.0) / 0.75
    assert priors["p"].pillar_z["combat"] == pytest.approx(raw * (2 / 22))
    assert priors["p"].n_games == 2


def test_aggregate_prior_excludes_the_as_of_day():
    """The aggregate is strictly before ``as_of``; a same-day game must not enter."""
    as_of = date(2026, 7, 1)
    ctx = _hand_ctx(
        [
            ("p", as_of - timedelta(days=60), "mid", {"combat": 1.0}),
            ("p", as_of, "mid", {"combat": -1.0}),
        ]
    )
    priors = aggregate_prior(ctx, as_of)

    assert priors["p"].n_games == 1
    assert priors["p"].pillar_z["combat"] == pytest.approx(1.0 * (1 / 21))


def test_aggregate_prior_skips_absent_pillars_and_filters_players():
    as_of = date(2026, 7, 1)
    ctx = _hand_ctx(
        [
            ("p", as_of - timedelta(days=1), "mid", {"combat": 1.0}),
            ("q", as_of - timedelta(days=1), "mid", {"vision": 0.5}),
        ]
    )
    priors = aggregate_prior(ctx, as_of, players={"p"})

    assert set(priors) == {"p"}
    assert "laning" not in priors["p"].pillar_z


def test_walk_forward_pillar_aggregates_are_pregame():
    """An appearance's aggregate reflects strictly earlier days only."""
    d1, d2 = date(2026, 5, 1), date(2026, 6, 1)
    ctx = _hand_ctx(
        [
            ("p", d1, "mid", {"combat": 1.0}),
            ("p", d2, "mid", {"combat": -1.0}),
        ]
    )
    agg = walk_forward_pillar_aggregates(ctx, half_life_days=60.0, k=20.0)

    combat = PILLARS.index("combat")
    assert np.isnan(agg[0, combat])
    assert agg[1, combat] == pytest.approx(1.0 * (1 / 21))
