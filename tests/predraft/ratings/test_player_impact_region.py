"""Region bookkeeping: one team-level gap, never per-player, and an exact league resolution."""

from __future__ import annotations

import pytest

from league_predictor.predraft.ratings.player_impact import game_eta
from tests.predraft.ratings._impact_helpers import (
    _LCK,
    _LEC,
    _MINOR,
    _facts_conn,
    _levels,
    _production_region_logodds,
    _seed_game,
    _zero_priors,
)


def test_same_region_rosters_contribute_no_region_term():
    """Two rosters of equal level cancel, so a domestic game prices no region at all."""
    same = _levels(a1=_LCK, a2=_LCK, a3=_LCK, b1=_LCK, b2=_LCK, b3=_LCK)
    assert same.gap(("a1", "a2", "a3"), ("b1", "b2", "b3")) == 0.0


def test_cross_region_game_contributes_exactly_one_team_level_gap():
    """A full cross-region game prices the region difference once, at team level."""
    side_a = tuple(f"a{i}" for i in range(5))
    side_b = tuple(f"b{i}" for i in range(5))
    regions = _levels(**dict.fromkeys(side_a, _LCK), **dict.fromkeys(side_b, _LEC))
    assert regions.gap(side_a, side_b) == pytest.approx(_LCK - _LEC)
    # Per-player, not modal: an imported LEC player moves the side by exactly his fifth.
    mixed = _levels(**dict.fromkeys(side_a[:4], _LCK), a4=_LEC, **dict.fromkeys(side_b, _LEC))
    assert mixed.gap(side_a, side_b) == pytest.approx((4 * _LCK + _LEC) / 5 - _LEC)


def test_a_short_lineup_prices_proportionally_less_region():
    """Lineups narrower than five exist in the facts; each present player carries his own fifth."""
    regions = _levels(a1=_LCK, a2=_LCK, a3=_LCK, **{f"b{i}": _LEC for i in range(5)})
    assert regions.gap(("a1", "a2", "a3"), tuple(f"b{i}" for i in range(5))) == pytest.approx(
        (3 * _LCK - 5 * _LEC) / 5
    )


def test_v9_prices_each_player_by_his_exact_league_through_the_region_table():
    """A feeder-league player takes his POOL's fitted level, not a tier anchor."""
    import sqlite3

    from league_predictor.db.schema import apply_current_schema
    from league_predictor.predraft.ratings.player_impact import fit as impact_fit
    from league_predictor.predraft.ratings.region_rating import RegionRatingTable

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    rows = [("g1", "LCK", "kp"), ("g2", "LCKC", "cp"), ("g3", "DCup", "up")]
    for gid, league, pid in rows:
        conn.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, status)"
            " VALUES (?, ?, 'ta', 'tb', '2026-05-01T10:00:00+00:00', 'completed')",
            (f"s_{gid}", league),
        )
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner) VALUES (?, ?, 1, 'ta')",
            (gid, f"s_{gid}"),
        )
        conn.execute(
            "INSERT INTO game_players (game_id, team_id, player_id) VALUES (?, 'ta', ?)",
            (gid, pid),
        )
    conn.commit()
    table = RegionRatingTable(
        r={"LCK": 0.2, "LCKC": -1.1},
        tier_effect={1: 0.0, 2: -0.9, 3: -1.7},
        home={},
    )
    from datetime import date as _date

    levels, region_of, _pool_hit = impact_fit._resolve_player_levels_by_league(
        conn, _date(2026, 6, 1), table
    )
    assert levels.of("kp") == table.r["LCK"]
    assert levels.of("cp") == table.r["LCKC"]  # exact pool level, not the tier-2 anchor
    assert levels.of("up") == table.tier_effect[3]  # DCup: no pool row -> learned tier level
    assert region_of["kp"] == "LCK"
    assert region_of["cp"] == ""


def test_an_unresolved_player_prices_the_weakest_tier_not_a_neutral_major():
    """A player the resolution never saw takes the tier anchor, never an average major region."""
    regions = _levels(**{f"b{i}": _LCK for i in range(5)})
    side_a = tuple(f"a{i}" for i in range(5))
    # Every 'a' player is unknown: the side reads as the default tier, so the gap is real.
    assert regions.gap(side_a, tuple(f"b{i}" for i in range(5))) == pytest.approx(_MINOR - _LCK)
    assert regions.of("a0") == _MINOR


def test_a_rosters_five_shares_re_sum_to_its_side_of_the_gap():
    """The published per-player share is exactly a fifth of the roster's side of the gap."""
    roster_a = tuple(f"a{i}" for i in range(5))
    roster_b = tuple(f"b{i}" for i in range(5))
    regions = _levels(**dict.fromkeys(roster_a, _LCK), **dict.fromkeys(roster_b[:4], _LEC))
    shares = sum(regions.share(p) for p in roster_a) - sum(regions.share(p) for p in roster_b)
    assert shares == pytest.approx(regions.gap(roster_a, roster_b))


def test_a_player_the_fit_never_saw_is_priced_at_his_roles_replacement_value():
    """Replacement values price only post-cutoff debutants — everyone in the corpus has a value."""
    from league_predictor.predraft.ratings.player_impact import ImpactValues

    values = ImpactValues(
        player_value={"a1": 0.4},
        replacement_value={"top": -0.2, "mid": -0.5},
        player_role={"debutant": "top"},
        region=_levels(a1=_LCK, debutant=_LCK),
        alpha=0.0,
        beta=0.0,
    )
    assert game_eta(values, ("a1", "debutant"), ("a1",), 0) == pytest.approx(-0.2)
    # A debutant with no known role at all takes the mid replacement value.
    assert game_eta(values, ("a1", "unrostered"), ("a1",), 0) == pytest.approx(-0.5)


def test_region_never_enters_eta_twice():
    """The eta's only region term is alpha * gap — player values carry none of it."""
    from league_predictor.predraft.ratings.player_impact import ImpactValues

    values = ImpactValues(
        player_value={"a1": 0.2, "b1": -0.1},
        replacement_value={},
        player_role={},
        region=_levels(a1=_LCK, b1=_LEC),
        alpha=2.0,
        beta=0.0,
    )
    eta = game_eta(values, ("a1",), ("b1",), 0)
    assert eta == pytest.approx(0.2 - (-0.1) + 2.0 * (_LCK - _LEC) / 5)


def test_region_levels_resolve_exactly_never_by_substring():
    """A feeder league that merely CONTAINS a major key prices its own pool, not that region."""
    from datetime import date

    from league_predictor.predraft.ratings.player_impact.fit import (
        _resolve_player_levels_by_league,
    )

    table = _production_region_logodds()
    conn = _facts_conn()
    _seed_game(conn, gid="g1", day="2026-03-01", team1="AA", team2="BB", winner="AA", league="LCKC")
    _seed_game(conn, gid="g2", day="2026-03-02", team1="CC", team2="DD", winner="CC", league="NACL")
    _seed_game(
        conn, gid="g3", day="2026-03-03", team1="EE", team2="FF", winner="EE", league="LTA N"
    )

    levels, region_of, _pool_hit = _resolve_player_levels_by_league(conn, date(2026, 5, 1), table)
    conn.close()

    tier2_level = table.tier_effect[2]  # LCKC/NACL: no pool row in this table -> tier level
    assert region_of["AA_mid"] == ""
    assert levels.of("AA_mid") == pytest.approx(tier2_level)
    assert region_of["CC_top"] == ""
    assert levels.of("CC_top") == pytest.approx(tier2_level)
    assert levels.of("CC_top") != table.r["LCS"]
    assert levels.of("AA_mid") != table.r["LCK"]
    # LTA N folds to the LCS pool and normalizes to the NA region facet.
    assert region_of["EE_bot"] == "NA"
    assert levels.of("EE_bot") == pytest.approx(table.r["LCS"])
    assert region_of["FF_sup"] == "NA"  # every rostered player, not just the qualified ones


def test_region_levels_ignore_an_international_last_game():
    """A player's level is their home league's, not the tournament they last travelled to."""
    from datetime import date

    from league_predictor.predraft.ratings.player_impact.fit import (
        _resolve_player_levels_by_league,
    )

    table = _production_region_logodds()
    conn = _facts_conn()
    _seed_game(conn, gid="g1", day="2026-03-01", team1="AA", team2="BB", winner="AA", league="LCK")
    _seed_game(conn, gid="g2", day="2026-04-01", team1="AA", team2="BB", winner="AA", league="WLDs")

    levels, region_of, _pool_hit = _resolve_player_levels_by_league(conn, date(2026, 5, 1), table)
    conn.close()
    assert region_of["AA_mid"] == "LCK"
    assert levels.of("AA_mid") == pytest.approx(table.r["LCK"])


def test_a_collapsed_region_resolution_is_an_error_not_a_quiet_fit(monkeypatch):
    """Below the coverage floor the fit refuses, naming the cause and the operator action."""
    from datetime import date, timedelta

    from league_predictor.predraft.ratings.player_impact import fit as impact_fit
    from league_predictor.predraft.ratings.player_impact import prepare_impact_fit

    conn = _facts_conn()
    # A league spelling outside league_tier.LEAGUES: no home, no pool row, only fallbacks.
    for index in range(26):
        day = (date(2026, 1, 5) + timedelta(days=index * 4)).isoformat()
        _seed_game(
            conn, gid=f"u{index}", day=day, team1="AA", team2="BB", winner="AA", league="XXQ"
        )
    monkeypatch.setattr(impact_fit, "_stat_context_priors", _zero_priors)

    with pytest.raises(RuntimeError, match=r"league_tier.LEAGUES"):
        prepare_impact_fit(conn, date(2026, 5, 15))
    conn.close()
