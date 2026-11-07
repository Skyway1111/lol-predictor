"""Pins for the learned region-rating fit: leak-safety, migration, identity, pricing."""

import sqlite3
from datetime import date

import pytest

from league_predictor.db.schema import apply_current_schema
from league_predictor.predraft.ratings.region_rating import (
    RegionRatingTable,
    fit_region_ratings,
)
from league_predictor.probability import ELO_ORIGIN, ELO_PER_LOGODDS

THROUGH = "2026-07-01"


def _five(team, star=None):
    players = [f"{team}_p{i}" for i in range(5)]
    if star is not None:
        players[0] = star
    return players


def _seed(matches):
    """In-memory facts DB from (mid, day, league, t1, t2, winner, players1, players2)."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    for i, (mid, day, league, t1, t2, winner, players1, players2) in enumerate(matches):
        ta, tb = sorted((t1, t2))
        conn.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, status) "
            "VALUES (?, ?, ?, ?, ?, 'completed')",
            (f"ser_{mid}", league, ta, tb, f"{day}T10:00:00+00:00"),
        )
        blue = t1 if i % 2 == 0 else t2
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner) "
            "VALUES (?, ?, 1, ?, ?, ?)",
            (mid, f"ser_{mid}", blue, t2 if blue == t1 else t1, winner),
        )
        for team, players in ((t1, players1), (t2, players2)):
            for pid in players:
                conn.execute(
                    "INSERT INTO game_players (game_id, team_id, player_id) VALUES (?, ?, ?)",
                    (mid, team, pid),
                )
    conn.commit()
    return conn


def _league_block(league, teams, start_day, n, winner_fn, mid_prefix):
    """n round-robin-ish games inside one league; winner chosen by winner_fn(t1, t2)."""
    out = []
    for i in range(n):
        t1, t2 = teams[i % len(teams)], teams[(i + 1) % len(teams)]
        day = date.fromisoformat(start_day).toordinal() + i // 4
        out.append(
            (
                f"{mid_prefix}{i}",
                date.fromordinal(day).isoformat(),
                league,
                t1,
                t2,
                winner_fn(t1, t2),
                _five(t1),
                _five(t2),
            )
        )
    return out


def _two_region_matches(msi_lck_wins=8):
    matches = _league_block("LCK", ["K1", "K2", "K3"], "2026-01-10", 36, lambda a, _b: a, "k")
    matches += _league_block("LEC", ["E1", "E2", "E3"], "2026-01-10", 36, lambda a, _b: a, "e")
    for i in range(10):
        matches.append(
            (
                f"msi{i}",
                "2026-05-10",
                "MSI",
                "K1",
                "E1",
                "K1" if i < msi_lck_wins else "E1",
                _five("K1"),
                _five("E1"),
            )
        )
    return matches


def test_cross_pool_wins_raise_the_winning_pool():
    table = fit_region_ratings(_seed(_two_region_matches()), through=THROUGH)
    assert table.rating("LCK") > table.rating("LEC")
    assert table.elo("LCK") == ELO_ORIGIN + table.rating("LCK") * ELO_PER_LOGODDS


def test_the_fit_ignores_games_on_or_after_the_cutoff():
    base = _two_region_matches()
    future = [
        (f"f{i}", "2026-07-01", "MSI", "K1", "E1", "E1", _five("K1"), _five("E1"))
        for i in range(10)
    ]
    table = fit_region_ratings(_seed(base), through=THROUGH)
    with_future = fit_region_ratings(_seed(base + future), through=THROUGH)
    assert table == with_future


def test_future_schedule_facts_cannot_move_the_table_through_the_home_map():
    """A post-cutoff league move must not re-home a team's pre-cutoff games (audit probe pin)."""
    base = _two_region_matches()
    # K1 "moves" to LPL after the cutoff: 40 post-cutoff LPL games — enough to flip the
    # modal home for 2026 and to clear _MIN_GAMES — plus enough to seat LPL as a pool.
    future = [
        (
            f"lpl{i}",
            "2026-07-05",
            "LPL",
            "K1",
            f"L{i % 3}",
            "K1",
            _five("K1"),
            _five(f"L{i % 3}"),
        )
        for i in range(40)
    ]
    table = fit_region_ratings(_seed(base), through=THROUGH)
    with_future = fit_region_ratings(_seed(base + future), through=THROUGH)
    assert with_future.league_of("K1", 2026) == "LCK"  # pre-cutoff games stay LCK-homed
    assert "LPL" not in with_future.r  # a pool with zero pre-cutoff games never exports
    assert with_future.home == table.home
    assert set(with_future.r) == set(table.r)
    # The future players widen the column space, permuting float-summation order; the
    # information content must be identical to the last ulp-scale reassociation.
    for pool, value in table.r.items():
        assert with_future.r[pool] == pytest.approx(value, abs=1e-9)
    for tier, value in table.tier_effect.items():
        assert with_future.tier_effect[tier] == pytest.approx(value, abs=1e-9)


def test_the_cutoff_accepts_date_and_iso_string_equally():
    conn = _seed(_two_region_matches())
    assert fit_region_ratings(conn, through=THROUGH) == fit_region_ratings(
        conn, through=date(2026, 7, 1)
    )


def test_a_tier_with_no_fitted_pool_prices_at_the_weakest_fitted_level():
    """The fitted fallback anchors below the field, never at the population average."""
    table = fit_region_ratings(_seed(_two_region_matches()), through=THROUGH)
    assert set(table.r) == {"LCK", "LEC"}  # tier-1 pools only in this fixture
    weakest = min(table.r.values())
    assert table.tier_effect[2] == weakest
    assert table.tier_effect[3] == weakest
    assert table.tier_effect[2] < 0.0  # strictly below the tier-1 origin


def test_migration_moves_an_isolated_pool_off_its_tier_level():
    # LCKC never plays a cross-pool game; the star's LCK evidence is the only bridge.
    def fixture(*, with_migration):
        star = "star" if with_migration else "c1_p0"
        matches = _league_block("LCKC", ["C1", "C2", "C3"], "2026-01-10", 36, lambda *_: "C1", "c")
        matches = [
            (mid, day, lg, t1, t2, w, _five(t1, star if t1 == "C1" else None), _five(t2))
            for (mid, day, lg, t1, t2, w, _p1, _p2) in matches
        ]
        matches += _league_block("LCK", ["K1", "K2", "K3"], "2026-03-01", 36, lambda a, _b: a, "k")
        for i in range(12):
            matches.append(
                (
                    f"m{i}",
                    "2026-05-01",
                    "LCK",
                    "K4",
                    ["K1", "K2", "K3"][i % 3],
                    "K4",
                    _five("K4", star if with_migration else None),
                    _five(["K1", "K2", "K3"][i % 3]),
                )
            )
        return fit_region_ratings(_seed(matches), through=THROUGH)

    migrated = fixture(with_migration=True)
    isolated = fixture(with_migration=False)
    assert migrated.rating("LCKC") > isolated.rating("LCKC")


def _table(**overrides):
    base = {
        "r": {"LCK": 0.1, "LEC": -0.1, "LFL": -0.8, "PRM": -0.9},
        "tier_effect": {1: 0.0, 2: -0.85, 3: -1.5},
        "home": {"K1|2026": "LCK", "E1|2026": "LEC", "F1|2026": "LFL", "P1|2026": "PRM"},
        "beta_cross_tier": 1.0,
        "beta_within_tier1": 1.0,
    }
    base.update(overrides)
    return RegionRatingTable(**base)


def test_same_pool_and_unmapped_matchups_are_byte_identical_through_adjust():
    table = _table(home={"K1|2026": "LCK", "K2|2026": "LCK", "E1|2026": "LEC"})
    served = 0.6234567
    as_of = date(2026, 6, 1)
    assert table.adjust(served, "K1", "K2", as_of) is served
    assert table.adjust(served, "K1", "X9", as_of) is served
    assert table.adjust(served, "K1", "E1", as_of) != served


def test_same_tier_below_tier1_stays_identity_and_cross_tier_shifts():
    table = _table()
    served = 0.5
    as_of = date(2026, 6, 1)
    assert table.adjust(served, "F1", "P1", as_of) is served  # tier2 vs tier2
    assert table.adjust(served, "K1", "F1", as_of) > served  # tier1 over tier2
    zeroed = table.with_betas(cross_tier=0.0, within_tier1=1.0)
    assert zeroed.adjust(served, "K1", "F1", as_of) is served


def test_a_miss_prices_at_its_tier_never_average():
    table = _table()
    assert table.rating("SomeUnknownCup") == table.tier_effect[3]
    assert table.rating(None) == table.tier_effect[3]
    assert table.rating("VCS") == table.tier_effect[2]  # known tier-2 league, no fit row


def test_rating_folds_pool_aliases():
    table = _table(r={"LCS": 0.05, "LCK": 0.1})
    assert table.rating("LTA N") == table.r["LCS"]


def test_meta_round_trip_is_exact_and_json_safe():
    import json

    table = _table()
    assert RegionRatingTable.from_meta(table.to_meta()) == table
    fitted = fit_region_ratings(_seed(_two_region_matches()), through=THROUGH)
    meta = json.loads(json.dumps(fitted.to_meta()))  # np types would fail to serialize
    assert RegionRatingTable.from_meta(meta) == fitted


def test_repeat_calls_hit_the_vintage_cache(tmp_path):
    db_path = tmp_path / "facts.db"
    mem = _seed(_two_region_matches())
    disk = sqlite3.connect(db_path)
    mem.backup(disk)
    disk.row_factory = sqlite3.Row
    first = fit_region_ratings(disk, through=THROUGH)
    assert fit_region_ratings(disk, through=THROUGH) is first
