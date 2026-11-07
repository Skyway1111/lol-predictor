"""Verify leak-free Impact-based Draft features and their artifact provenance."""

from __future__ import annotations

import math
import sqlite3
from typing import TYPE_CHECKING, cast

import pytest

from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.grader.pro_excess import (
    HALF_LIFE_DAYS,
    SHRINKAGE_K,
    _build_impact_baseline,
    require_impact_history,
)
from league_predictor.probability import sigmoid as _sigmoid
from tests.draft_grading.grader._stats_helpers import insert_game, sql_pro_stats
from tests.draft_grading.grader.conftest import make_grader_test_db

if TYPE_CHECKING:
    from league_predictor.vocab.roles import Role


def _insert_impact(db: sqlite3.Connection, player_id: str, asof: str, logodds: float) -> None:
    db.execute(
        "INSERT INTO impact_history (player_id, asof_date, impact_logodds) VALUES (?, ?, ?)",
        (player_id, asof, logodds),
    )
    db.commit()


class TestBuildImpactBaseline:
    """The pure builder: strict-before reads, unrated zeros, exact 0.5 neutrality."""

    def test_game_on_a_grid_date_does_not_read_that_dates_fit(self):
        """A impact row dated exactly the game date is invisible (strict <)."""
        impact_rows = [("p1", "2025-06-10", 5.0)]
        lineups = [
            ("g1", "t1", "p1", "2025-06-10"),  # ON the grid date
            ("g1", "t2", "q1", "2025-06-10"),
            ("g2", "t1", "p1", "2025-06-11"),  # one day later
            ("g2", "t2", "q1", "2025-06-11"),
        ]
        baseline = _build_impact_baseline(impact_rows, lineups)
        assert baseline[("t1", "g1")] == 0.5  # exactly: no rating strictly before
        assert baseline[("t1", "g2")] == pytest.approx(_sigmoid(5.0))

    def test_latest_strictly_before_wins(self):
        """Between two grid fits, a game reads the earlier one; after both, the later."""
        impact_rows = [("p1", "2025-01-01", 1.0), ("p1", "2025-02-01", -1.0)]
        lineups = [
            ("mid", "t1", "p1", "2025-01-20"),
            ("mid", "t2", "q1", "2025-01-20"),
            ("late", "t1", "p1", "2025-03-01"),
            ("late", "t2", "q1", "2025-03-01"),
        ]
        baseline = _build_impact_baseline(impact_rows, lineups)
        assert baseline[("t1", "mid")] == pytest.approx(_sigmoid(1.0))
        assert baseline[("t1", "late")] == pytest.approx(_sigmoid(-1.0))

    def test_unrated_player_contributes_zero(self):
        """Only rated players move the sum; the enemy side's rating subtracts."""
        impact_rows = [("p1", "2025-01-01", 2.0), ("q1", "2025-01-01", 0.5)]
        lineups = [
            ("g1", "t1", "p1", "2025-02-01"),
            ("g1", "t1", "p_unrated", "2025-02-01"),
            ("g1", "t2", "q1", "2025-02-01"),
            ("g1", "t2", "q_unrated", "2025-02-01"),
        ]
        baseline = _build_impact_baseline(impact_rows, lineups)
        assert baseline[("t1", "g1")] == pytest.approx(_sigmoid(2.0 - 0.5))
        assert baseline[("t2", "g1")] == pytest.approx(_sigmoid(0.5 - 2.0))

    def test_all_unrated_game_is_exactly_half(self):
        """No rated player on either side → the baseline is EXACTLY 0.5."""
        impact_rows = [("someone_else", "2025-01-01", 3.0)]
        lineups = [
            ("g1", "t1", "a1", "2025-02-01"),
            ("g1", "t2", "b1", "2025-02-01"),
        ]
        baseline = _build_impact_baseline(impact_rows, lineups)
        assert baseline[("t1", "g1")] == 0.5
        assert baseline[("t2", "g1")] == 0.5


class TestRequireImpactHistory:
    def test_missing_table_raises_naming_the_fix(self):
        db = sqlite3.connect(":memory:")  # no schema at all
        with pytest.raises(RuntimeError, match="lol-predict backfill impact"):
            require_impact_history(db)

    def test_empty_table_raises_naming_the_fix(self):
        db = make_grader_test_db()
        with pytest.raises(RuntimeError, match="lol-predict backfill impact"):
            require_impact_history(db)

    def test_populated_table_passes(self):
        db = make_grader_test_db()
        _insert_impact(db, "p1", "2025-01-01", 0.3)
        require_impact_history(db)  # no raise


class TestComputeProExcessImpactBt:
    def test_hand_checked_excess_against_the_bt_baseline(self):
        """One win by a rated-strong lineup: excess = w*(1 - sigmoid(2)) / (w + K)."""
        db = make_grader_test_db()
        _insert_impact(db, "p_star", "2025-01-01", 2.0)
        insert_game(
            db, "g1", "2025-02-01", "Jinx", "bot", "p_star", "t1", 1, enemy_player="q_nobody"
        )

        excess, _pr, _w = BulkStatsCache(db, baseline="impact_bt").pro_stats(
            "Jinx", "bot", "2025-03-01"
        )

        w = math.exp(-math.log(2) / HALF_LIFE_DAYS * 28)
        expected = w * (1.0 - _sigmoid(2.0)) / (w + SHRINKAGE_K)
        assert excess == pytest.approx(expected, rel=1e-12)

    def test_star_lineup_win_is_worth_less_than_unknown_lineup_win(self):
        """On identical 1-0 records, the star-fielded champion's excess is the smaller one."""
        db = make_grader_test_db()
        _insert_impact(db, "p_star", "2025-01-01", 2.0)
        insert_game(
            db, "g1", "2025-02-01", "Carried", "mid", "p_star", "strong", 1, enemy_player="e1"
        )
        insert_game(
            db, "g2", "2025-02-01", "SelfMade", "mid", "p_new", "fresh", 1, enemy_player="e2"
        )

        carried, _, _ = BulkStatsCache(db, baseline="impact_bt").pro_stats(
            "Carried", "mid", "2025-03-01"
        )
        self_made, _, _ = BulkStatsCache(db, baseline="impact_bt").pro_stats(
            "SelfMade", "mid", "2025-03-01"
        )
        assert self_made > carried > 0.0

    def test_unknown_baseline_raises(self):
        db = make_grader_test_db()
        with pytest.raises(ValueError, match="unknown pro_excess_baseline"):
            BulkStatsCache(db, baseline="elo").pro_stats("Jinx", "bot", "2025-03-01")


class TestSqlCacheAgreementImpactBt:
    """The SQL path and its BulkStatsCache twin agree under impact_bt, as under team_form."""

    def _seeded_db(self) -> sqlite3.Connection:
        db = make_grader_test_db()
        games = [
            ("g01", "2025-11-05", "Jinx", "bot", "p1", 1),
            ("g02", "2025-11-20", "Jinx", "bot", "p1", 0),
            ("g03", "2025-12-10", "Jinx", "bot", "p2", 1),
            ("g04", "2026-01-04", "Jinx", "bot", "p1", 1),
            ("g05", "2026-02-15", "Jinx", "bot", "p1", 0),
            ("g06", "2025-12-01", "Ahri", "mid", "p2", 1),
            ("g07", "2026-01-20", "Ahri", "mid", "p2", 0),
            ("g08", "2026-02-01", "Ahri", "mid", "p2", 1),
            ("g09", "2026-01-10", "Ksante", "top", "p1", 0),
            ("g10", "2026-02-20", "Ksante", "top", "p1", 1),
            # Exactly on the as-of date — both paths must exclude it.
            ("g11", "2026-03-01", "Jinx", "bot", "p1", 1),
        ]
        for match_id, date, champ, role, player, result in games:
            insert_game(
                db, match_id, date, champ, role, player, "t1", result, enemy_player=f"e_{match_id}"
            )
        # A biweekly-ish grid: p1 rated twice (rating drifts), p2 once, e_g03 once.
        for pid, asof, lo in [
            ("p1", "2025-11-01", 0.4),
            ("p1", "2026-01-01", -0.6),
            ("p2", "2025-12-05", 1.1),
            ("e_g03", "2025-12-05", 0.9),
        ]:
            _insert_impact(db, pid, asof, lo)
        return db

    def test_pro_excess_agree(self):
        db = self._seeded_db()
        cache = BulkStatsCache(db, baseline="impact_bt")
        assert cache.baseline == "impact_bt"
        combos = cast(
            "list[tuple[str, Role]]",
            [("Jinx", "bot"), ("Ahri", "mid"), ("Ksante", "top"), ("Yone", "mid")],
        )
        for as_of in ("2026-03-01", "2026-01-15"):
            for champ, role in combos:
                sql = sql_pro_stats(db, champ, role, as_of, baseline="impact_bt")
                bulk = cache.pro_stats(champ, role, as_of)
                assert bulk == pytest.approx(sql, rel=1e-9, abs=1e-12), (
                    f"{champ}/{role} @ {as_of}: SQL {sql} != Bulk {bulk}"
                )


class TestBaselineGuards:
    def test_cache_unknown_baseline_raises(self):
        db = make_grader_test_db()
        with pytest.raises(ValueError, match="unknown pro_excess_baseline"):
            BulkStatsCache(db, baseline="elo")

    def test_impact_bt_cache_on_empty_impact_history_raises(self):
        db = make_grader_test_db(game_count=6)
        with pytest.raises(RuntimeError, match="lol-predict backfill impact"):
            BulkStatsCache(db, baseline="impact_bt")
