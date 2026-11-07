"""Unit tests for player scoring: pure functions plus compute_position_baselines on facts."""

import sqlite3
from datetime import date

import pytest

from league_predictor.db.schema import apply_current_schema
from league_predictor.predraft.ratings.player_scoring import (
    PILLAR_STATS_BY_ROLE,
    apply_shrinkage,
    compute_pillar_score,
    compute_position_baselines,
    performance_score_to_grade,
    pillar_stats_for,
    z_to_performance_score,
)


class TestApplyShrinkage:
    def test_zero_games(self):
        assert apply_shrinkage(2.0, 0) == 0.0

    def test_k_equals_games(self):
        # n=20, k=20 → factor = 0.5
        assert abs(apply_shrinkage(2.0, 20) - 1.0) < 1e-9

    def test_high_games(self):
        # n=100, k=20 → factor = 100/120 ≈ 0.833
        result = apply_shrinkage(3.0, 100)
        assert abs(result - 3.0 * 100 / 120) < 1e-9

    def test_three_games(self):
        # n=3, k=20 → factor = 3/23 ≈ 0.130
        result = apply_shrinkage(2.0, 3)
        assert abs(result - 2.0 * 3 / 23) < 1e-9

    def test_ten_games(self):
        # n=10, k=20 → factor = 10/30 ≈ 0.333
        result = apply_shrinkage(1.5, 10)
        assert abs(result - 1.5 * 10 / 30) < 1e-9


class TestZToPerformanceScore:
    def test_zero_z(self):
        assert z_to_performance_score(0.0) == 50.0

    def test_positive_z(self):
        assert z_to_performance_score(1.0) == 60.0

    def test_negative_z(self):
        assert z_to_performance_score(-1.0) == 40.0

    def test_clips_high(self):
        assert z_to_performance_score(10.0) == 99.0

    def test_clips_low(self):
        assert z_to_performance_score(-10.0) == 1.0

    def test_elite_threshold(self):
        # S grade = 80+ = z >= 3.0
        assert z_to_performance_score(3.0) == 80.0


class TestPerformanceScoreToGrade:
    def test_s_grade(self):
        assert performance_score_to_grade(80.0) == "S"
        assert performance_score_to_grade(99.0) == "S"

    def test_a_grade(self):
        assert performance_score_to_grade(65.0) == "A"
        assert performance_score_to_grade(79.9) == "A"

    def test_b_grade(self):
        assert performance_score_to_grade(50.0) == "B"
        assert performance_score_to_grade(64.9) == "B"

    def test_c_grade(self):
        assert performance_score_to_grade(35.0) == "C"
        assert performance_score_to_grade(49.9) == "C"

    def test_d_grade(self):
        assert performance_score_to_grade(34.9) == "D"
        assert performance_score_to_grade(1.0) == "D"


class TestPillarStatsByRole:
    def test_carry_roles_have_three_pillars(self):
        for role in ("top", "jng", "mid", "bot"):
            assert set(PILLAR_STATS_BY_ROLE[role]) == {"laning", "combat", "vision"}, role

    def test_jng_combat_drops_killparticipation(self):
        assert "killparticipation" not in PILLAR_STATS_BY_ROLE["jng"]["combat"]
        assert "killparticipation" in PILLAR_STATS_BY_ROLE["mid"]["combat"]


class TestPillarStatsFor:
    def test_known_role(self):
        assert pillar_stats_for("sup") == PILLAR_STATS_BY_ROLE["sup"]
        assert pillar_stats_for("mid") == PILLAR_STATS_BY_ROLE["mid"]

    def test_unknown_role_falls_back_to_mid(self):
        assert pillar_stats_for("unknown_role") == PILLAR_STATS_BY_ROLE["mid"]

    def test_none_falls_back_to_mid(self):
        assert pillar_stats_for(None) == PILLAR_STATS_BY_ROLE["mid"]


class TestComputePillarScore:
    def test_full_quality_when_all_present(self):
        zscores = {"a": 1.0, "b": 2.0, "c": 0.0}
        present = {"a": True, "b": True, "c": True}
        score, dq = compute_pillar_score(zscores, ["a", "b", "c"], present)
        assert abs(score - 1.0) < 1e-9  # (1+2+0)/3
        assert dq == "full"

    def test_partial_when_some_missing(self):
        """Partial pillars are attenuated by sqrt(n_present/n_expected)."""
        zscores = {"a": 1.0, "b": 3.0}
        present = {"a": True, "b": True, "c": False}
        score, dq = compute_pillar_score(zscores, ["a", "b", "c"], present)
        # Raw mean = 2.0; attenuated by √(2/3) ≈ 0.8165.
        assert abs(score - 2.0 * (2 / 3) ** 0.5) < 1e-9
        assert dq == "partial"

    def test_dropped_when_no_stats_present(self):
        zscores: dict[str, float] = {}
        present = {"a": False, "b": False}
        score, dq = compute_pillar_score(zscores, ["a", "b"], present)
        assert score is None
        assert dq == "dropped"

    def test_present_default_false_when_key_missing_from_mask(self):
        """A stat absent from the mask counts as missing: one-of-two attenuates by sqrt(1/2)."""
        zscores = {"a": 1.0}
        present = {"a": True}  # "b" not in mask at all
        score, dq = compute_pillar_score(zscores, ["a", "b"], present)
        assert abs(score - (1 / 2) ** 0.5) < 1e-9
        assert dq == "partial"

    def test_single_stat_pillar_not_attenuated(self):
        """A pillar expecting one stat, present, is full quality with no attenuation."""
        zscores = {"x": 1.5}
        present = {"x": True}
        score, dq = compute_pillar_score(zscores, ["x"], present)
        assert abs(score - 1.5) < 1e-9
        assert dq == "full"


class TestComputePositionBaselines:
    """compute_position_baselines over the facts tables: role means, KDA, tier filter."""

    @staticmethod
    def _facts_db() -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        apply_current_schema(conn)
        return conn

    @staticmethod
    def _game(conn, *, gid, league, day):
        """One one-game series in ``league`` on ``day`` (canonical A<B)."""
        conn.execute(
            "INSERT OR IGNORE INTO leagues (league_id, name) VALUES (?, ?)", (league, league)
        )
        conn.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, "
            "best_of, winner, status) VALUES (?, ?, 'A', 'B', ?, 1, 'A', 'completed')",
            (f"ser_{gid}", league, f"{day}T10:00:00+00:00"),
        )
        conn.execute(
            "INSERT INTO games (game_id, series_id, game_no, winner) VALUES (?, ?, 1, 'A')",
            (gid, f"ser_{gid}"),
        )

    @staticmethod
    def _pick(conn, *, gid, pid, role, kills, deaths, assists, golddiffat15):
        conn.execute(
            "INSERT INTO game_players (game_id, team_id, player_id, role, kills, "
            "deaths, assists, golddiffat15) VALUES (?, 'A', ?, ?, ?, ?, ?, ?)",
            (gid, pid, role, kills, deaths, assists, golddiffat15),
        )

    def test_reads_facts_tables_for_role_means(self):
        conn = self._facts_db()
        # Two mids over two games in the window: golddiffat15 100 and 300 -> mean 200.
        self._game(conn, gid="g1", league="LCK", day="2026-03-01")
        self._game(conn, gid="g2", league="LCK", day="2026-03-05")
        self._pick(
            conn, gid="g1", pid="mid1", role="mid", kills=4, deaths=2, assists=6, golddiffat15=100.0
        )
        self._pick(
            conn, gid="g2", pid="mid1", role="mid", kills=6, deaths=2, assists=4, golddiffat15=300.0
        )
        conn.commit()

        baselines = compute_position_baselines(conn, date(2026, 3, 20), min_games=1)
        conn.close()
        assert "mid" in baselines
        means, _ = baselines["mid"]
        assert means["golddiffat15"] == pytest.approx(200.0)

    def test_kda_is_a_per_player_aggregate(self):
        """KDA baseline is the mean over players of sum(k+a)/sum(deaths+eps)."""
        conn = self._facts_db()
        self._game(conn, gid="g1", league="LCK", day="2026-03-01")
        self._game(conn, gid="g2", league="LCK", day="2026-03-05")
        # One mid, a zero-death game + a normal game: agg = (9+11)/(0+2+eps) ~= 10.
        self._pick(
            conn, gid="g1", pid="mid1", role="mid", kills=5, deaths=0, assists=4, golddiffat15=100.0
        )
        self._pick(
            conn, gid="g2", pid="mid1", role="mid", kills=4, deaths=2, assists=7, golddiffat15=100.0
        )
        conn.commit()

        means, _ = compute_position_baselines(conn, date(2026, 3, 20), min_games=1)["mid"]
        conn.close()
        assert means["kda"] < 50, f"KDA baseline exploded: {means['kda']}"
        assert means["kda"] == pytest.approx(20 / 2.001, abs=0.05)

    def test_tier_filter_reads_series_league_id(self):
        """The tier filter reads series.league_id: a tier-1 request excludes tier-2 rows."""
        conn = self._facts_db()
        self._game(conn, gid="g_lck", league="LCK", day="2026-03-01")  # tier 1
        self._game(conn, gid="g_lckc", league="LCKC", day="2026-03-02")  # tier 2
        self._pick(
            conn,
            gid="g_lck",
            pid="mid_major",
            role="mid",
            kills=4,
            deaths=2,
            assists=6,
            golddiffat15=100.0,
        )
        self._pick(
            conn,
            gid="g_lckc",
            pid="mid_minor",
            role="mid",
            kills=4,
            deaths=2,
            assists=6,
            golddiffat15=900.0,
        )
        conn.commit()

        means, _ = compute_position_baselines(conn, date(2026, 3, 20), min_games=1, tier=1)["mid"]
        conn.close()
        # Only the LCK game entered the tier-1 baseline (100, not 900/500).
        assert means["golddiffat15"] == pytest.approx(100.0)

    def test_window_excludes_games_outside_trailing_days(self):
        conn = self._facts_db()
        self._game(conn, gid="g_in", league="LCK", day="2026-03-01")
        self._game(conn, gid="g_old", league="LCK", day="2025-01-01")  # >180d before
        self._pick(
            conn,
            gid="g_in",
            pid="mid1",
            role="mid",
            kills=4,
            deaths=2,
            assists=6,
            golddiffat15=100.0,
        )
        self._pick(
            conn,
            gid="g_old",
            pid="mid1",
            role="mid",
            kills=4,
            deaths=2,
            assists=6,
            golddiffat15=999.0,
        )
        conn.commit()

        means, _ = compute_position_baselines(
            conn, date(2026, 3, 20), trailing_days=180, min_games=1
        )["mid"]
        conn.close()
        assert means["golddiffat15"] == pytest.approx(100.0)
