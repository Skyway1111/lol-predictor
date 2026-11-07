"""Tests for the decayed pro-excess stat: SQL path, team form, and the bulk twin."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pytest

from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
from league_predictor.draft_grading.grader.pro_excess import (
    _build_team_form,
    team_form_offsets,
)
from tests.draft_grading.grader._stats_helpers import insert_game, sql_pro_stats
from tests.draft_grading.grader.conftest import make_grader_test_db

if TYPE_CHECKING:
    import sqlite3

    from league_predictor.vocab.roles import Role


class TestProExcess:
    def test_decay_weights_recent_more(self):
        """Recent wins outweigh old losses — excess is positive."""
        db = make_grader_test_db()
        # 3 wins 10 days ago
        for i in range(3):
            insert_game(db, f"recent_{i}", "2026-01-20", "Jinx", "bot", "p1", "t1", 1)
        # 3 losses 90 days ago
        for i in range(3):
            insert_game(db, f"old_{i}", "2025-11-01", "Jinx", "bot", "p1", "t1", 0)

        excess, _pr, n_eff = BulkStatsCache(db, half_life_days=30.0).pro_stats(
            "Jinx", "bot", "2026-01-30"
        )
        assert excess > 0.0
        assert n_eff > 0.0

    def test_walk_forward_excludes_future(self):
        """Games on or after as_of_date are excluded."""
        db = make_grader_test_db()
        insert_game(db, "past", "2026-01-01", "Ahri", "mid", "p1", "t1", 1)
        insert_game(db, "future", "2026-02-01", "Ahri", "mid", "p1", "t1", 0)

        excess, _, _ = BulkStatsCache(db).pro_stats("Ahri", "mid", "2026-01-15")
        # Only the win is visible; team form before it is the 0.5 prior, so
        # excess = (1 - 0.5) shrunk toward 0 -> strictly positive.
        assert excess > 0.0

    def test_no_data_returns_neutral(self):
        """Champion with no games returns neutral (0.0 excess, 0.0 PR, 0.0 weight)."""
        db = make_grader_test_db()
        excess, pr, n_eff = BulkStatsCache(db).pro_stats("Yone", "mid", "2026-01-01")
        assert excess == 0.0
        assert pr == 0.0
        assert n_eff == 0.0

    def test_form_debias_neutralizes_carried_champion(self):
        """A champion that only wins on an already-strong team has the muted excess."""
        db = make_grader_test_db()
        # 'strong_team' banks a long win streak on a filler champion first.
        for i in range(8):
            insert_game(
                db, f"streak_{i}", f"2025-10-{i + 1:02d}", "Filler", "top", "ps", "strong", 1
            )
        # Carried champion: 5 wins on the already-strong team.
        for i in range(5):
            insert_game(
                db, f"carry_{i}", f"2025-12-{i + 1:02d}", "Carried", "mid", "pc", "strong", 1
            )
        # Self-made champion: 5 wins on a fresh team (no prior form).
        for i in range(5):
            insert_game(
                db, f"self_{i}", f"2025-12-{i + 1:02d}", "SelfMade", "mid", "pm", "fresh", 1
            )

        carried, _, _ = BulkStatsCache(db).pro_stats("Carried", "mid", "2026-01-15")
        self_made, _, _ = BulkStatsCache(db).pro_stats("SelfMade", "mid", "2026-01-15")
        assert self_made > carried


class TestTeamForm:
    def test_series_time_and_own_maps_control_visibility(self):
        db = make_grader_test_db()
        for sid, hour in (("early", 10), ("tie", 10), ("late", 12)):
            db.execute(
                "INSERT INTO series (series_id, team_a, team_b, scheduled_ts) VALUES (?, 'A', 'B', ?)",
                (sid, f"2026-01-01T{hour}:00:00+00:00"),
            )
        for gid, sid, number in (
            ("z", "early", 1),
            ("y", "early", 2),
            ("x", "tie", 1),
            ("a", "late", 1),
        ):
            db.execute(
                "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner) "
                "VALUES (?, ?, ?, 'A', 'B', 'A')",
                (gid, sid, number),
            )
        offsets = team_form_offsets(db, ["z", "y", "x", "a"])
        assert offsets[[0, 2]] == pytest.approx([0.0, 0.0])
        assert offsets[1] == pytest.approx(0.3646431135879092)
        assert offsets[3] > offsets[1]

    def test_fresh_team_is_prior(self):
        """A team's first game carries the 0.5 form prior."""
        rows = [("m1", "2025-01-01", "s1", 1, "A", "B", "A")]
        form = _build_team_form(rows)
        assert form[("A", "m1")] == pytest.approx(0.5)
        assert form[("B", "m1")] == pytest.approx(0.5)

    def test_winning_team_form_climbs(self):
        """Repeated wins push a team's trailing form above 0.5; losses below."""
        rows = [
            ("m1", "2025-01-01", "s1", 1, "A", "B", "A"),
            ("m2", "2025-01-02", "s2", 1, "A", "B", "A"),
            ("m3", "2025-01-03", "s3", 1, "A", "B", "A"),
        ]
        form = _build_team_form(rows)
        # A won m1, m2 before m3 -> form before m3 is above 0.5; B lost both.
        assert form[("A", "m3")] > 0.5
        assert form[("B", "m3")] < 0.5


class TestBulkCacheAgreesWithSqlPath:
    """The per-query SQL path and the vectorized BulkStatsCache agree on identical data."""

    def _seeded_db(self) -> sqlite3.Connection:
        """Varied corpus: 3 champions x roles, 2 players, plus a game on the as-of boundary."""
        db = make_grader_test_db()
        games = [
            # row shape: match_id, date, champion, role, player, result
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
            insert_game(db, match_id, date, champ, role, player, "t1", result)
        return db

    def test_pro_excess_agree(self):
        db = self._seeded_db()
        cache = BulkStatsCache(db)
        combos = cast(
            "list[tuple[str, Role]]",
            [("Jinx", "bot"), ("Ahri", "mid"), ("Ksante", "top"), ("Yone", "mid")],
        )  # Yone: absent champion, neutral in both paths
        for as_of in ("2026-03-01", "2026-01-15"):
            for champ, role in combos:
                sql = sql_pro_stats(db, champ, role, as_of)
                bulk = cache.pro_stats(champ, role, as_of)
                assert bulk == pytest.approx(sql, rel=1e-9, abs=1e-12), (
                    f"{champ}/{role} @ {as_of}: SQL {sql} != Bulk {bulk}"
                )

    def test_prefix_cache_excludes_every_same_day_row(self):
        """The fast prefix lookup preserves the strict-before-date leakage rule."""
        db = make_grader_test_db()
        # Deliberately insert the dates out of order; the cache must establish its
        # own chronological order rather than inheriting SQLite row order.
        insert_game(db, "future", "2026-03-02", "Jinx", "bot", "p1", "t1", 0)
        insert_game(db, "boundary_a", "2026-03-01", "Jinx", "bot", "p1", "t1", 1)
        insert_game(db, "past", "2026-02-28", "Jinx", "bot", "p1", "t1", 0)
        insert_game(db, "boundary_b", "2026-03-01", "Jinx", "bot", "p1", "t1", 1)

        cache = BulkStatsCache(db)
        sql = sql_pro_stats(db, "Jinx", "bot", "2026-03-01")
        bulk = cache.pro_stats("Jinx", "bot", "2026-03-01")

        assert bulk == pytest.approx(sql, rel=1e-9, abs=1e-12)
        # Only the single 2026-02-28 row is visible at the boundary.
        expected_weight = 2 ** (-1 / 30)
        assert bulk[2] == pytest.approx(expected_weight)
