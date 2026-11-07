"""Tests for the skill-filter corpus builder: day tensors, masks, walk-forward bookkeeping."""

from __future__ import annotations

from datetime import date

import pytest

from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
from league_predictor.predraft.ratings.skill_filter.corpus_schema import CTX_CHANNELS
from tests.predraft.conftest import facts_db_from_legacy
from tests.predraft.ratings.conftest import _five, _match


def _db(matches, player_games):
    """In-memory facts DB from the toy row tuples (see ``facts_db_from_legacy``)."""
    return facts_db_from_legacy(matches, player_games)


def _five_named(mid, tid, pids, **kw):
    """Five player rows with explicit player ids in role order."""
    rows = _five(mid, tid, "ignored", **kw)
    return [(mid, tid, pid, *row[3:]) for pid, row in zip(pids, rows, strict=True)]


class TestGapChurnDebutPrecompute:
    def test_gap_churn_and_debut_on_toy_history(self):
        # Day 1: AA (a1..a5) beats BB (b1..b5). Day 10: AA swaps sup a5 -> a6.
        aa1 = [f"a{i}" for i in range(1, 6)]
        aa2 = [f"a{i}" for i in (1, 2, 3, 4, 6)]
        bb = [f"b{i}" for i in range(1, 6)]
        conn = _db(
            [
                _match("m1", "2026-01-01", "AA", "BB", "AA"),
                _match("m2", "2026-01-10", "AA", "BB", "BB", patch="14.02"),
            ],
            _five_named("m1", "AA", aa1)
            + _five_named("m1", "BB", bb, side="Red")
            + _five_named("m2", "AA", aa2)
            + _five_named("m2", "BB", bb, side="Red"),
        )
        corpus = build_corpus(conn, through=date(2026, 2, 1), norm_through=date(2026, 2, 1))

        assert [d.date for d in corpus.days] == [date(2026, 1, 1), date(2026, 1, 10)]
        d1, d2 = corpus.days

        # Reference (sorted) orientation: AA is side 0, and a_won is AA-relative.
        assert d1.a_won.tolist() == [1.0]
        assert d2.a_won.tolist() == [0.0]

        # Day 1 is everyone's debut: gaps masked out, debut flags set.
        assert d1.debut[0].tolist() == [[1.0] * 5, [1.0] * 5]

        # Day 10: AA's four returning starters carry a 9-day gap; a6 debuts.
        assert d2.gaps[0, 0, :4].tolist() == [9.0] * 4
        assert d2.debut[0, 0].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0]
        assert d2.debut[0, 1].tolist() == [0.0] * 5

        # Churn counts swapped starters vs the team's last fielded five.
        churn = CTX_CHANNELS.index("churn_count")
        assert d2.ctx[0, 0, churn] == 1.0
        assert d2.ctx[0, 1, churn] == 0.0

        # Vocab covers exactly the 11 distinct players.
        assert corpus.vocab.n_players == 11


class TestRosterContinuity:
    """roster_cont counts start-of-day consecutive prior games of the team's current five."""

    def test_continuity_counts_consecutive_games_of_the_same_five(self):
        aa = [f"a{i}" for i in range(1, 6)]
        bb = [f"b{i}" for i in range(1, 6)]
        conn = _db(
            [
                _match("m1", "2026-01-01", "AA", "BB", "AA"),
                _match("m2", "2026-01-10", "AA", "BB", "AA"),
                _match("m3", "2026-01-20", "AA", "BB", "AA"),
            ],
            _five_named("m1", "AA", aa)
            + _five_named("m1", "BB", bb, side="Red")
            + _five_named("m2", "AA", aa)
            + _five_named("m2", "BB", bb, side="Red")
            + _five_named("m3", "AA", aa)
            + _five_named("m3", "BB", bb, side="Red"),
        )
        d1, d2, d3 = build_corpus(
            conn, through=date(2026, 2, 1), norm_through=date(2026, 2, 1)
        ).days
        # Reference orientation: AA is side 0. Prior consecutive games of the current five.
        assert d1.roster_cont[0, 0].item() == 0.0  # debut of the five
        assert d2.roster_cont[0, 0].item() == 1.0
        assert d3.roster_cont[0, 0].item() == 2.0

    def test_continuity_resets_on_roster_change(self):
        aa = [f"a{i}" for i in range(1, 6)]
        aa2 = [f"a{i}" for i in (1, 2, 3, 4, 6)]  # sup a5 -> a6
        bb = [f"b{i}" for i in range(1, 6)]
        conn = _db(
            [
                _match("m1", "2026-01-01", "AA", "BB", "AA"),
                _match("m2", "2026-01-10", "AA", "BB", "AA"),
                _match("m3", "2026-01-20", "AA", "BB", "AA"),
                _match("m4", "2026-02-01", "AA", "BB", "AA"),
            ],
            _five_named("m1", "AA", aa)
            + _five_named("m1", "BB", bb, side="Red")
            + _five_named("m2", "AA", aa)
            + _five_named("m2", "BB", bb, side="Red")
            + _five_named("m3", "AA", aa2)
            + _five_named("m3", "BB", bb, side="Red")
            + _five_named("m4", "AA", aa2)
            + _five_named("m4", "BB", bb, side="Red"),
        )
        _d1, d2, d3, d4 = build_corpus(
            conn, through=date(2026, 3, 1), norm_through=date(2026, 3, 1)
        ).days
        assert d2.roster_cont[0, 0].item() == 1.0  # AA's five settled one game
        assert d3.roster_cont[0, 0].item() == 0.0  # the swap resets the run
        assert d4.roster_cont[0, 0].item() == 1.0  # the new five now has one game together
        assert d4.roster_cont[0, 1].item() == 3.0  # BB never changed: m1/m2/m3 prior

    def test_continuity_zero_for_incomplete_five(self):
        aa = [f"a{i}" for i in range(1, 6)]
        bb = [f"b{i}" for i in range(1, 6)]
        conn = _db(
            [
                _match("m1", "2026-01-01", "AA", "BB", "AA"),
                _match("m2", "2026-01-10", "AA", "BB", "AA"),
            ],
            _five_named("m1", "AA", aa)
            + _five_named("m1", "BB", bb, side="Red")
            + _five_named("m2", "AA", aa)[:4]  # AA fields only four known at m2
            + _five_named("m2", "BB", bb, side="Red"),
        )
        _d1, d2 = build_corpus(conn, through=date(2026, 2, 1), norm_through=date(2026, 2, 1)).days
        assert d2.roster_cont[0, 0].item() == 0.0  # incomplete five -> synergy unknown


class TestLineupStatsProjection:
    """_lineup_stats projects game_players rows to _LINEUP_STAT_KEYS with verbatim values."""

    def _toy_db(self):
        aa = [f"a{i}" for i in range(1, 6)]
        aa2 = [f"a{i}" for i in (1, 2, 3, 4, 6)]
        bb = [f"b{i}" for i in range(1, 6)]
        return _db(
            [
                _match("m1", "2026-01-01", "AA", "BB", "AA"),
                _match("m2", "2026-01-10", "AA", "BB", "BB", patch="14.02"),
            ],
            _five_named("m1", "AA", aa)
            + _five_named("m1", "BB", bb, side="Red")
            + _five_named("m2", "AA", aa2)
            + _five_named("m2", "BB", bb, side="Red"),
        )

    def test_projected_keys_cover_every_downstream_read(self):
        # The projection must carry every key any downstream consumer reads off a stats dict:
        # the player channels plus the team-kill/objective columns _team_channels reads.
        from league_predictor.predraft.ratings.skill_filter.corpus_build import (
            _LINEUP_STAT_KEYS,
            PLAYER_CHANNELS,
        )

        consumed = set(PLAYER_CHANNELS) | {
            "teamkills",
            "golddiffat15",
            "firsttower",
            "firstdragon",
            "firstbaron",
        }
        assert consumed.issubset(set(_LINEUP_STAT_KEYS))

    def test_corpus_tensors_identical_to_full_dict_projection(self):
        # Monkeypatch _lineup_stats to return full dict(r) rows (the pre-optimization behavior)
        # and assert every corpus day tensor is bit-for-bit identical to the projected build.
        import torch

        from league_predictor.predraft.ratings.skill_filter import (
            corpus_build as corpus_mod,
        )

        projected = build_corpus(
            self._toy_db(), through=date(2026, 2, 1), norm_through=date(2026, 2, 1)
        )

        orig = corpus_mod._lineup_stats

        def _full_dict_lineup_stats(conn, *, through):
            out = orig(conn, through=through)  # reuse the SQL + dedup, then widen to full rows
            cols = ", ".join(f"gp.{c}" for c in corpus_mod.PLAYER_CHANNELS)
            rows = conn.execute(
                f"""SELECT gp.game_id AS match_id, gp.team_id, gp.player_id, gp.role,
                           gp.teamkills, gp.teamdeaths,
                           gp.firsttower, gp.firstdragon, gp.firstbaron, {cols}
                    FROM game_players gp
                    JOIN games g ON g.game_id = gp.game_id
                    JOIN series s ON s.series_id = g.series_id
                    WHERE g.winner IS NOT NULL AND s.scheduled_ts < ?
                          AND gp.role IS NOT NULL""",
                (through.isoformat(),),
            ).fetchall()
            full: dict = {}
            for r in rows:
                key = (str(r["match_id"]), str(r["team_id"]))
                full.setdefault(key, {}).setdefault(str(r["role"]), (str(r["player_id"]), dict(r)))
            assert full.keys() == out.keys()
            return full

        corpus_mod._lineup_stats = _full_dict_lineup_stats
        try:
            full = build_corpus(
                self._toy_db(), through=date(2026, 2, 1), norm_through=date(2026, 2, 1)
            )
        finally:
            corpus_mod._lineup_stats = orig

        assert [d.date for d in projected.days] == [d.date for d in full.days]
        tensor_fields = (
            "players",
            "orgs",
            "a_won",
            "gaps",
            "debut",
            "roster_cont",
            "team_gaps",
            "perf",
            "perf_mask",
            "team_feat",
            "team_mask",
            "ctx",
        )
        for dp, df in zip(projected.days, full.days, strict=True):
            for f in tensor_fields:
                assert torch.equal(getattr(dp, f), getattr(df, f)), f
        # NormStats ride alongside -- the player-channel stats must match too.
        assert torch.equal(projected.norm.player_mean, full.norm.player_mean)
        assert torch.equal(projected.norm.team_mean, full.norm.team_mean)


class TestNormStats:
    def _two_day_corpus(self, norm_through: date):
        # Day 1 gd15 = 100 everywhere; day 10 gd15 = 500 everywhere.
        aa = [f"a{i}" for i in range(1, 6)]
        bb = [f"b{i}" for i in range(1, 6)]
        conn = _db(
            [
                _match("m1", "2026-01-01", "AA", "BB", "AA"),
                _match("m2", "2026-01-10", "AA", "BB", "BB"),
            ],
            _five_named("m1", "AA", aa, gd15=100.0)
            + _five_named("m1", "BB", bb, gd15=100.0, side="Red")
            + _five_named("m2", "AA", aa, gd15=500.0)
            + _five_named("m2", "BB", bb, gd15=500.0, side="Red"),
        )
        return build_corpus(conn, through=date(2026, 2, 1), norm_through=norm_through)

    def test_norm_stats_ignore_games_after_norm_through(self):
        from league_predictor.predraft.ratings.skill_filter.corpus_schema import PLAYER_CHANNELS

        ch = PLAYER_CHANNELS.index("golddiffat15")

        early = self._two_day_corpus(date(2026, 1, 5))
        assert early.norm.player_mean[ch].item() == pytest.approx(100.0)

        full = self._two_day_corpus(date(2026, 2, 1))
        assert full.norm.player_mean[ch].item() == pytest.approx(300.0)
        # Raw tensors stay raw — normalization is never baked into the corpus.
        assert early.days[0].perf[0, 0, 0, ch].item() == pytest.approx(100.0)
