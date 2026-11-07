"""DraftSuggestEngine behavior over fake predictors: roles, legality pools, minimax search."""

# file-length-ok: one module's contract mirror, split adds cross-file hops

from __future__ import annotations

import pytest

from league_predictor.draft_grading.suggest import (
    _MIN_CANDIDATE_N_EFF,
    DraftSuggestEngine,
    _playable_role_pools,
)
from league_predictor.vocab.roles import ROLE_ORDER


class _FakeScore:
    prob_blue_wins = 0.61
    grade_blue = "A"
    grade_red = "C"
    edge_pp_blue = 4.5


class _FakeIndex:
    def __init__(self, names):
        self.known_champions = sorted(names)


UNPICKED = "--"


def _engine_with_counts(counts, predictor=None):
    return DraftSuggestEngine(predictor, role_counts=counts, role_pools={})


class _FakePredictor:
    """Fake of the swap engine surface ``suggest`` drives; the standing draft predicts 0.5.

    Attributes:
        table: {(role, champion): p_blue} answering stage-1 sweeps on the picking side.
        replies: {(candidate, reply): p_blue} for stage-2, defaulting to the stage-1 value.
        direct: ``predict_batch`` values for a candidate-only board, defaulting to ``table``.
        direct_replies: ``predict_batch`` values for a candidate-plus-reply board.
        enemy_picks: the enemy's standing picks, so a placed reply is distinguishable.
    """

    def __init__(
        self,
        table,
        replies=None,
        picking_side="blue",
        our_picks=(),
        direct=None,
        direct_replies=None,
        enemy_picks=(),
    ):
        self._table = table
        self._replies = replies or {}
        self._picking = picking_side
        self._our_picks = set(our_picks)
        self._direct = direct or {}
        self._direct_replies = direct_replies or {}
        self._enemy_picks = set(enemy_picks)
        self.index = _FakeIndex({champ for _, champ in table})

    def predict(self, draft, stats_cache=None):
        try:
            self._placed_candidate(draft)
        except StopIteration:  # the standing draft, no candidate placed
            score = _FakeScore()
            score.prob_blue_wins = 0.5
            return score
        return self.predict_batch([draft], stats_cache)[0]

    def predict_batch(self, drafts, stats_cache=None):  # noqa: ARG002 - mirrors DraftPredictor.predict_batch; callers bind stats_cache by keyword
        out = []
        for draft in drafts:
            cand = self._placed_candidate(draft)
            reply = self._placed_reply(draft)
            score = _FakeScore()
            if reply is None:
                score.prob_blue_wins = self._direct.get(cand, self._table[cand])
            else:
                key = (cand[1], reply)
                score.prob_blue_wins = self._direct_replies.get(
                    key, self._replies.get(key, self._table[cand])
                )
            out.append(score)
        return out

    def _placed_candidate(self, draft):
        comp = draft.blue if self._picking == "blue" else draft.red
        return next(
            (s.role, s.champion_id)
            for s in comp.slots
            if s.champion_id != UNPICKED and s.champion_id not in self._our_picks
        )

    def _placed_reply(self, draft):
        comp = draft.red if self._picking == "blue" else draft.blue
        return next(
            (
                s.champion_id
                for s in comp.slots
                if s.champion_id != UNPICKED and s.champion_id not in self._enemy_picks
            ),
            None,
        )

    def predict_swaps_marginalized_many(self, entries, **_kw):
        out = []
        for draft, (req,) in entries:
            if req.side == self._picking:  # stage 1: alternatives are our candidates
                out.append([[self._table[(req.role, alt)] for alt in req.alternatives]])
            else:  # stage 2: alternatives are enemy replies to the placed candidate
                cand = self._placed_candidate(draft)
                out.append(
                    [
                        [
                            self._replies.get((cand[1], alt), self._table[cand])
                            for alt in req.alternatives
                        ]
                    ]
                )
        return out


def _suggest_engine(counts, pools, table, replies=None, side="blue", our_picks=(), **kw):
    return DraftSuggestEngine(
        _FakePredictor(table, replies, picking_side=side, our_picks=our_picks, **kw),
        role_counts=counts,
        role_pools=pools,
    )


# All-mid-main pool: every champion lands mid, so tests isolate ranking from roles.
_MID_COUNTS = {(c, "mid"): 100 for c in ("GlassCannon", "Solid", "Counter", "Filler")}
_MID_POOLS = {"mid": {"GlassCannon", "Solid", "Counter", "Filler"}}
_MID_TABLE = {
    ("mid", "GlassCannon"): 0.99,
    ("mid", "Solid"): 0.60,
    ("mid", "Counter"): 0.55,
    ("mid", "Filler"): 0.50,
}


def test_suggest_rollout_end_value_curates_and_orders_the_seats():
    # A pick is valued at the draft it leads to: GlassCannon wins 99% greedy but the enemy
    # holds its counter, so its realized end value collapses to 0.01 and it ranks last.
    replies = {("GlassCannon", "Counter"): 0.01, ("GlassCannon", "Solid"): 0.95}
    engine = _suggest_engine(_MID_COUNTS, _MID_POOLS, _MID_TABLE, replies)
    _, lines = engine.suggest([], [], [], [], side="blue")
    assert [line["champion"] for line in lines] == ["Solid", "Counter", "Filler", "GlassCannon"]
    glass = lines[-1]
    assert glass["p_blue_end"] == pytest.approx(0.01)  # where the draft actually lands
    assert glass["p_blue_now"] == pytest.approx(0.99)  # the board the moment it's picked
    assert glass["reply"] == {"champion": "Counter", "role": "mid"}
    # Solid's rollout: the enemy answers Counter (its best by default), draft
    # stalls (mid is the only pool), and the end board scores 0.60.
    assert lines[0]["p_blue_end"] == pytest.approx(0.60)

    # With only three seats, the countered trap is the one that falls off.
    _, cut = engine.suggest([], [], [], [], side="blue", top_n=3)
    assert [line["champion"] for line in cut] == ["Solid", "Counter", "Filler"]


def test_suggest_red_side_curates_by_its_own_perspective():
    # Red picking: the enemy (blue) reply MAXIMIZES p_blue. Counter saves blue
    # against red's GlassCannon, so red's top_n cut drops it symmetrically.
    replies = {("GlassCannon", "Counter"): 0.99}  # p_blue after blue's Counter reply
    table = {  # stage-1 p_blue: red wants LOW values
        ("mid", "GlassCannon"): 0.01,
        ("mid", "Solid"): 0.40,
        ("mid", "Counter"): 0.45,
        ("mid", "Filler"): 0.50,
    }
    engine = _suggest_engine(_MID_COUNTS, _MID_POOLS, table, replies, side="red")
    _, lines = engine.suggest([], [], [], [], side="red", top_n=3)
    assert [line["champion"] for line in lines] == ["Solid", "Counter", "Filler"]


def test_suggest_seats_by_exact_after_reply_value_not_the_swap_approximation():
    # The swap path's approximation error can reorder near-tied candidates, so seats follow
    # the exact forward value (0.70 here), not the swap-path value (0.30).
    replies = {("GlassCannon", "Counter"): 0.30}
    direct_replies = {("GlassCannon", "Counter"): 0.70}
    engine = _suggest_engine(
        _MID_COUNTS, _MID_POOLS, _MID_TABLE, replies, direct_replies=direct_replies
    )
    _, lines = engine.suggest([], [], [], [], side="blue", top_n=3)
    assert [line["champion"] for line in lines] == ["GlassCannon", "Solid", "Counter"]
    glass = lines[0]
    assert glass["p_blue"] == pytest.approx(0.70)
    assert glass["p_blue_now"] == pytest.approx(0.99)
    assert glass["reply"] == {"champion": "Counter", "role": "mid"}


def test_suggest_respects_taken_roles():
    # Mundo and Nidalee are jungle-only; with Mundo already picked, Nidalee has
    # no plausible seat and must not be suggested. Gnar (top) remains legal.
    counts = {("Mundo", "jng"): 900, ("Nidalee", "jng"): 800, ("Gnar", "top"): 500}
    pools = {"jng": {"Mundo", "Nidalee"}, "top": {"Gnar"}}
    table = {("top", "Gnar"): 0.55, ("jng", "Nidalee"): 0.70, ("jng", "Mundo"): 0.60}
    engine = _suggest_engine(counts, pools, table, side="red", our_picks=["Mundo"])
    _, lines = engine.suggest([], ["Mundo"], [], [], side="red")
    assert [line["champion"] for line in lines] == ["Gnar"]
    assert lines[0]["role"] == "top"


def test_suggest_reports_flex_move():
    # Mundo flexes top; drafting the dedicated jungler Nidalee moves him there,
    # and the line must say so.
    counts = {("Mundo", "jng"): 900, ("Mundo", "top"): 800, ("Nidalee", "jng"): 850}
    pools = {"jng": {"Mundo", "Nidalee"}, "top": {"Mundo"}}
    table = {("jng", "Nidalee"): 0.60, ("top", "Mundo"): 0.55, ("jng", "Mundo"): 0.55}
    engine = _suggest_engine(counts, pools, table, side="blue", our_picks=["Mundo"])
    _, lines = engine.suggest(["Mundo"], [], [], [], side="blue")
    nidalee = next(line for line in lines if line["champion"] == "Nidalee")
    assert nidalee["role"] == "jng"
    assert nidalee["moved"] == [{"champion": "Mundo", "from": "jng", "to": "top"}]


def test_suggest_no_reply_when_enemy_draft_complete():
    # Enemy has all five picks: no stage 2, greedy values, reply is None.
    enemy = ["E1", "E2", "E3", "E4", "E5"]
    counts = {**_MID_COUNTS, **dict.fromkeys(zip(enemy, ROLE_ORDER, strict=True), 50)}
    engine = _suggest_engine(counts, _MID_POOLS, _MID_TABLE, enemy_picks=enemy)
    _, lines = engine.suggest([], enemy, [], [], side="blue")
    assert [line["champion"] for line in lines] == ["GlassCannon", "Solid", "Counter", "Filler"]
    assert all(line["reply"] is None for line in lines)
    assert lines[0]["p_blue"] == pytest.approx(0.99)
    assert lines[0]["p_blue_now"] == pytest.approx(0.99)  # no reply: both values coincide
    # No teammate can follow (mid is the only pool) and the enemy is done, so
    # the rollout ends on this very board: end value == board value.
    assert lines[0]["p_blue_end"] == pytest.approx(0.99)


def test_suggest_p_blue_now_is_the_direct_board_value():
    # p_blue_now comes from the direct path the scoreboard runs, so a line shows exactly
    # the number the board will show once the pick lands.
    enemy = ["E1", "E2", "E3", "E4", "E5"]
    counts = {**_MID_COUNTS, **dict.fromkeys(zip(enemy, ROLE_ORDER, strict=True), 50)}
    direct = {key: p - 0.02 for key, p in _MID_TABLE.items()}
    engine = _suggest_engine(counts, _MID_POOLS, _MID_TABLE, direct=direct, enemy_picks=enemy)
    _, lines = engine.suggest([], enemy, [], [], side="blue")
    assert lines[0]["champion"] == "GlassCannon"
    assert lines[0]["p_blue_now"] == pytest.approx(0.97)
    assert lines[0]["p_blue"] == pytest.approx(0.97)  # no reply left: minimax IS the board value


def test_suggest_excludes_used_and_caps_top_n():
    counts = {(c, "mid"): 100 for c in "ABCDEFG"}
    table = {("mid", c): 0.5 + i / 100 for i, c in enumerate("ABCDEFG")}
    pools = {"mid": set("ABCDEFG")}
    engine = _suggest_engine(counts, pools, table)
    _, lines = engine.suggest([], [], ["G"], ["F"], side="blue", top_n=3)
    # G and F are banned; the remaining candidates rank E > D > C, capped at 3.
    assert [line["champion"] for line in lines] == ["E", "D", "C"]


def test_suggest_never_shoves_a_teammate_into_a_foreign_role():
    # Yuumi outbidding Thresh for support must not send Thresh to the jungle
    # (a role he doesn't play): the Yuumi candidacy is dropped entirely.
    counts = {("Thresh", "sup"): 11771, ("Thresh", "jng"): 6, ("Yuumi", "sup"): 3993}
    pools = {"sup": {"Thresh", "Yuumi"}}  # jng pool: nobody
    table = {("sup", "Yuumi"): 0.99, ("sup", "Thresh"): 0.55}
    engine = _suggest_engine(counts, pools, table, our_picks=["Thresh"])
    _, lines = engine.suggest(["Thresh"], [], [], [], side="blue")
    assert [line["champion"] for line in lines] == []


def test_playable_role_pools_needs_a_real_role_share():
    counts = {
        ("Rakan", "sup"): 16390,
        ("Rakan", "jng"): 26,  # over the 15-game floor, but 0.16% of his games
        ("Nidalee", "jng"): 4078,
        ("Nidalee", "top"): 108,  # 2.6% — a real pro flex, stays in
    }
    recent = dict(counts)  # everything current: recency is not the variable here
    pools = _playable_role_pools(counts, recent)
    assert "Rakan" not in pools["jng"]
    assert pools["sup"] == {"Rakan"}
    assert pools["jng"] == {"Nidalee"}
    assert pools["top"] == {"Nidalee"}


def test_playable_role_pools_drops_dead_regimes():
    # A dead regime clears every lifetime floor, so legality also requires games in the
    # role inside the trailing window.
    counts = {
        ("Rell", "sup"): 4200,
        ("Rell", "jng"): 610,  # 12.7% of her games — all from the dead regime
        ("Vi", "jng"): 3100,
    }
    recent = {
        ("Rell", "sup"): 180,
        ("Rell", "jng"): 0,  # no jungle games in the trailing window
        ("Vi", "jng"): 44,
    }
    pools = _playable_role_pools(counts, recent)
    assert pools["jng"] == {"Vi"}
    assert pools["sup"] == {"Rell"}

    # Two recent games is still below the recency floor; three clears it.
    recent[("Rell", "jng")] = 2
    assert "Rell" not in _playable_role_pools(counts, recent)["jng"]
    recent[("Rell", "jng")] = 3
    assert "Rell" in _playable_role_pools(counts, recent)["jng"]


def test_suggest_drops_candidates_below_the_evidence_floor():
    # A cell below the n_eff floor collapses the gate weight and washes the eval to the
    # tools side, so it must not be a candidate however good its board value looks.

    n_eff = dict.fromkeys(("Solid", "Counter", "Filler"), 100.0)
    n_eff["GlassCannon"] = _MIN_CANDIDATE_N_EFF - 1.0
    engine = DraftSuggestEngine(
        _FakePredictor(_MID_TABLE),
        role_counts=_MID_COUNTS,
        role_pools=_MID_POOLS,
        stats_cache=_StubStats(n_eff),
    )
    _, lines = engine.suggest([], [], [], [], side="blue")
    assert [line["champion"] for line in lines] == ["Solid", "Counter", "Filler"]


def test_suggest_falls_back_to_cold_candidates_when_none_are_evidenced():
    # Mass bans can exhaust the evidenced pool; an empty panel would stall the
    # rollout, so the floor relaxes rather than suggesting nothing.

    engine = DraftSuggestEngine(
        _FakePredictor(_MID_TABLE),
        role_counts=_MID_COUNTS,
        role_pools=_MID_POOLS,
        stats_cache=_StubStats({}),  # every candidate cold
    )
    _, lines = engine.suggest([], [], [], [], side="blue")
    assert [line["champion"] for line in lines] == ["GlassCannon", "Solid", "Counter", "Filler"]


class _RoleStats:
    """Per-(champion, role) n_eff stub — the evidence floor is a cell property."""

    def __init__(self, n_eff: dict[tuple[str, str], float]):
        self._n_eff = n_eff

    def pro_stats(self, champion, role, _as_of):
        return (0.0, 0.0, float(self._n_eff.get((champion, role), 0.0)))

    def n_eff(self, champion, role, as_of):
        return float(self.pro_stats(champion, role, as_of)[2])


def test_flex_move_through_a_cold_cell_is_dropped():
    # The evidence floor applies to the moved teammate's cell too: Nidalee would flex Mundo
    # into a cold top cell, so that line drops while Karma (moves nobody) stays.
    counts = {
        ("Mundo", "jng"): 900,
        ("Mundo", "top"): 800,
        ("Nidalee", "jng"): 850,
        ("Karma", "sup"): 500,
    }
    pools = {"jng": {"Mundo", "Nidalee"}, "top": {"Mundo"}, "sup": {"Karma"}}
    table = {("jng", "Nidalee"): 0.60, ("top", "Mundo"): 0.55, ("sup", "Karma"): 0.52}
    stats = _RoleStats(
        {("Nidalee", "jng"): 100.0, ("Mundo", "jng"): 100.0, ("Karma", "sup"): 100.0}
    )
    engine = DraftSuggestEngine(
        _FakePredictor(table, our_picks=["Mundo"]),
        role_counts=counts,
        role_pools=pools,
        stats_cache=stats,
    )
    _, lines = engine.suggest(["Mundo"], [], [], [], side="blue")
    assert [line["champion"] for line in lines] == ["Karma"]


def test_suggest_rejects_unknown_side():
    engine = _suggest_engine({}, {}, {})
    with pytest.raises(ValueError, match="side"):
        engine.suggest([], [], [], [], side="purple")


def test_assign_roles_prefers_played_roles():
    engine = _engine_with_counts(
        {
            ("Ahri", "mid"): 900,
            ("Ahri", "top"): 3,
            ("Nautilus", "sup"): 500,
            ("Nautilus", "jng"): 40,
        }
    )
    comp = engine._roles.assign(["Nautilus", "Ahri"])
    # Only picked roles appear; lifecycle's score_hypothetical_draft fills the rest.
    assert comp == {"mid": "Ahri", "sup": "Nautilus"}


def test_assign_roles_never_stacks_a_role():
    # Two mid-mains cannot both take mid — the weaker claim moves elsewhere.
    engine = _engine_with_counts({("Ahri", "mid"): 900, ("Orianna", "mid"): 800})
    comp = engine._roles.assign(["Ahri", "Orianna"])
    assert comp["mid"] == "Ahri"
    assert sorted(comp.values()) == ["Ahri", "Orianna"]


def test_score_rejects_illegal_drafts():
    engine = _engine_with_counts({})
    with pytest.raises(ValueError, match="more than 5"):
        engine.score(["A", "B", "C", "D", "E", "F"], [], [], [])
    with pytest.raises(ValueError, match="used twice"):
        engine.score(["Ahri"], ["Ahri"], [], [])
    with pytest.raises(ValueError, match="used twice"):
        engine.score(["Ahri"], [], [], ["Ahri"])


# ── stage 3: the full-draft rollout ──────────────────────────────────────


class _ScriptedPolicyEngine(DraftSuggestEngine):
    """Engine with the base policy scripted per (side, picks placed), recording every query."""

    def __init__(self, *args, script, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_calls = []
        self.policy_batches = []  # width of each _policy_picks call
        self._script = dict(script)  # (side, n_blue, n_red) -> champion | None

    def _policy_picks(self, jobs):
        self.policy_batches.append(len(jobs))
        out = []
        for blue_picks, red_picks, blue_bans, red_bans, side in jobs:
            self.policy_calls.append(
                (side, tuple(blue_picks), tuple(red_picks), tuple(blue_bans), tuple(red_bans))
            )
            out.append(self._script.get((side, len(blue_picks), len(red_picks))))
        return out


_ROLLOUT_RED = ["Rtop", "Rjng", "Rmid", "Rbot", "Rsup"]


def _scripted_engine(replies=None):
    """Blue picks from an empty board; the scripted enemy drafts _ROLLOUT_RED role by role."""
    counts = {("A", "mid"): 100, ("B", "mid"): 100}
    counts.update(dict.fromkeys(zip(_ROLLOUT_RED, ROLE_ORDER, strict=True), 100))
    pools = {"mid": {"A", "B"}}
    table = {("mid", "A"): 0.55, ("mid", "B"): 0.60}
    script = {("red", 1, n): champ for n, champ in enumerate(_ROLLOUT_RED)}
    return _ScriptedPolicyEngine(
        _FakePredictor(table, replies),
        role_counts=counts,
        role_pools=pools,
        script=script,
    )


def test_rollout_follows_the_snake_order_and_threads_bans():
    # After blue's candidate lands (board 1-0) the remaining turns are R R B B R R B B R;
    # a side with no pick skips, and each distinct state is asked once (memo).
    engine = _scripted_engine(replies={("A", "Rtop"): 0.61, ("B", "Rtop"): 0.41})
    _, lines = engine.suggest([], [], ["Zed"], [], side="blue")
    assert [ln["champion"] for ln in lines] == ["A", "B"]
    assert lines[0]["p_blue_end"] == pytest.approx(0.61)
    assert lines[1]["p_blue_end"] == pytest.approx(0.41)

    a_calls = [c for c in engine.policy_calls if c[1] == ("A",)]
    assert [c[0] for c in a_calls] == ["red", "red", "blue", "red", "red", "blue", "red"]
    # The enemy board grows through the scripted picks, and bans ride along.
    assert a_calls[1][2] == ("Rtop",)
    assert a_calls[-1][2] == tuple(_ROLLOUT_RED[:4])
    assert all(c[3] == ("Zed",) for c in a_calls)


def test_suggest_without_rollout_is_the_fast_one_ply_search():
    # rollout=False returns one-ply order with no p_blue_end; the frontend fills end values
    # in via one batched rollout_values call.
    replies = {("GlassCannon", "Counter"): 0.01, ("GlassCannon", "Solid"): 0.95}
    engine = _suggest_engine(_MID_COUNTS, _MID_POOLS, _MID_TABLE, replies)
    _, lines = engine.suggest([], [], [], [], side="blue", rollout=False)
    assert [line["champion"] for line in lines] == ["GlassCannon", "Solid", "Counter", "Filler"]
    assert all("p_blue_end" not in line for line in lines)


def test_rollout_value_is_one_candidates_end_value():
    # The single-candidate seam: same number the blocking suggest attaches as
    # p_blue_end.
    replies = {("GlassCannon", "Counter"): 0.01, ("GlassCannon", "Solid"): 0.95}
    engine = _suggest_engine(_MID_COUNTS, _MID_POOLS, _MID_TABLE, replies)

    def end_value(champion: str) -> float:
        return engine.rollout_value([], [], [], [], side="blue", champion=champion)

    assert end_value("GlassCannon") == pytest.approx(0.01)
    assert end_value("Solid") == pytest.approx(0.60)


def test_rollout_values_matches_sequential_and_aligns_to_candidates():
    # The dashboard's batched seam: one lockstep call returns exactly the
    # numbers the per-candidate reads produce, aligned to the request order.
    replies = {("GlassCannon", "Counter"): 0.01, ("GlassCannon", "Solid"): 0.95}
    engine = _suggest_engine(_MID_COUNTS, _MID_POOLS, _MID_TABLE, replies)
    batched = engine.rollout_values([], [], [], [], side="blue", champions=["Solid", "GlassCannon"])
    assert batched == [pytest.approx(0.60), pytest.approx(0.01)]
    assert engine.rollout_values([], [], [], [], side="blue", champions=[]) == []


def test_rollout_values_advances_all_boards_in_lockstep():
    # Two candidates roll out together: every ply's un-memoized policy queries
    # arrive as ONE batch (width 2), never one candidate at a time.
    engine = _scripted_engine()
    engine.rollout_values([], [], [], [], side="blue", champions=["A", "B"])
    assert engine.policy_batches
    assert all(width == 2 for width in engine.policy_batches)
    # A repeat batch replays entirely from the memo — no new policy queries.
    n_calls = len(engine.policy_calls)
    engine.rollout_values([], [], [], [], side="blue", champions=["A", "B"])
    assert len(engine.policy_calls) == n_calls


def test_rollout_values_rejects_illegal_candidates():
    engine = _suggest_engine(_MID_COUNTS, _MID_POOLS, _MID_TABLE)
    with pytest.raises(ValueError, match="used twice"):
        engine.rollout_values(["Solid"], [], [], [], side="red", champions=["Filler", "Solid"])


def test_rollout_value_rejects_illegal_candidates():
    engine = _suggest_engine(_MID_COUNTS, _MID_POOLS, _MID_TABLE)
    with pytest.raises(ValueError, match="used twice"):
        engine.rollout_value(["Solid"], [], [], [], side="red", champion="Solid")
    with pytest.raises(ValueError, match="side"):
        engine.rollout_value([], [], [], [], side="purple", champion="Solid")


def test_rollout_policy_is_memoized_across_suggest_calls():
    engine = _scripted_engine()
    engine.suggest([], [], [], [], side="blue")
    n_first = len(engine.policy_calls)
    assert n_first > 0
    engine.suggest([], [], [], [], side="blue")
    assert len(engine.policy_calls) == n_first  # every state replayed from the memo

    # Bans change the game — a different ban set must not answer from the memo.
    engine.suggest([], [], ["Zed"], [], side="blue")
    assert len(engine.policy_calls) > n_first


# ── blend serving (format-3) through the war-room engine ─────────────────


class _CacheCapture:
    """A format-2 predictor stand-in recording the stats_cache each predict saw."""

    def __init__(self):
        self.seen = []
        self.index = _FakeIndex([])

    def predict(self, _draft, stats_cache=None):
        self.seen.append(stats_cache)
        return _FakeScore()


def test_score_threads_the_selected_score_cache():
    pred = _CacheCapture()
    sentinel = object()
    engine = DraftSuggestEngine(pred, role_counts={}, role_pools={}, score_cache=sentinel)
    engine.score([], [], [], [])
    assert pred.seen == [sentinel]


# Five enemy picks (one per role) close the enemy draft so suggest skips the
# stage-2 refinement and the stage-1 blend ranking is what's asserted.
_ENEMY = {f"E{i}": role for i, role in enumerate(ROLE_ORDER)}


class _StubStats:
    """A BulkStatsCache stand-in: ``pro_stats`` third element is the decayed n_eff."""

    def __init__(self, n_eff: dict[str, float]):
        self._n_eff = n_eff

    def pro_stats(self, champion, _role, _as_of):
        return (0.0, 0.0, float(self._n_eff.get(champion, 0.0)))

    def n_eff(self, champion, role, as_of):
        return float(self.pro_stats(champion, role, as_of)[2])
