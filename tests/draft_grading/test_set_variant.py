"""The format-4 serving pins: the exact mirror, the pre-draft half, swaps, batch shape."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from league_predictor.draft_grading.grader.predictor import FORWARD_ROWS, SwapRequest
from league_predictor.draft_grading.set_variant import swap_boards
from league_predictor.vocab.roles import MIRROR_SLOTS
from tests.draft_grading._set_helpers import (
    BiasedSetEncoder,
    DropoutSetEncoder,
    StubInputs,
    StubStats,
    mirrored,
    role_swapped,
    slot_row,
    stub_draft,
    stub_family,
    stub_index,
)
from tests.draft_grading.conftest import BENCH, BLUE_PICKS, RED_PICKS

STATS = StubStats()


@pytest.fixture(scope="module")
def index():
    return stub_index()


@pytest.fixture(scope="module")
def family(index):
    return stub_family(index)


# --- The served function: antisymmetric by construction ------------------------------


def test_a_mirrored_draft_scores_exactly_one_minus_p(family):
    """Bitwise, not to a tolerance: p is read once off the favoured side of one score."""
    draft = stub_draft(blue_bans=["Kalista"], red_bans=["Yone"])
    score = family.predict(draft, stats_cache=STATS)
    mirror = family.predict(mirrored(draft), stats_cache=STATS)
    assert mirror.prob_blue_wins == 1.0 - score.prob_blue_wins
    assert score.prob_blue_wins != 0.5  # a mirror pin on a flat scorer proves nothing
    assert (mirror.grade_blue, mirror.grade_red) == (score.grade_red, score.grade_blue)


def test_identical_teams_score_exactly_one_half(family):
    """No intercept reaches the served score, so a board against itself is the coin."""
    draft = stub_draft(BLUE_PICKS, BLUE_PICKS)
    assert family.predict(draft, stats_cache=STATS).prob_blue_wins == 0.5


def test_edge_pp_mirrors_the_score(family):
    draft = stub_draft()
    p = family.predict(draft, stats_cache=STATS).prob_blue_wins
    assert family.edge_pp(p) == -family.edge_pp(1.0 - p)
    assert family.grade_semantics == "edge"


def test_an_uncalibrated_family_refuses_to_price_an_edge(index):
    family = stub_family(index, calibrated=False)
    assert family.grade_semantics == "quantile"
    assert family.predict(stub_draft(), stats_cache=STATS).edge_pp_blue is None
    with pytest.raises(RuntimeError, match="no frozen edge curve"):
        family.edge_pp(0.6)


def test_a_seed_that_is_not_antisymmetric_is_refused_at_construction(index):
    """The mirror rests entirely on the half-swap negating the score; the gate reads it."""
    with pytest.raises(RuntimeError, match="seed 0"):
        stub_family(index, encoder_cls=BiasedSetEncoder)


# --- Batching: one shape per forward, so a row never reads its company ----------------


def test_predict_batch_row_equals_predict(family):
    drafts = [stub_draft(), stub_draft(BLUE_PICKS, RED_PICKS, blue_bans=["Kalista"])]
    drafts += [
        role_swapped(stub_draft(), "blue", role, champions[0]) for role, champions in BENCH.items()
    ]
    batched = [s.prob_blue_wins for s in family.predict_batch(drafts, stats_cache=STATS)]
    singles = [family.predict(d, stats_cache=STATS).prob_blue_wins for d in drafts]
    assert batched == singles
    assert len(set(batched)) > 1  # the drafts differ, so the pin is not comparing one number


def test_every_forward_runs_at_the_fixed_row_count(family, monkeypatch):
    """A row's score never depends on the batch: every forward is FORWARD_ROWS rows."""
    module = family.modules[0]
    real_forward = module.forward
    rows: list[int] = []

    def recording_forward(x):
        rows.append(int(x.shape[0]))
        return real_forward(x)

    monkeypatch.setattr(module, "forward", recording_forward)
    family.predict_batch([stub_draft(), mirrored(stub_draft())], stats_cache=STATS)
    family.predict(stub_draft(), stats_cache=STATS)
    assert rows
    assert set(rows) == {FORWARD_ROWS}


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity")
def test_cuda_seed_replay_is_exact_and_skips_python_forwards(monkeypatch):
    from tests.draft_grading.test_set_encoder import _encoder

    module = _encoder().cuda()
    family = stub_family(seeds=1)
    family.modules = (module,)
    x = torch.randn(FORWARD_ROWS + 1, 10, module.d_in)
    padded = torch.cat([x, torch.zeros(FORWARD_ROWS - 1, 10, module.d_in)]).cuda()
    with torch.no_grad():
        expected = torch.cat([module(block)[0] for block in padded.split(FORWARD_ROWS)])[: len(x)]
    calls = []
    forward = module.forward

    def recording(rows):
        calls.append(len(rows))
        return forward(rows)

    monkeypatch.setattr(module, "forward", recording)
    actual = family.seed_scores(x)
    count = len(calls)
    assert np.array_equal(actual[0], expected.cpu().numpy())
    assert np.array_equal(family.seed_scores(x[:1]), actual[:, :1])
    assert len(calls) == count
    assert np.array_equal(family.seed_scores(x[:, MIRROR_SLOTS]), -actual)


def test_a_module_left_in_training_mode_still_serves_one_number(index):
    """A fit hands the predictor its own modules; every forward evals, as ``_extract`` does."""
    family = stub_family(index, encoder_cls=DropoutSetEncoder)
    for module in family.modules:
        module.train()
    draft = stub_draft()
    first = family.predict(draft, stats_cache=STATS).prob_blue_wins
    for module in family.modules:
        module.train()
    assert family.predict(draft, stats_cache=STATS).prob_blue_wins == first


def test_batch_surfaces_refuse_a_missing_stats_cache(family):
    with pytest.raises(RuntimeError, match="predict_batch"):
        family.predict_batch([stub_draft()])
    with pytest.raises(RuntimeError, match="predict_swaps_marginalized_many"):
        family.predict_swaps_marginalized_many([(stub_draft(), [])])


# --- Swaps: real boards, one row each -------------------------------------------------


def test_a_swap_changes_only_its_own_slot_input_row(index):
    """The counterfactual board a swap scores differs from the original in two rows only."""
    draft = stub_draft()
    request = SwapRequest(
        side="blue",
        role="mid",
        alternatives=["Azir"],
        enemy_role="mid",
        enemy_responses=[("Orianna", 1.0)],
    )
    build = StubInputs(index)
    base = build([draft]).numpy()[0]
    swapped = build(swap_boards(draft, request)).numpy()[0]
    moved = {row for row in range(len(base)) if not np.array_equal(base[row], swapped[row])}
    assert moved == {slot_row("blue", "mid"), slot_row("red", "mid")}


def test_a_marginalized_swap_is_the_real_swapped_board(family):
    """One enemy response is the counter-pick query: it must equal scoring that board."""
    draft = stub_draft()
    alternatives = ["Azir", "Orianna", RED_PICKS["mid"]]
    got = family.predict_swaps(draft, "blue", "mid", alternatives, stats_cache=STATS)
    want = [
        family.predict(role_swapped(draft, "blue", "mid", alt), stats_cache=STATS).prob_blue_wins
        for alt in alternatives
    ]
    assert got.tolist() == pytest.approx(want, abs=1e-6)


def test_the_marginalized_swap_is_the_weighted_mean_over_the_enemy_responses(family):
    draft = stub_draft()
    responses = [("Azir", 0.7), ("Orianna", 0.3)]
    got = family.predict_swaps_marginalized(
        draft,
        "blue",
        "mid",
        ["Ahri"],
        enemy_role="mid",
        enemy_responses=responses,
        stats_cache=STATS,
    )
    want = sum(
        weight
        * family.predict(
            role_swapped(role_swapped(draft, "red", "mid", enemy), "blue", "mid", "Ahri"),
            stats_cache=STATS,
        ).prob_blue_wins
        for enemy, weight in responses
    )
    assert got[0] == pytest.approx(want, abs=1e-12)


def test_empty_swap_requests_answer_without_a_forward(family):
    draft = stub_draft()
    empty = SwapRequest("blue", "mid", [], "mid", [("Azir", 1.0)])
    no_reply = SwapRequest("blue", "mid", ["Azir", "Orianna"], "mid", [])
    rows = family.predict_swaps_marginalized_batch(draft, [empty, no_reply], stats_cache=STATS)
    assert rows[0].shape == (0,)
    assert rows[1].tolist() == [0.0, 0.0]


def test_a_swap_query_off_the_supported_shape_raises(family):
    with pytest.raises(ValueError, match="only enemy_role == role"):
        family.predict_swaps_marginalized(
            stub_draft(),
            "blue",
            "mid",
            ["Azir"],
            enemy_role="top",
            enemy_responses=[("Gnar", 1.0)],
            stats_cache=STATS,
        )
    with pytest.raises(ValueError, match="side must be"):
        family.predict_swaps(stub_draft(), "purple", "mid", ["Azir"], stats_cache=STATS)


# --- The design the leak audit cuts ---------------------------------------------------


def test_the_design_names_every_slot_column_slot_major(family):
    names = family.design_columns
    assert len(names) == 10 * len(StubInputs.COLUMNS)
    assert names[0] == "kit_0_blue_top"
    assert names[len(StubInputs.COLUMNS) * slot_row("red", "sup")] == "kit_0_red_sup"


def test_the_builder_mirrors_by_the_half_swap(index):
    """Every builder owes this over every column: the served mirror is only as strong."""
    draft = stub_draft(blue_bans=["Kalista"], red_bans=["Yone"])
    build = StubInputs(index)
    assert torch.equal(build([mirrored(draft)]), build([draft])[:, list(MIRROR_SLOTS)])


def test_empty_swap_queries_still_validate_the_side(family):
    with pytest.raises(ValueError, match="side must be"):
        family.predict_swaps_marginalized(
            stub_draft(),
            "purple",
            "mid",
            [],
            enemy_role="mid",
            enemy_responses=[],
            stats_cache=STATS,
        )
