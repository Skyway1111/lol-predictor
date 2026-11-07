"""The set fit's pins: bit-for-bit reproduction, the tail-only early stop, the offset, the arms."""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest
import torch

from league_predictor.draft_grading.grader.dense_labels import DenseTargets
from league_predictor.draft_grading.grader.set_encoder import DraftSetEncoder
from league_predictor.draft_grading.grader.set_training import (
    SetWeights,
    _batch_terms,
    _Corpus,
    _split_at,
    fit_set_encoder,
    set_scores,
)
from league_predictor.draft_grading.grader.slot_inputs import SlotInputs
from league_predictor.draft_grading.recipe import AUX_HEAD_NAMES, DRAFT_RECIPE, SetTrainConfig
from league_predictor.vocab.roles import MIRROR_SLOTS, N_SLOTS

#: A small, fast arm of the production recipe; every pin below is structural, not a read.
FIT = dc.replace(DRAFT_RECIPE.set_recipe(), seeds=2, max_epochs=3, batch_size=32, device="cpu")
N_GAMES = 200


def _corpus(config=FIT, seed: int = 4):
    """Seeded slot rows, outcomes, offsets and dense labels of the right shapes."""
    rng = np.random.default_rng(seed)
    d_in = SlotInputs.d_in(config)
    x = torch.from_numpy(rng.normal(size=(N_GAMES, N_SLOTS, d_in)).astype(np.float32))
    outcomes = (rng.random(N_GAMES) < 0.5).astype(np.float64)
    offsets = rng.normal(scale=0.5, size=N_GAMES)
    labels = {
        "dpm": rng.normal(400, 80, (N_GAMES, N_SLOTS)).astype(np.float32),
        "damage_share": rng.random((N_GAMES, N_SLOTS)).astype(np.float32),
        "class": rng.integers(0, 30, (N_GAMES, N_SLOTS)),
        "gd15": rng.normal(0, 900, N_GAMES).astype(np.float32),
        "long_game": (rng.random(N_GAMES) < 0.5).astype(np.float32),
    }
    masks = {name: np.ones(value.shape, dtype=bool) for name, value in labels.items()}
    masks["long_game"][::3] = False
    return x, outcomes, offsets, DenseTargets(labels, masks)


def _fit(config=FIT):
    x, y, offsets, dense = _corpus(config)
    return fit_set_encoder(x, y, offsets, dense, config, device="cpu")


@pytest.fixture(scope="module")
def fitted() -> SetWeights:
    return _fit()


# --- Reproduction ---------------------------------------------------------------------


def test_a_synthetic_fit_reproduces_bit_for_bit(fitted):
    again = _fit()
    assert [report["best_epoch"] for report in again.history] == [
        report["best_epoch"] for report in fitted.history
    ]
    for first, second in zip(fitted.states, again.states, strict=True):
        assert set(first) == set(second)
        assert all(torch.equal(first[key], second[key]) for key in first)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph parity")
@pytest.mark.parametrize("sparse", [False, True])
def test_graphed_training_preserves_every_weight_and_loss(monkeypatch, sparse):
    config = dc.replace(FIT, seeds=1, device="cuda")
    x, y, offsets, dense = _corpus(config)
    if sparse:
        dense.masks["gd15"][1:] = False
    capture = torch.cuda.make_graphed_callables
    with monkeypatch.context() as eager:
        eager.setattr(torch.cuda, "make_graphed_callables", lambda model, *_a, **_kw: model)
        expected = fit_set_encoder(x, y, offsets, dense, config, epochs=2)
    captures = []

    def recording(model, *args, **kwargs):
        captures.append(model)
        return capture(model, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", recording)
    actual = fit_set_encoder(x, y, offsets, dense, config, epochs=2)
    assert len(captures) == 1
    assert actual.history == expected.history
    assert all(torch.equal(v, actual.states[0][k]) for k, v in expected.states[0].items())


def test_the_fit_carries_one_state_per_seed_and_its_own_input_spec(fitted):
    assert len(fitted.states) == FIT.seeds == len(fitted.history)
    assert [block["name"] for block in fitted.input_spec["blocks"]] == [
        "role",
        "kit",
        "power",
        "pro_form",
    ]
    assert len(fitted.input_spec["columns"]) == fitted.d_in == SlotInputs.d_in(FIT)
    assert fitted.config["outcome_offset"] == "team_form"
    assert SetTrainConfig.from_dict(fitted.config) == dc.replace(FIT, device=None)


def test_a_loader_rebuilds_the_layout_and_the_config_from_the_payload_alone(fitted):
    """The serving path reads the artifact, never source (`TRAPS.md`, the artifact-over-source rule)."""
    rebuilt = SlotInputs.from_spec(fitted.input_spec)
    assert rebuilt.columns() == SlotInputs(FIT).columns()
    assert rebuilt.width == fitted.d_in
    assert SetTrainConfig.from_dict(fitted.config).aux_heads == FIT.aux_heads


def test_the_weights_round_trip_through_a_weights_only_payload(fitted, tmp_path):
    """The format-4 save writes this payload, so it holds tensors and primitives alone."""
    path = tmp_path / "weights.pt"
    torch.save(fitted.to_state(), path)
    back = SetWeights(**torch.load(path, weights_only=True))
    assert (back.d_in, back.n_champions) == (fitted.d_in, fitted.n_champions)
    assert back.input_spec == fitted.input_spec
    assert back.config == fitted.config == DRAFT_RECIPE.set_recipe().to_dict() | back.config
    assert back.label_scale == fitted.label_scale
    assert torch.equal(back.states[0]["score.weight"], fitted.states[0]["score.weight"])


def test_the_served_score_is_the_seed_mean_and_mirrors_exactly(fitted):
    x, *_ = _corpus()
    scores = set_scores(fitted, FIT, x[:16])
    mirrored = set_scores(fitted, FIT, x[:16][:, list(MIRROR_SLOTS)])
    assert np.array_equal(mirrored, -scores)
    members = []
    for state in fitted.states:
        model = DraftSetEncoder(FIT, fitted.d_in, fitted.n_champions)
        model.load_state_dict(state)
        with torch.no_grad():
            members.append(model.eval()(x[:16])[0])
    assert np.allclose(scores, torch.stack(members).mean(dim=0).numpy(), atol=1e-6)


@pytest.mark.parametrize("n", [1, 3, 7, 129])
def test_a_served_score_is_the_same_in_any_batch(fitted, n):
    """Every forward runs at a fixed row count, the tail padded, so no row reads its company."""
    x, *_ = _corpus()
    assert np.array_equal(set_scores(fitted, FIT, x[:n]), set_scores(fitted, FIT, x)[:n])


def test_a_forward_width_off_the_pinned_row_count_is_refused(fitted):
    x, *_ = _corpus()
    with pytest.raises(ValueError, match="multiple of FORWARD_ROWS"):
        set_scores(fitted, FIT, x[:4], chunk=300)


# --- The early stop -------------------------------------------------------------------


def test_the_early_stop_reads_only_the_chronological_tail(fitted):
    x, y, offsets, dense = _corpus()
    split = _split_at(N_GAMES)
    tail_flipped = y.copy()
    tail_flipped[split:] = 1.0 - tail_flipped[split:]
    moved = fit_set_encoder(x, tail_flipped, offsets, dense, FIT, device="cpu")
    assert [report["train"] for report in moved.history] == [
        report["train"] for report in fitted.history
    ]
    assert [report["val_main"] for report in moved.history] != [
        report["val_main"] for report in fitted.history
    ]


def test_the_fit_slice_is_the_first_ninety_percent():
    assert _split_at(200) == 180
    assert _split_at(1) == 1
    assert _split_at(3) == 2


# --- The offset -----------------------------------------------------------------------


def _terms(offsets):
    x, y, _, dense = _corpus()
    torch.manual_seed(0)
    model = DraftSetEncoder(FIT, SlotInputs.d_in(FIT))
    torch.nn.init.normal_(model.score.weight)
    corpus = _Corpus(
        x,
        outcomes=y,
        offsets=offsets,
        labels=dense.labels,
        masks=dense.masks,
        weights=np.ones(N_GAMES),
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        return _batch_terms(model, corpus, *next(corpus.batches(np.arange(32), 32)))


def test_prepared_batches_preserve_masked_losses_with_shuffle_and_tail():
    from league_predictor.draft_grading.grader.set_training import _head_loss

    x, y, offsets, dense = _corpus()
    dense.masks["gd15"][:] = False
    dense.masks["dpm"][::3, ::2] = False
    corpus = _Corpus(
        x,
        outcomes=y,
        offsets=offsets,
        labels=dense.labels,
        masks=dense.masks,
        weights=np.ones(N_GAMES),
        device=torch.device("cpu"),
    )
    order = np.random.default_rng(17).permutation(N_GAMES)
    for idx, selections in corpus.batches(order, 32):
        target = corpus.labels["dpm"][idx]
        prediction = target + torch.arange(10)
        mask = torch.from_numpy(dense.masks["dpm"][idx.numpy()])
        expected = torch.nn.functional.mse_loss(prediction[mask], target[mask])
        assert torch.equal(_head_loss("dpm", prediction, target, selections["dpm"]), expected)
        assert _head_loss("gd15", corpus.y[idx], corpus.y[idx], selections["gd15"]) is None


def test_the_offset_enters_the_main_head_and_nothing_else():
    _, _, offsets, _ = _corpus()
    with_offset, without = _terms(offsets), _terms(None)
    assert float(with_offset["main"]) != float(without["main"])
    for head in AUX_HEAD_NAMES:
        assert float(with_offset[head]) == float(without[head])


def test_no_offset_reaches_the_served_score(fitted):
    """`set_scores` takes the rows alone; the offset has no parameter to ride."""
    x, *_ = _corpus()
    assert np.array_equal(set_scores(fitted, FIT, x[:8]), set_scores(fitted, FIT, x[:8]))


def test_an_offset_without_the_recipe_field_is_refused():
    x, y, offsets, dense = _corpus()
    with pytest.raises(ValueError, match="travel together"):
        fit_set_encoder(x, y, offsets, dense, dc.replace(FIT, outcome_offset="none"), device="cpu")
    with pytest.raises(ValueError, match="travel together"):
        fit_set_encoder(x, y, None, dense, FIT, device="cpu")


def test_a_configured_head_without_labels_is_refused():
    x, y, offsets, _ = _corpus()
    with pytest.raises(ValueError, match="no dense labels"):
        fit_set_encoder(x, y, offsets, None, FIT, device="cpu")


# --- The six arms ---------------------------------------------------------------------

V7 = DRAFT_RECIPE.set_recipe()
ARMS = {
    "full": V7,
    "no_power": dc.replace(V7, power=False),
    "no_heads": dc.replace(V7, aux_heads=()),
    "no_pro_form": dc.replace(V7, pro_form=False),
    "identity_residual": dc.replace(V7, identity_residual=True),
    "plain_outcome": dc.replace(V7, outcome_offset="none"),
}


@pytest.mark.parametrize("name", [n for n in ARMS if n != "full"])
def test_each_arm_differs_from_full_by_exactly_one_recipe_field(name):
    full, arm = V7.to_dict(), ARMS[name].to_dict()
    assert [key for key in full if full[key] != arm[key]] == [
        {
            "no_power": "power",
            "no_heads": "aux_heads",
            "no_pro_form": "pro_form",
            "identity_residual": "identity_residual",
            "plain_outcome": "outcome_offset",
        }[name]
    ]


@pytest.mark.parametrize("name", list(ARMS))
def test_every_arm_fits_a_synthetic_corpus(name):
    arm = dc.replace(ARMS[name], seeds=1, max_epochs=1, batch_size=32, device="cpu")
    x, y, offsets, dense = _corpus(arm)
    if arm.identity_residual:
        x[..., -1] = torch.randint(0, 24, (N_GAMES, N_SLOTS)).float()
    weights = fit_set_encoder(
        x,
        y,
        offsets if arm.outcome_offset != "none" else None,
        dense if arm.aux_heads else None,
        arm,
        n_champions=25 if arm.identity_residual else 0,
        device="cpu",
    )
    assert weights.d_in == SlotInputs.d_in(arm)
    assert set_scores(weights, arm, x[:4]).shape == (4,)


# --- training/set_fit.py: the corpus, the knowledge, the folds, the final fit ------------


SMOKE = dc.replace(
    DRAFT_RECIPE.set_recipe(),
    power=False,
    seeds=1,
    max_epochs=4,
    batch_size=64,
    device="cpu",
    outcome_since="2025-01-01",
)


@pytest.fixture(scope="module")
def seeded_db():
    from tests.draft_grading.grader.conftest import make_grader_test_db

    return make_grader_test_db(game_count=120)


@pytest.fixture(scope="module")
def fitted_candidate(seeded_db):
    from datetime import date

    from league_predictor.draft_grading.training.set_fit import fit_set_candidate

    return fit_set_candidate(seeded_db, SMOKE, through=date(2026, 1, 1))


def test_the_knowledge_member_carries_the_blocks_the_config_names():
    from league_predictor.draft_grading.training.set_fit import build_set_knowledge

    knowledge, index = build_set_knowledge(SMOKE)
    assert knowledge.kit is not None
    assert knowledge.power is None
    assert knowledge.kit.shape[0] == len(index)
    assert not knowledge.kit[-1].any()


def test_the_champion_count_is_the_index_length_unknown_included():
    """One spare row would pin a row nothing indexes and leave the real UNKNOWN trainable."""
    from league_predictor.draft_grading.grader.set_encoder import _IdentityResidual
    from league_predictor.draft_grading.training.set_fit import build_set_knowledge

    _, index = build_set_knowledge(SMOKE)
    table = _IdentityResidual(len(index), 4, 0.5)
    assert table.n_known == index.UNKNOWN
    assert table.residual.weight.shape[0] == len(index)


def test_the_oof_pool_is_exactly_the_chronological_calibration_tail(fitted_candidate, seeded_db):
    from league_predictor.draft_grading.grader.oof_calibration import calibration_tail_start
    from league_predictor.draft_grading.training.set_fit import load_set_corpus

    corpus = load_set_corpus(seeded_db, since=SMOKE.outcome_since, until="2026-01-01")
    tail = calibration_tail_start(len(corpus.match_ids))
    assert fitted_candidate[3][0] == corpus.match_ids[tail:]
    assert fitted_candidate[2].n_pool == len(corpus.match_ids) - tail
    assert fitted_candidate[2].n_folds == SMOKE.oof_folds


def test_the_final_fit_runs_the_folds_chosen_epochs_over_every_row(fitted_candidate):
    """Early-stopping on the calibration tail would choose an epoch on the graded rows."""
    chosen = fitted_candidate[2].chosen_rounds
    assert chosen >= 1
    for report in fitted_candidate[0].history:
        assert report["epochs_run"] == chosen == report["best_epoch"]
        assert report["val_main"] == []


def test_the_candidate_returns_what_the_format_four_save_reads(fitted_candidate):
    weights, knowledge, calibration, pool = fitted_candidate
    assert len(weights.input_spec["columns"]) == weights.d_in
    assert knowledge.kit is not None
    assert 0.25 <= calibration.temperature <= 4.0
    assert len(pool[1]) == len(pool[2]) == len(pool[0])


def test_every_head_s_kendall_log_var_is_recorded_and_floored(fitted_candidate):
    log_vars = fitted_candidate[0].history[0]["log_vars"]
    assert set(log_vars) == {"main", *SMOKE.aux_heads}
    assert min(log_vars.values()) >= SMOKE.log_var_floor
    # the floor is a recipe field, so two arms that move it stamp different dicts.
    assert fitted_candidate[0].config["log_var_floor"] == SMOKE.log_var_floor


# --- Fit, save, load, serve: the synthetic end-to-end pin --------------------------------


SMOKE_RECIPE = dc.replace(DRAFT_RECIPE, set_side=SMOKE)


@pytest.fixture(scope="module")
def served(fitted_candidate, seeded_db, tmp_path_factory):
    """The fitted candidate saved as a format-4 family and reloaded by the ONE loader."""
    from league_predictor.draft_grading.live import load_grader
    from league_predictor.draft_grading.training.set_fit import save_set_candidate

    root = tmp_path_factory.mktemp("set_family") / "draft"
    save_set_candidate(root, fitted_candidate, recipe=SMOKE_RECIPE)
    return load_grader(seeded_db, artifact=root, device="cpu")


@pytest.fixture(scope="module")
def seeded_boards(seeded_db):
    """A seeded board and its mirror, plus the stats cache every batch surface takes."""
    from dataclasses import replace

    from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
    from league_predictor.draft_grading.state import DraftState

    drafts = list(DraftState.bulk_load(seeded_db, ["game_005"]).values())
    return drafts, [replace(d, blue=d.red, red=d.blue) for d in drafts], BulkStatsCache(seeded_db)


def test_the_saved_family_serves_the_recipe_it_was_fit_under(served, fitted_candidate):
    """The stamp, the calibration and the layout all come off the artifact itself."""
    from league_predictor.draft_grading.set_variant import SetDraftPredictor

    assert isinstance(served, SetDraftPredictor)
    assert served.temperature == fitted_candidate[2].temperature
    assert served.design_columns[: len(fitted_candidate[0].input_spec["columns"])] == [
        f"{column}_blue_top" for column in fitted_candidate[0].input_spec["columns"]
    ]


def test_the_identity_arm_saves_through_the_antisymmetry_gate(seeded_db, tmp_path):
    """The gate's probe draws champion IDS in the identity column, never a normal draw."""
    # A `randn` draw there indexes the residual table out of range: the arm fits, then
    # dies in `save_set_candidate` (TRAPS, the structural-probe entry).
    from datetime import date

    from league_predictor.draft_grading.training.set_fit import (
        fit_set_candidate,
        save_set_candidate,
    )

    arm = dc.replace(SMOKE, identity_residual=True)
    candidate = fit_set_candidate(seeded_db, arm, through=date(2026, 1, 1))
    recipe = dc.replace(SMOKE_RECIPE, set_side=arm)
    save_set_candidate(tmp_path / "draft", candidate, recipe=recipe)


def test_a_fitted_family_mirrors_bitwise_and_batches_to_its_own_single_read(served, seeded_boards):
    """The whole pipeline end to end: one device, one board, the two serving surfaces agree."""
    drafts, mirrored_drafts, cache = seeded_boards
    score = served.predict(drafts[0], stats_cache=cache)
    assert 0.0 < score.prob_blue_wins < 1.0
    mirror = served.predict(mirrored_drafts[0], stats_cache=cache)
    assert mirror.prob_blue_wins == 1.0 - score.prob_blue_wins
    batched = served.predict_batch(drafts + mirrored_drafts, stats_cache=cache)
    assert batched[0].prob_blue_wins == score.prob_blue_wins
    assert batched[1].prob_blue_wins == mirror.prob_blue_wins
