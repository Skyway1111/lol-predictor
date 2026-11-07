"""The set encoder's structural pins: the mirror, the equal board, the slot row, the budget."""

from __future__ import annotations

import dataclasses as dc

import numpy as np
import pytest
import torch

from league_predictor.draft_grading.grader.set_encoder import (
    KIT_CROSS_PAIRS,
    DraftSetEncoder,
    production_wiring,
)
from league_predictor.draft_grading.grader.slot_inputs import (
    BoundSlotInputs,
    SlotInputs,
    design_blocks,
    design_columns,
)
from league_predictor.draft_grading.recipe import AUX_HEAD_NAMES, DRAFT_RECIPE, SetTrainConfig
from league_predictor.vocab.roles import MIRROR_SLOTS, N_SLOTS
from tests.draft_grading._set_helpers import POOL, StubStats, board, boards, kit_knowledge

V7 = DRAFT_RECIPE.set_recipe()
IDENTITY = dc.replace(V7, identity_residual=True)


@pytest.fixture(scope="module")
def tables():
    return kit_knowledge()


def _rows(drafts, config=V7):
    return SlotInputs(config).build(drafts, kit_knowledge(), StubStats(vary=True))


def _encoder(config=V7, seed: int = 0) -> DraftSetEncoder:
    """An encoder with its score head drawn, not zero-init: a zero head pins nothing."""
    torch.manual_seed(seed)
    n_champions = len(kit_knowledge().champions) + 1 if config.identity_residual else 0
    model = DraftSetEncoder(config, SlotInputs.d_in(config), n_champions).eval()
    torch.nn.init.normal_(model.score.weight)
    return model


# --- The slot row ---------------------------------------------------------------------


def test_every_input_column_is_named_exactly_once():
    columns = SlotInputs(V7).columns()
    assert len(set(columns)) == len(columns) == SlotInputs.d_in(V7)


def test_the_persisted_spec_names_one_column_per_input_width():
    """4a's loader reads ``columns`` and rebuilds the layout from it, never from source."""
    for config in (V7, IDENTITY, dc.replace(V7, power=False)):
        spec = SlotInputs(config).to_spec()
        assert len(spec["columns"]) == SlotInputs.d_in(config)
        assert [entry["name"] for entry in spec["blocks"]] == [
            block.name for block in SlotInputs(config).spec
        ]
        assert SlotInputs.from_spec(spec).to_spec() == spec


def test_a_spec_whose_blocks_do_not_partition_its_columns_is_refused():
    spec = SlotInputs(V7).to_spec()
    with pytest.raises(ValueError, match="span"):
        SlotInputs.from_spec({**spec, "blocks": spec["blocks"][:-1]})
    with pytest.raises(ValueError, match="names no block"):
        SlotInputs.from_spec(
            {**spec, "blocks": [{"name": "invented", "start": 0, "stop": len(spec["columns"])}]}
        )


def test_every_set_design_column_belongs_to_exactly_one_spec_block():
    """`leak_audit` splits a set family by its FROZEN spec, so no column goes unnamed."""
    spec = SlotInputs(IDENTITY).to_spec()
    d_in = len(spec["columns"])
    blocks = design_blocks(spec, d_in)
    assert set(blocks) == {"role", "kit", "power", "pro_form", "champion_id"}
    claimed = np.concatenate(list(blocks.values()))
    assert sorted(claimed) == list(range(N_SLOTS * d_in))
    assert len(design_columns(spec)) == N_SLOTS * d_in


def test_the_binder_carries_the_knowledge_and_builds_its_own_stats_cache(tables):
    """`production_wiring` hands a loader ``(encoder_factory, inputs_factory)``, no module names."""
    encoder_factory, inputs_factory = production_wiring()
    assert encoder_factory is DraftSetEncoder
    bound = inputs_factory(SlotInputs(V7).to_spec(), tables)
    assert isinstance(bound, BoundSlotInputs)
    rows = bound(boards(3), StubStats(vary=True))
    assert tuple(rows.shape) == (3, N_SLOTS, SlotInputs.d_in(V7))
    with pytest.raises(RuntimeError, match="no connection"):
        bound(boards(3))


def test_a_denied_block_is_absent_from_the_row_not_zeroed():
    full = SlotInputs(V7)
    power = full.block("power")
    denied = SlotInputs(dc.replace(V7, power=False))
    assert [block.name for block in denied.spec] == ["role", "kit", "pro_form"]
    assert denied.width == full.width - (power.stop - power.start)
    with pytest.raises(KeyError, match="power"):
        denied.block("power")


def test_the_mirrored_board_is_the_half_swap_of_the_rows(tables):
    """Every column, the pro-form pair included, over boards that differ in all ten slots."""
    drafts = boards(24)
    flipped = [dc.replace(d, blue=d.red, red=d.blue) for d in drafts]
    rows = SlotInputs(IDENTITY).build(drafts, tables, StubStats(vary=True))
    mirrored = SlotInputs(IDENTITY).build(flipped, tables, StubStats(vary=True))
    assert torch.equal(mirrored, rows[:, list(MIRROR_SLOTS)])


def test_a_swap_on_one_slot_changes_only_that_slot_row(tables):
    blue = [pool[0] for pool in POOL]
    red = [pool[1] for pool in POOL]
    swapped = [*blue[:2], POOL[2][3], *blue[3:]]
    rows = SlotInputs(V7).build([board("g", blue, red)], tables, StubStats(vary=True))
    after = SlotInputs(V7).build([board("g", swapped, red)], tables, StubStats(vary=True))
    moved = (rows != after).any(dim=-1)[0]
    assert moved.tolist() == [False, False, True, False, False] + [False] * 5


def test_the_role_one_hot_marks_the_slot_s_own_role():
    rows = _rows([board("g", [p[0] for p in POOL], [p[1] for p in POOL])])
    role = SlotInputs(V7).block("role")
    assert torch.equal(rows[0, :, role].argmax(dim=-1), torch.tensor([0, 1, 2, 3, 4] * 2))


def test_the_power_block_reaches_the_row_in_z_units(tables):
    """The engine's channels are raw gold and seconds; one projection reads them beside kit."""
    shifted = tables.power.table * 4000.0 + 9000.0
    shifted[-1] = 0.0
    raw = dc.replace(tables, power=tables.power._replace(table=shifted))
    rows = SlotInputs(V7).build(boards(24), raw, StubStats(vary=True))
    power = rows[..., SlotInputs(V7).block("power")]
    assert float(power.std()) == pytest.approx(1.0, abs=0.4)
    assert float(power.abs().max()) < 6.0


def test_a_champion_the_grid_never_ran_reads_the_power_corpus_mean(tables):
    blanked = tables.power.table.copy()
    blanked[0] = 0.0
    rows = SlotInputs(V7).build(
        [board("g", [POOL[0][0], *[p[1] for p in POOL[1:]]], [p[2] for p in POOL])],
        dc.replace(tables, power=tables.power._replace(table=blanked)),
        StubStats(vary=True),
    )
    assert torch.equal(rows[0, 0, SlotInputs(V7).block("power")], torch.zeros(44))


# --- The encoder ----------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 5, 16, 17, 64, 512])
def test_the_mirrored_draft_scores_the_exact_negation(n):
    """At every batch size: the two perspectives are one forward's halves, differenced once."""
    model = _encoder()
    x = torch.randn(n, N_SLOTS, model.d_in)
    with torch.no_grad():
        score, _ = model(x)
        mirrored, _ = model(x[:, list(MIRROR_SLOTS)])
    assert torch.equal(mirrored, -score)


def test_identical_teams_score_exactly_zero():
    model = _encoder()
    side = torch.randn(6, 5, model.d_in)
    with torch.no_grad():
        score, _ = model(torch.cat([side, side], dim=1))
    assert torch.equal(score, torch.zeros(6))


def test_a_swap_on_one_slot_moves_the_score():
    model = _encoder()
    x = torch.randn(4, N_SLOTS, model.d_in)
    moved = x.clone()
    moved[:, 7] = torch.randn(4, model.d_in)
    with torch.no_grad():
        before, _ = model(x)
        after, _ = model(moved)
    assert not torch.equal(before, after)


def test_the_auxiliary_heads_are_present_exactly_when_configured():
    x = torch.randn(3, N_SLOTS, SlotInputs.d_in(V7))
    with torch.no_grad():
        _, aux = _encoder()(x)
        _, none = _encoder(dc.replace(V7, aux_heads=()))(x)
    assert set(aux) == set(AUX_HEAD_NAMES)
    assert none == {}
    assert aux["dpm"].shape == (3, N_SLOTS)
    assert aux["class"].shape[:2] == (3, N_SLOTS)
    assert aux["gd15"].shape == (3,)


def test_the_slot_heads_mirror_with_the_board_and_the_game_heads_negate():
    """Training-only slot projections allow float32 rounding under a row permutation."""
    model = _encoder()
    x = torch.randn(5, N_SLOTS, model.d_in)
    with torch.no_grad():
        _, aux = model(x)
        _, mirrored = model(x[:, list(MIRROR_SLOTS)])
    tolerance = 8 * torch.finfo(torch.float32).eps
    for name in ("dpm", "damage_share", "class"):
        torch.testing.assert_close(
            mirrored[name], aux[name][:, list(MIRROR_SLOTS)], rtol=tolerance, atol=tolerance
        )
    for name in ("gd15", "long_game"):
        assert torch.equal(mirrored[name], -aux[name])


def test_the_encoder_carries_no_layer_that_mixes_rows():
    """Attention spans a board's ten tokens; a cross-row layer would make a score contextual."""
    forbidden = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.InstanceNorm1d)
    assert not [m for m in _encoder().modules() if isinstance(m, forbidden)]


def test_a_row_reads_the_same_in_any_company_at_a_fixed_forward_width():
    """The GEMM's blocking moves bits on a ragged batch, so `set_scores` pads; here, 512s."""
    model = _encoder()
    x = torch.randn(512, N_SLOTS, model.d_in)
    with torch.no_grad():
        full, _ = model(x)
        half, _ = model(x[:256])
        shuffled, _ = model(x[torch.arange(511, -1, -1)])
    assert torch.equal(half, full[:256])
    assert torch.equal(shuffled, full.flip(0))


def test_the_encoder_refuses_a_width_that_is_not_its_config_s_own():
    with pytest.raises(ValueError, match="slot row is"):
        DraftSetEncoder(V7, SlotInputs.d_in(V7) + 1)


def test_the_identity_arm_refuses_to_build_without_a_champion_universe():
    with pytest.raises(ValueError, match="n_champions"):
        DraftSetEncoder(dc.replace(V7, identity_residual=True), SlotInputs.d_in(V7) + 1)


def test_the_identity_residual_starts_at_zero_and_is_penalised():
    arm = dc.replace(V7, identity_residual=True)
    model = _encoder(arm)
    base = _encoder(V7)
    x = torch.randn(4, N_SLOTS, SlotInputs.d_in(arm))
    x[..., -1] = torch.randint(0, 20, (4, N_SLOTS)).float()
    with torch.no_grad():
        assert torch.equal(model(x)[0], model(x)[0])
    assert float(model.identity_penalty().detach()) == 0.0
    assert float(base.identity_penalty().detach()) == 0.0
    assert model.identity is not None


def test_a_config_that_denies_every_input_block_is_refused():
    with pytest.raises(ValueError, match="every input block is denied"):
        SetTrainConfig(kit=False, power=False, pro_form=False, cross_pairs=False)


# --- The named cross-team pairs -------------------------------------------------------


def test_every_named_pair_reads_two_kit_families_the_vector_carries():
    from league_predictor.draft_grading.champion_attributes.kit_vector import KIT_FAMILY_NAMES

    named = {name for pair in KIT_CROSS_PAIRS for name in pair}
    assert named <= set(KIT_FAMILY_NAMES)
    assert len(set(KIT_CROSS_PAIRS)) == len(KIT_CROSS_PAIRS)


def test_a_pair_term_is_the_team_sums_offence_against_the_enemy_s_defence():
    """``blue_off * red_def - red_off * blue_def``, so the term negates under the mirror."""
    from league_predictor.draft_grading.champion_attributes.kit_vector import KIT_FAMILY_NAMES

    model = _encoder()
    kit = SlotInputs(V7).block("kit").start
    x = torch.zeros(1, N_SLOTS, model.d_in)
    offence, defence = KIT_CROSS_PAIRS[0]
    x[0, 0, kit + KIT_FAMILY_NAMES.index(offence)] = 3.0
    x[0, 5, kit + KIT_FAMILY_NAMES.index(defence)] = 5.0
    with torch.no_grad():
        terms = model._cross_terms(x)
    assert float(terms[0, 0]) == pytest.approx(15.0)
    with torch.no_grad():
        assert torch.equal(model._cross_terms(x[:, list(MIRROR_SLOTS)]), -terms)


def test_denying_the_pairs_narrows_the_difference_the_heads_read():
    plain = _encoder(dc.replace(V7, cross_pairs=False))
    assert plain.score.weight.shape[1] == V7.width
    assert _encoder().score.weight.shape[1] == V7.width + len(KIT_CROSS_PAIRS)


def test_a_trained_identity_table_still_mirrors_exactly():
    """The zero-init table pins nothing, so fill it before asking whether the mirror holds."""
    model = _encoder(IDENTITY)
    assert model.identity is not None
    torch.nn.init.normal_(model.identity.residual.weight)
    x = torch.randn(16, N_SLOTS, model.d_in)
    x[..., -1] = torch.randint(0, len(kit_knowledge().champions), (16, N_SLOTS)).float()
    with torch.no_grad():
        score, _ = model(x)
        mirrored, _ = model(x[:, list(MIRROR_SLOTS)])
    assert torch.equal(mirrored, -score)
    assert float(model.identity_penalty().detach()) > 0.0
