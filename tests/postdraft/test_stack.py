"""The fitted stack: utilization recovery, mirror symmetry, missing teams, the holdout gate."""

from __future__ import annotations

import numpy as np
import pytest

from league_predictor.postdraft import stack as stack_module
from league_predictor.postdraft.stack import Stack, StackRow, fit_global, fit_stack

SIG = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731 - fixture-local outcome rate


def _corpus(hot_coefficient: float, *, seed: int = 7) -> list[StackRow]:
    """20 pairs converting their draft edge at beta = 1, plus HOT converting at another rate."""
    rng = np.random.default_rng(seed)
    rows: list[StackRow] = []

    def add(blue: str, red: str, l_draft: float, p_true: float, n: int) -> None:
        for _ in range(n):
            day = len(rows)
            rows.append(
                StackRow(
                    date=f"2026-{1 + day // 800:02d}-{1 + day % 28:02d}",
                    l_served=0.0,
                    l_draft=l_draft,
                    blue_team=blue,
                    red_team=red,
                    blue_won=bool(rng.random() < p_true),
                )
            )

    for i in range(20):
        edge = 0.6 if i % 2 else -0.6
        add(f"T{i}", f"T{(i + 1) % 20}", edge, float(SIG(edge)), 60)
    for i in range(20):
        add("HOT", f"T{i}", 0.5, float(SIG(hot_coefficient * 0.5)), 40)
    return rows


def test_a_team_beating_its_draft_edge_earns_a_positive_u() -> None:
    """HOT converts its draft edge at 3x the field, so its utilization must lift beta."""
    stack = fit_stack(_corpus(3.0))
    background = np.mean([abs(stack.u.get(f"T{i}", 0.0)) for i in range(20)])
    assert stack.u["HOT"] > 0.0
    assert stack.u["HOT"] > 3 * background
    assert stack.holdout_gain > 0.0
    assert stack.penalty in (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


def test_the_fit_records_its_corpus_size_per_team() -> None:
    """``n_oos`` counts the oos rows each team appears in, on either side."""
    stack = fit_stack(_corpus(3.0))
    assert stack.n_rows == 2000
    assert stack.n_oos["HOT"] == 800
    assert sum(stack.n_oos.values()) == 2 * stack.n_rows


def test_a_team_effect_that_reverses_out_of_time_is_written_as_zero() -> None:
    """FLIP converts at 4x through the train block and at -4x through the holdout block."""
    rng = np.random.default_rng(5)
    rows: list[StackRow] = []

    def add(blue: str, red: str, l_draft: float, coefficient: float, n: int) -> None:
        for _ in range(n):
            day = len(rows)
            rows.append(
                StackRow(
                    date=f"2026-{1 + (day // 340) % 12:02d}-{1 + day % 28:02d}",
                    l_served=0.0,
                    l_draft=l_draft,
                    blue_team=blue,
                    red_team=red,
                    blue_won=bool(rng.random() < float(SIG(coefficient * l_draft))),
                )
            )

    for block, flip_coefficient in ((15, 4.0), (5, -4.0)):
        for i in range(20):
            add(f"T{i}", f"T{(i + 1) % 20}", 0.6 if i % 2 else -0.6, 1.0, block)
        for i in range(10):
            add("FLIP", f"T{i}", 0.8, flip_coefficient, 4 * block)

    stack = fit_stack(rows)
    assert stack.holdout_gain < 0.0
    assert stack.u == {}
    assert all(stack.logit_final(0.0, 1.0, t, t) == stack.beta for t in stack.n_oos)


def test_the_mirrored_game_reads_exactly_one_minus_p() -> None:
    """Swapping the sides negates both logits and keeps ``u_blue + u_red``, so p mirrors."""
    stack = fit_stack(_corpus(3.0))
    p = stack.p_blue(0.4, -0.25, "HOT", "T3")
    mirrored = stack.p_blue(-0.4, 0.25, "T3", "HOT")
    assert p + mirrored == pytest.approx(1.0, abs=1e-12)


def test_a_team_the_fit_never_saw_reads_u_zero() -> None:
    """An unknown team contributes nothing; the game prices on the global coefficients."""
    stack = Stack(
        a=1.2, beta=0.8, u={"A": 0.5}, n_oos={"A": 4}, penalty=1.0, holdout_gain=0.1, n_rows=4
    )
    assert stack.logit_final(1.0, 1.0, "ghost", "other") == pytest.approx(1.2 + 0.8)
    assert stack.logit_final(1.0, 1.0, "A", "ghost") == pytest.approx(1.2 + 1.3)


def test_a_corpus_too_thin_to_split_raises_instead_of_defaulting() -> None:
    """No oos rows means no stack; the rebuild must refuse rather than serve a default."""
    with pytest.raises(RuntimeError, match="at least 2 rows"):
        fit_stack([])


def test_fit_global_recovers_planted_coefficients() -> None:
    """The two-coefficient arm the A/B scores is a plain intercept-free logistic fit."""
    rng = np.random.default_rng(11)
    l_served = rng.normal(0.0, 1.0, 20000)
    l_draft = rng.normal(0.0, 1.0, 20000)
    y = (rng.random(20000) < SIG(1.5 * l_served + 0.5 * l_draft)).astype(float)
    a, beta = fit_global(l_served, l_draft, y)
    assert a == pytest.approx(1.5, abs=0.1)
    assert beta == pytest.approx(0.5, abs=0.1)


def test_the_penalty_is_tuned_inside_the_train_block_and_gated_on_the_untouched_holdout(
    monkeypatch,
) -> None:
    """No penalty candidate ever sees the outer holdout the team terms are gated on."""
    rows = _corpus(3.0)
    n = len(rows)
    outer_train = int(n * 0.75)
    inner_train = int(outer_train * 0.75)
    fits: list[tuple[int, float, bool]] = []
    real_fit = stack_module._fit

    def recording_fit(design, *, penalty, team_aware):
        fits.append((len(design.y), penalty, team_aware))
        return real_fit(design, penalty=penalty, team_aware=team_aware)

    monkeypatch.setattr(stack_module, "_fit", recording_fit)
    stack = fit_stack(rows)
    grid = stack_module.PENALTY_GRID
    candidates = [(size, pen) for size, pen, aware in fits if aware and size == inner_train]
    assert sorted(pen for _s, pen in candidates) == sorted(grid)
    # The gate's two arms fit on the whole train block, and only the chosen penalty does.
    gate_arms = [(size, pen, aware) for size, pen, aware in fits if size == outer_train]
    assert gate_arms == [(outer_train, 0.0, False), (outer_train, stack.penalty, True)]
    # The final refit alone sees every row.
    assert [size for size, _p, _a in fits if size == n] == [n]
