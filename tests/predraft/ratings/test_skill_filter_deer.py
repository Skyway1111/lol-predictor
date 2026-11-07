"""Parity gates for the quasi-DEER fit engine against the sequential replay engine, on CPU."""

from __future__ import annotations

import dataclasses
from dataclasses import fields
from datetime import date

import pytest
import torch
import torch.nn.functional as F  # noqa: N812 - the torch.nn.functional convention
from torch.func import stack_module_state

from league_predictor.predraft.ratings.skill_filter import deer
from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
from league_predictor.predraft.ratings.skill_filter.model import (
    BatchedSkillFilter,
    FilterConfig,
    SkillFilter,
)
from league_predictor.predraft.ratings.skill_filter.recipe import RATING_RECIPE
from league_predictor.predraft.ratings.skill_filter.replay import (
    replay_for_serving,
)
from tests.predraft.ratings._replay_reference import replay_for_training
from tests.predraft.ratings.conftest import _conn, _mixed_conn, _varied_conn, _weekly_conn

_LOSS_END = date(2026, 3, 15)
_VAL_WINDOW = (date(2026, 3, 15), date(2026, 8, 1))


def _corpus64(conn):
    """Fixture corpus in f32 (the trainer dtype) and its f64 cast (the parity dtype)."""
    c32 = build_corpus(conn, through=date(2027, 1, 1), norm_through=_LOSS_END)
    return c32, _cast_corpus(c32, torch.float64)


def _cast_corpus(corpus, dtype):
    """Corpus with every floating tensor cast to dtype; int ids untouched."""

    def cast(obj):
        moved = {f.name: getattr(obj, f.name) for f in fields(obj)}
        for k, v in moved.items():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                moved[k] = v.to(dtype)
        return dataclasses.replace(obj, **moved)

    return dataclasses.replace(
        corpus, days=[cast(db) for db in corpus.days], norm=cast(corpus.norm)
    )


def _prod_model(corpus, *, seed: int = 0) -> SkillFilter:
    cfg = FilterConfig(**RATING_RECIPE.architecture(), seed=seed)
    m = SkillFilter(cfg, n_leagues=max(1, corpus.vocab.n_leagues)).double()
    m.eval()
    return m


_TRAINED_SD: dict[str, dict] = {}


def _trained_state_dict(name: str, corpus32) -> dict:
    """Deterministic production-shaped weights cached per fixture."""
    if name not in _TRAINED_SD:
        _TRAINED_SD[name] = _prod_model(corpus32).state_dict()
    return _TRAINED_SD[name]


def _bag_of(models: list[SkillFilter]):
    """Stacked f64 leaves + BatchedSkillFilter from solo members (the trainer's construction)."""
    params, _ = stack_module_state(models)
    leaves = {n: p.detach().clone().requires_grad_() for n, p in params.items()}
    return BatchedSkillFilter(leaves, models[0].config), leaves


def _solve(models, corpus64, *, with_margin: bool, **solver_kw):
    """One deer forward solve of ``models`` as a bag over ``corpus64``."""
    bag, leaves = _bag_of(models)
    flat = deer.flatten_corpus(corpus64, models[0])
    plans = deer.build_bag_plans(flat, len(models))
    sol = deer.solve_forward_bag(bag, flat, plans, with_margin=with_margin, **solver_kw)
    return bag, leaves, flat, plans, sol


class TestNonConvergenceSafety:
    """An exhausted residual budget is a hard failure, never a usable trajectory/gradient."""

    def test_forward_refuses_an_exhausted_iteration_budget(self):
        _, corpus = _corpus64(_conn())
        model = _prod_model(corpus)
        bag, _ = _bag_of([model])
        flat = deer.flatten_corpus(corpus, model)
        plans = deer.build_bag_plans(flat, 1)

        with pytest.raises(RuntimeError, match=r"forward.*0 iterations"):
            deer.solve_forward_bag(bag, flat, plans, with_margin=False, max_iters=0)

    def test_adjoint_refuses_an_exhausted_iteration_budget(self):
        _, corpus = _corpus64(_conn())
        model = _prod_model(corpus)
        bag, leaves, flat, plans, sol = _solve([model], corpus, with_margin=False)

        def loss(logits, _margin):
            return logits.square().mean()

        with pytest.raises(RuntimeError, match=r"adjoint.*0 iterations"):
            deer.backward_implicit_bag(
                bag,
                flat,
                sol,
                plans,
                loss,
                leaves,
                with_margin=False,
                max_iters=0,
            )


# ── CPU structural pins: the solver math itself ────────────────────────────── #


class TestChainScans:
    """The associative chain scans against brute-force sequential recurrences."""

    def _chains(self, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        n_ent, max_rank, c = 7, 13, 3
        lengths = torch.randint(1, max_rank + 1, (n_ent,), generator=g)
        ent, rank = [], []
        for e, ln in enumerate(lengths.tolist()):
            ent += [e] * ln
            rank += list(range(ln))
        ent_t, rank_t = torch.tensor(ent), torch.tensor(rank)
        m = ent_t.numel()
        D = torch.randn(m, c, generator=g, dtype=torch.float64)
        # debut rows (rank 0) carry nothing in — the solver's convention
        D[rank_t == 0] = 0.0
        R = torch.randn(m, c, generator=g, dtype=torch.float64)
        return D, R, ent_t, rank_t, n_ent, max_rank

    def test_chain_scan_solves_the_forward_recurrence(self):
        D, R, ent, rank, n_ent, max_rank = self._chains()
        out = deer._chain_scan(D, R, ent=ent, rank=rank, n_ent=n_ent, max_rank=max_rank)
        # reference: delta[m] = D[m] * delta[prev(m)] + R[m] along each entity chain
        ref = torch.zeros_like(R)
        prev: dict[int, int] = {}
        for m in range(ent.numel()):
            e = int(ent[m])
            carry = ref[prev[e]] if e in prev else torch.zeros(D.shape[-1], dtype=D.dtype)
            ref[m] = D[m] * carry + R[m]
            prev[e] = m
        assert float((out - ref).abs().max()) < 1e-12

    def test_chain_scan_rev_solves_the_transposed_system(self):
        D, R, ent, rank, n_ent, max_rank = self._chains(seed=1)
        out = deer._chain_scan_rev(D, R, ent=ent, rank=rank, n_ent=n_ent, max_rank=max_rank)
        # reference: y[m] = D[child(m)] * y[child(m)] + R[m], tails carry nothing
        ref = torch.zeros_like(R)
        child: dict[int, int] = {}
        for m in range(ent.numel() - 1, -1, -1):
            e = int(ent[m])
            carry = D[child[e]] * ref[child[e]] if e in child else torch.zeros_like(ref[m])
            ref[m] = carry + R[m]
            child[e] = m
        assert float((out - ref).abs().max()) < 1e-12

    def test_scan_k_folds_the_member_axis_losslessly(self):
        D, R, ent, rank, n_ent, max_rank = self._chains(seed=2)
        Dk = torch.stack([D, 0.5 * D, torch.zeros_like(D)])
        Rk = torch.stack([R, R + 1.0, R])
        out = deer._scan_k(
            deer._chain_scan, Dk, Rk, ent=ent, rank=rank, n_ent=n_ent, max_rank=max_rank
        )
        for k in range(3):
            solo = deer._chain_scan(
                Dk[k], Rk[k], ent=ent, rank=rank, n_ent=n_ent, max_rank=max_rank
            )
            assert float((out[k] - solo).abs().max()) < 1e-12

    def test_constant_row_contributes_exactly_its_own_residual(self):
        # The Jordan-chain landmine: a debut row (D forced to 0) must pass R through
        # untouched no matter what its neighbors carry.
        D, R, ent, rank, n_ent, max_rank = self._chains(seed=3)
        out = deer._chain_scan(D, R, ent=ent, rank=rank, n_ent=n_ent, max_rank=max_rank)
        debut = rank == 0
        assert float((out[debut] - R[debut]).abs().max()) == 0.0


class TestAndersonMixer:
    """Anderson(m) on a linear fixed point: same answer, decisively fewer iterations."""

    def _problem(self, n: int = 40, rho: float = 0.95):
        # Symmetric contraction with a clustered spectrum ({rho, rho/2, rho/10}) at spectral
        # radius rho -- the analog of the engine's chain-preconditioned operator.
        g = torch.Generator().manual_seed(7)
        q, _ = torch.linalg.qr(torch.randn(n, n, generator=g, dtype=torch.float64))
        eigs = torch.tensor([rho, rho / 2, rho / 10], dtype=torch.float64).repeat(n)[:n]
        A = q @ torch.diag(eigs) @ q.T
        b = torch.randn(n, generator=g, dtype=torch.float64)
        x_star = torch.linalg.solve(torch.eye(n, dtype=torch.float64) - A, b)
        return A, b, x_star

    def _iterate(self, mixer, A, b, tol=1e-10, max_iters=400):
        x = torch.zeros(A.shape[0], dtype=torch.float64)
        for it in range(max_iters):
            g = (A @ x + b) - x
            if float(g.abs().max()) <= tol:
                return x, it
            x = mixer.mix(x, g) if mixer is not None else x + g
        return x, max_iters

    def test_reaches_the_true_fixed_point(self):
        A, b, x_star = self._problem()
        x, _ = self._iterate(deer._AndersonMixer(5), A, b)
        assert float((x - x_star).abs().max()) < 1e-8

    def test_accelerates_the_plain_iteration(self):
        A, b, _ = self._problem()
        _, plain_iters = self._iterate(None, A, b)
        _, mixed_iters = self._iterate(deer._AndersonMixer(5), A, b)
        assert mixed_iters < plain_iters / 3, (mixed_iters, plain_iters)

    def test_restart_drops_the_memory_but_not_correctness(self):
        A, b, x_star = self._problem()
        mixer = deer._AndersonMixer(5)
        x = torch.zeros(A.shape[0], dtype=torch.float64)
        for it in range(200):
            g = (A @ x + b) - x
            if float(g.abs().max()) <= 1e-10:
                break
            if it % 7 == 0:
                mixer.restart()
            x = mixer.mix(x, g)
        assert float((x - x_star).abs().max()) < 1e-8


# ── forward parity vs the sequential engine (f64, fixtures) ────────────────── #


def _state_parity(model, corpus64, flat, sol):
    """Max abs diff to the sequential snapshots over every (entity, day), bag member 0."""
    tl = replay_for_serving(corpus64, model, norm=corpus64.norm)
    max_p = 0.0
    for m in range(flat.n_app):
        pid, rank = int(flat.app_ent[m]), int(flat.app_rank[m])
        max_p = max(max_p, float((sol.W[0, m] - tl.player_states[pid][rank]).abs().max()))
        assert tl.player_dates[pid][rank] == flat.day_dates[int(flat.app_day[m])]
    max_o = 0.0
    for j in range(flat.n_oapp):
        oid, rank = int(flat.oapp_ent[j]), int(flat.oapp_rank[j])
        max_o = max(max_o, float((sol.U[0, j] - tl.org_states[oid][rank]).abs().max()))
    return max_p, max_o


class TestForwardParity:
    """Solved trajectory matches the sequential replay to <=1e-6 in f64 (measured ~1e-10)."""

    @pytest.mark.parametrize(
        "make_conn", [_weekly_conn, _mixed_conn, _conn, _varied_conn], ids=lambda f: f.__name__
    )
    def test_init_weights(self, make_conn):
        _c32, c64 = _corpus64(make_conn())
        model = _prod_model(c64)
        seq = replay_for_training(c64, model)
        _, _, flat, _, sol = _solve([model], c64, with_margin=False)
        assert float((sol.logits[0] - seq.logits.detach()).abs().max()) < 1e-6
        max_p, max_o = _state_parity(model, c64, flat, sol)
        assert max_p < 1e-6, (max_p, max_o)
        assert max_o < 1e-6, (max_p, max_o)

    @pytest.mark.parametrize("make_conn", [_varied_conn, _conn], ids=lambda f: f.__name__)
    def test_trained_weights(self, make_conn):
        c32, c64 = _corpus64(make_conn())
        model = _prod_model(c64)
        sd = _trained_state_dict(make_conn.__name__, c32)
        model.load_state_dict({k: v.double() for k, v in sd.items()})
        seq = replay_for_training(c64, model)
        _, _, flat, _, sol = _solve([model], c64, with_margin=False)
        assert float((sol.logits[0] - seq.logits.detach()).abs().max()) < 1e-6
        max_p, max_o = _state_parity(model, c64, flat, sol)
        assert max_p < 1e-6, (max_p, max_o)
        assert max_o < 1e-6, (max_p, max_o)


# ── gradient parity at the FULL fit_bag loss (f64, varied fixture) ─────────── #


def _member_loss(logits, labels, m_pred, m_target, m_mask, train_mask):
    """The deployed per-member loss: W/L plus margin at weight 0.3."""
    wl = F.binary_cross_entropy_with_logits(logits[train_mask], labels[train_mask])
    masked = m_mask[train_mask]
    margin = (((m_pred[train_mask] - m_target[train_mask]) ** 2) * masked).sum()
    margin = margin / masked.sum().clamp(min=1.0)
    return wl + 0.3 * margin


class TestGradientParityFullLoss:
    """IFT gradients match sequential BPTT autograd at the full trainer loss, K=3, f64."""

    _K = 3
    _ABS_TOL = 1e-10
    _REL_TOL = 1e-8

    def _members(self, c32, c64, *, trained: bool):
        models = [_prod_model(c64, seed=s) for s in range(self._K)]
        if trained:
            sd = _trained_state_dict("_varied_conn", c32)
            g = torch.Generator().manual_seed(99)
            for k, m in enumerate(models):
                m.load_state_dict({n: v.double() for n, v in sd.items()})
                if k > 0:  # member-diverse realistic weights
                    with torch.no_grad():
                        for p in m.parameters():
                            p.add_(
                                0.05
                                * torch.randn(p.shape, generator=g).double()
                                * (p.abs().mean() + 1e-3)
                            )
        return models

    def _gate(self, *, trained: bool, **solver_kw):
        c32, c64 = _corpus64(_varied_conn())
        models = self._members(c32, c64, trained=trained)
        dates = [d for db in c64.days for d in [db.date] * db.a_won.shape[0]]
        train_mask = torch.tensor([d < _LOSS_END for d in dates])

        # sequential per-member BPTT reference
        ref_grads, ref_logits = [], []
        for m in models:
            seq = replay_for_training(c64, m, with_margin=True)
            loss = _member_loss(
                seq.logits,
                seq.labels,
                seq.margin_pred,
                seq.margin_target,
                seq.margin_mask,
                train_mask,
            )
            names, params = zip(*[(n, p) for n, p in m.named_parameters()], strict=True)
            gs = torch.autograd.grad(loss, params, allow_unused=True)
            ref_grads.append(
                {
                    n: (g if g is not None else torch.zeros_like(p))
                    for n, p, g in zip(names, params, gs, strict=True)
                }
            )
            ref_logits.append(seq.logits.detach())

        # deer solve + implicit backward at the same loss
        bag, leaves, flat, plans, sol = _solve(models, c64, with_margin=True, **solver_kw)
        labels = flat.a_won

        def bag_loss(logits, m_pred):
            total = logits.new_zeros(())
            for k in range(self._K):
                total = total + _member_loss(
                    logits[k],
                    labels,
                    m_pred[k],
                    flat.margin_target,
                    flat.margin_mask,
                    train_mask,
                )
            return total

        bwd_kw = {k: v for k, v in solver_kw.items() if k != "jvp_every"}  # forward-only knob
        grads, _ = deer.backward_implicit_bag(
            bag, flat, sol, plans, bag_loss, leaves, with_margin=True, **bwd_kw
        )

        logit_diff = max(float((sol.logits[k] - ref_logits[k]).abs().max()) for k in range(self._K))
        assert logit_diff < 1e-8, logit_diff
        for n, g in grads.items():
            for k in range(self._K):
                diff = float((g[k] - ref_grads[k][n]).abs().max())
                ref_mag = float(ref_grads[k][n].abs().max())
                assert diff < self._ABS_TOL, f"{n}[k={k}]: abs {diff}"
                # Relative error is meaningless for near-zero gradients; the absolute gate
                # above is the binding contract there.
                if ref_mag > 1e-6:
                    assert diff / ref_mag < self._REL_TOL, f"{n}[k={k}]: rel {diff / ref_mag}"

    def test_init_weights(self):
        self._gate(trained=False)

    def test_trained_scale_weights(self):
        self._gate(trained=True)

    def test_solver_config_does_not_move_the_answer(self):
        # Plain splitting (no Anderson, jvp diagonal every iteration) passes the same pins.
        self._gate(trained=True, anderson_m=0, jvp_every=1)


class TestBagKEquivalence:
    """The K axis is a pure fold: a K=3 bag solve ≡ three K=1 solves, f64."""

    def test_bag_matches_per_member_solves(self):
        c32, c64 = _corpus64(_varied_conn())
        sd = _trained_state_dict("_varied_conn", c32)
        g = torch.Generator().manual_seed(11)
        models = []
        for k in range(3):
            m = _prod_model(c64, seed=k)
            m.load_state_dict({n: v.double() for n, v in sd.items()})
            with torch.no_grad():
                for p in m.parameters():
                    p.add_(0.03 * torch.randn(p.shape, generator=g).double())
            models.append(m)
        _, _, _, _, sol = _solve(models, c64, with_margin=True)
        for k, m in enumerate(models):
            _, _, _, _, solo = _solve([m], c64, with_margin=True)
            assert float((sol.logits[k] - solo.logits[0]).abs().max()) < 1e-9
            assert float((sol.W[k] - solo.W[0]).abs().max()) < 1e-9
