"""Fit the deployed five-member Rating recipe on CUDA with DEER."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING

from league_predictor.predraft.ratings.skill_filter.model import (
    BatchedSkillFilter,
    FilterConfig,
    SkillFilter,
)
from league_predictor.predraft.ratings.skill_filter.recipe import RATING_RECIPE

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from datetime import date

    from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus


_DEER_VRAM_BASE_GB = 1.0
_DEER_VRAM_GB_PER_GAME_MEMBER = 1.7e-5


def _deer_vram_estimate_gb(n_games: int, k_members: int) -> float:
    """Estimate peak DEER-CUDA memory from the measured linear scaling law."""
    return _DEER_VRAM_BASE_GB + _DEER_VRAM_GB_PER_GAME_MEMBER * n_games * k_members


def _available_vram_gb() -> float:
    """Return free CUDA memory after releasing this process's reusable cache."""
    import torch

    torch.cuda.empty_cache()
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def _gate_deer_vram(n_games: int, k_members: int) -> None:
    """Refuse a fit whose measured memory class exceeds free GPU memory."""
    needed = _deer_vram_estimate_gb(n_games, k_members)
    available = _available_vram_gb()
    if available < needed:
        raise RuntimeError(
            f"Rating training needs an estimated ~{needed:.1f} GB of GPU memory "
            f"({n_games} games x {k_members} members), but only {available:.1f} GB is free. "
            "Free GPU memory or use a CUDA GPU with more VRAM; no slower fallback is supported."
        )


def _deer_available() -> bool:
    """Return whether the required CUDA runtime is available."""
    import torch

    return torch.cuda.is_available()


@dataclass(frozen=True, slots=True)
class FitResult:
    """One selected bag member and its convergence evidence."""

    model: SkillFilter
    val_logloss: float
    epochs: int


def fit_production_bag(
    corpus: Corpus,
    *,
    loss_end: date,
    val_window: tuple[date, date],
) -> list[FitResult]:
    """Fit the sole deployed K=5 recipe; callers control only the data cutoff."""
    if not _deer_available():
        raise RuntimeError(
            "Rating training requires a CUDA-capable GPU and the DEER engine; "
            "the slower CPU/compiled/checkpointed trainers are not supported."
        )
    if RATING_RECIPE.seeds != 5:  # noqa: PLR2004 - the deployed five-member ensemble
        raise RuntimeError("the deployed Rating recipe must contain exactly five members")

    n_games = sum(len(day.a_won) for day in corpus.days)
    _gate_deer_vram(n_games, RATING_RECIPE.seeds)
    return _fit_deployed_bag(corpus, loss_end=loss_end, val_window=val_window)


def _clip_grad_norm_per_member(
    params: Mapping, members: int, device: str, max_norm: float = 1.0
) -> None:
    """Clip each bag member independently without coupling its update to another member."""
    import torch

    squared = torch.zeros(members, device=device)
    for param in params.values():
        if param.grad is not None:
            squared = squared + param.grad.detach().reshape(members, -1).pow(2).sum(dim=1)
    coefficient = (max_norm / (squared.sqrt() + 1e-6)).clamp(max=1.0)
    for param in params.values():
        if param.grad is not None:
            param.grad.mul_(coefficient.view([members] + [1] * (param.grad.dim() - 1)))


def _deer_backward_live(
    live: Iterable[int], done: Sequence[bool], *, epoch: int, max_epochs: int
) -> list[int]:
    """Return members whose update can affect a later evaluated epoch."""
    if epoch >= max_epochs:
        return []
    return [member for member in live if not done[member]]


def _epoch_masks(flat, loss_end: date, val_window: tuple[date, date], device: str):
    """Per-game train and validation masks over the flattened replay.

    Raises:
        ValueError: the window selects no game to train on, or none to early-stop on.
    """
    import torch

    dates = [day for i, day in enumerate(flat.day_dates) for _ in range(flat.day_sizes[i])]
    train_mask = torch.tensor([day < loss_end for day in dates], device=device)
    val_mask = torch.tensor([val_window[0] <= day < val_window[1] for day in dates], device=device)
    if not bool(train_mask.any()):
        raise ValueError(f"no games before loss_end={loss_end} to train on")
    if not bool(val_mask.any()):
        raise ValueError(f"no games in val_window={val_window} to early-stop on")
    return train_mask, val_mask


def _member_loss(logits, margin_prediction, *, flat, train_mask, labels):
    """Return the fixed W/L plus margin objective and its bare W/L component."""
    import torch.nn.functional as F  # noqa: N812 - the torch.nn.functional convention

    win_loss = F.binary_cross_entropy_with_logits(logits[train_mask], labels[train_mask])
    mask = flat.margin_mask[train_mask]
    margin_loss = (
        ((margin_prediction[train_mask] - flat.margin_target[train_mask]) ** 2) * mask
    ).sum() / mask.sum().clamp(min=1.0)
    return win_loss + RATING_RECIPE.margin_weight * margin_loss, win_loss


def _bag_loss(logits, margin_predictions, *, member_loss):
    """Sum independent member losses along the fixed bag axis."""
    assert margin_predictions is not None
    total = logits.new_zeros(())
    for position in range(logits.shape[0]):
        loss, _win_loss = member_loss(logits[position], margin_predictions[position])
        total = total + loss
    return total


def _bag_for(positions, *, leaves, config, flat, plans_by_count, device):
    """The bag over ``positions`` of the stacked leaves, with the scatter plans for its width.

    Returns:
        ``(bag, plans)``; plans are memoized in ``plans_by_count`` since they depend only on
        the member count.
    """
    import torch

    # sightline-ok: 35 - torch-loading module stays off the cold import path
    from league_predictor.predraft.ratings.skill_filter.deer import build_bag_plans

    index = torch.tensor(positions, dtype=torch.int64, device=device)
    sub = BatchedSkillFilter({name: parameter[index] for name, parameter in leaves.items()}, config)
    if len(positions) not in plans_by_count:
        plans_by_count[len(positions)] = build_bag_plans(flat, len(positions))
    return sub, plans_by_count[len(positions)]


def _fit_deployed_bag(  # noqa: PLR0915 - linear AdamW schedule over per-member bookkeeping
    corpus: Corpus,
    *,
    loss_end: date,
    val_window: tuple[date, date],
) -> list[FitResult]:
    """Run the fixed AdamW K=5 schedule through DEER's exact implicit gradient."""
    import torch
    import torch.nn.functional as F  # noqa: N812 - the torch.nn.functional convention
    from torch.func import stack_module_state

    # sightline-ok: 35 - torch-loading module stays off the cold import path
    from league_predictor.predraft.ratings.skill_filter.deer import (
        backward_implicit_bag,
        build_bag_plans,
        flatten_corpus,
        solve_forward_bag,
    )

    device = "cuda"
    config = FilterConfig(**RATING_RECIPE.architecture())
    seeds = list(range(RATING_RECIPE.seeds))
    members = len(seeds)
    n_leagues = max(1, corpus.vocab.n_leagues)
    models = [
        SkillFilter(dataclasses.replace(config, seed=seed), n_leagues=n_leagues) for seed in seeds
    ]
    stacked, _buffers = stack_module_state(models)
    leaves = {
        name: parameter.detach().to(device).requires_grad_(True)  # noqa: FBT003 - requires_grad_
        for name, parameter in stacked.items()
    }
    bag = BatchedSkillFilter(leaves, models[0].config)

    flat = flatten_corpus(corpus, models[0], norm=corpus.norm).to(device)
    plans = build_bag_plans(flat, members)
    plans_by_count = {members: plans}
    train_mask, val_mask = _epoch_masks(flat, loss_end, val_window, device)

    labels = flat.a_won
    member_loss = partial(_member_loss, flat=flat, train_mask=train_mask, labels=labels)
    bag_loss = partial(_bag_loss, member_loss=member_loss)
    sub_bag = partial(
        _bag_for,
        leaves=leaves,
        config=bag.config,
        flat=flat,
        plans_by_count=plans_by_count,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        list(leaves.values()),
        lr=RATING_RECIPE.lr,
        weight_decay=RATING_RECIPE.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=RATING_RECIPE.max_epochs
    )

    best_val = [float("inf")] * members

    best_state: list[dict | None] = [None] * members
    best_epoch = [0] * members
    since_best = [0] * members
    done = [False] * members

    previous_w = previous_u = None
    previous_adjoint = None
    live = list(range(members))
    epoch = 0
    for epoch in range(1, RATING_RECIPE.max_epochs + 1):
        if len(live) < members:
            epoch_bag, epoch_plans = sub_bag(live)
        else:
            epoch_bag, epoch_plans = bag, plans

        solution = solve_forward_bag(
            epoch_bag,
            flat,
            epoch_plans,
            with_margin=True,
            W0=previous_w,
            U0=previous_u,
        )
        assert solution.m_pred is not None
        with torch.no_grad():
            for position, member in enumerate(live):
                validation = float(
                    F.binary_cross_entropy_with_logits(
                        solution.logits[position][val_mask], labels[val_mask]
                    )
                )
                if validation < best_val[member]:
                    _loss, _win_loss = member_loss(
                        solution.logits[position], solution.m_pred[position]
                    )
                    best_val[member] = validation

                    best_state[member] = {
                        name: parameter[member].detach().cpu().clone()
                        for name, parameter in leaves.items()
                    }
                    best_epoch[member] = epoch
                    since_best[member] = 0
                else:
                    since_best[member] += 1
                    if since_best[member] >= RATING_RECIPE.patience:
                        done[member] = True

        backward_live = _deer_backward_live(
            live,
            done,
            epoch=epoch,
            max_epochs=RATING_RECIPE.max_epochs,
        )
        if not backward_live:
            break
        if backward_live != live:
            keep = torch.tensor([live.index(member) for member in backward_live], device=device)
            backward_bag, backward_plans = sub_bag(backward_live)
            backward_solution = dataclasses.replace(
                solution,
                W=solution.W[keep],
                U=solution.U[keep],
                logits=solution.logits[keep],
                m_pred=solution.m_pred[keep],
                dd_p=solution.dd_p[keep],
                dd_o=solution.dd_o[keep],
            )
            backward_adjoint = (
                (previous_adjoint[0][keep], previous_adjoint[1][keep])
                if previous_adjoint is not None
                else None
            )
        else:
            backward_bag = epoch_bag
            backward_plans = epoch_plans
            backward_solution = solution
            backward_adjoint = previous_adjoint

        gradients, previous_adjoint = backward_implicit_bag(
            backward_bag,
            flat,
            backward_solution,
            backward_plans,
            bag_loss,
            leaves,
            with_margin=True,
            lam0=backward_adjoint,
        )
        optimizer.zero_grad(set_to_none=True)
        for name, gradient in gradients.items():
            leaves[name].grad = gradient
        _clip_grad_norm_per_member(leaves, members, device)
        optimizer.step()
        scheduler.step()
        previous_w, previous_u = backward_solution.W.detach(), backward_solution.U.detach()
        live = backward_live

    results = []
    for member in range(members):
        selected = best_state[member] or {
            name: parameter[member].detach().cpu().clone() for name, parameter in leaves.items()
        }
        models[member].load_state_dict(selected)
        models[member].eval()
        results.append(
            FitResult(
                model=models[member],
                val_logloss=best_val[member],
                epochs=best_epoch[member],
            )
        )
    return results
