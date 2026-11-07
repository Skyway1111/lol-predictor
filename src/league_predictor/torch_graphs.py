"""Shared CUDA training capture with owned inputs."""

import warnings
from contextlib import contextmanager
from typing import Any

import torch
from torch.utils._pytree import tree_leaves, tree_map


@contextmanager
def graph_training():
    """Scope PyTorch's warning for the graph API's intentional gradient stream handoff."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The AccumulateGrad node's stream", category=UserWarning
        )
        yield


def _warmup(module: torch.nn.Module, samples: tuple) -> None:
    inputs = (*tree_leaves(samples), *module.parameters())
    trainable = tuple(t for t in inputs if t.requires_grad)
    for _ in range(3):
        outputs = tuple(t for t in tree_leaves(module(*samples)) if t.requires_grad)
        torch.autograd.grad(
            outputs, trainable, tuple(torch.zeros_like(t) for t in outputs), allow_unused=True
        )


def capture_training(module: torch.nn.Module, samples: tuple, *, pool: Any = None) -> None:
    """Capture forwards and gradients; replay must never overwrite a caller's corpus."""
    owned = tree_map(torch.clone, samples)
    # Reuse the caller's stream so each fit does not retain another cuBLAS workspace.
    _warmup(module, owned)
    torch.cuda.make_graphed_callables(module, owned, num_warmup_iters=0, pool=pool)
