"""The place every draw a draft fit makes is pinned, so a refit reproduces bit-for-bit."""

from __future__ import annotations

import random

import numpy as np
import torch

#: RNG seed for encoder construction and the masked pre-train; the booster bag pins its own
#: at ``1000+i`` and the set encoder's seeds at ``set_training.SEED_BASE + i``.
PRETRAIN_SEED = 42


def pin_fit_randomness() -> None:
    """Pin every draw the fit makes, so a refit at fixed data reproduces bit-for-bit."""
    # CUDA sums embedding gradients with atomics, so the seeds alone do not pin the fit.
    # Torch raises on any op it has no deterministic kernel for, rather than drifting.
    torch.use_deterministic_algorithms(True)  # noqa: FBT003 - torch takes the flag positionally
    torch.manual_seed(PRETRAIN_SEED)
    random.seed(PRETRAIN_SEED)
    np.random.seed(PRETRAIN_SEED)  # noqa: NPY002 - pins library-internal legacy draws
