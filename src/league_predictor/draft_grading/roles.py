"""The play-count role law: roles for a set of picks by maximum summed log play counts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import permutations

import numpy as np

from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER

#: Every ordered choice of ``k`` distinct roles, ``(P, k)`` role indexes, in itertools order;
#: the argmax tie-break is the first permutation, the order suggest's role maps rely on.
_PERMUTATIONS: dict[int, np.ndarray] = {
    k: np.array(list(permutations(range(N_ROLES), k)), dtype=np.int64)
    for k in range(1, N_ROLES + 1)
}


def best_role_permutations(logs: np.ndarray) -> np.ndarray:
    """Role index per slot maximizing the summed log counts, for ``(N, k, 5)`` rows.

    Returns:
        ``(N, k)`` role indexes, distinct within a row; ``k == 0`` gives an empty ``(N, 0)``.
    """
    n, k, _ = logs.shape
    if k == 0:
        return np.empty((n, 0), dtype=np.int64)
    perms = _PERMUTATIONS[k]
    scores = logs[:, np.arange(k)[None, :], perms].sum(axis=2)
    return perms[scores.argmax(axis=1)]


def role_log_counts(
    role_counts: Mapping[tuple[str, str], int], champions: Sequence[str]
) -> np.ndarray:
    """``(len(champions), 5)`` ``log1p`` play counts in ``ROLE_ORDER``, the law's input."""
    return np.array(
        [
            [math.log1p(role_counts.get((champ, role), 0)) for role in ROLE_ORDER]
            for champ in champions
        ],
        dtype=np.float64,
    )


class PlayCountRoles:
    """The law over champion NAMES: memoized role maps for pick lists and candidate pools."""

    def __init__(self, role_counts: Mapping[tuple[str, str], int]) -> None:
        self._counts = role_counts
        # champ -> its 5 log-count terms, and tuple(picks) -> role map: both unbounded by
        # design (tiny entries; the owning engine reloads on a retrain).
        self._logs: dict[str, np.ndarray] = {}
        self._memo: dict[tuple[str, ...], dict[str, str]] = {}

    def log_counts(self, champion: str) -> np.ndarray:
        """``log1p`` play count per role for one champion, cached."""
        row = self._logs.get(champion)
        if row is None:
            row = role_log_counts(self._counts, [champion])[0]
            self._logs[champion] = row
        return row

    def assign(self, picks: Sequence[str]) -> dict[str, str]:
        """Role -> champion for at most five picks, by maximum summed log play counts."""
        if not picks:
            return {}
        key = tuple(picks)
        if key not in self._memo:
            self._memo[key] = self.assign_with(picks[:-1], [picks[-1]])[picks[-1]]
        return dict(self._memo[key])

    def assign_with(
        self, picks: Sequence[str], candidates: Sequence[str]
    ) -> dict[str, dict[str, str]]:
        """The role map of ``picks + [c]`` for every candidate ``c``, in one vectorized pass.

        Returns:
            candidate -> role map; the maps are shared with the memo, so treat them read-only.
        """
        if not candidates:
            return {}
        base = np.stack([self.log_counts(c) for c in picks]) if picks else np.empty((0, N_ROLES))
        rows = np.stack([np.concatenate([base, self.log_counts(c)[None, :]]) for c in candidates])
        best = best_role_permutations(rows)
        out: dict[str, dict[str, str]] = {}
        for cand, roles in zip(candidates, best, strict=True):
            mapping = {
                ROLE_ORDER[int(r)]: champ for champ, r in zip([*picks, cand], roles, strict=True)
            }
            self._memo.setdefault((*picks, cand), mapping)
            out[cand] = mapping
        return out
