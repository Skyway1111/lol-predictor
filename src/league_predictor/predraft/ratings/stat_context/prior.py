"""Per-player aggregates over the context z, and the team-sum weights that price them."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np

from league_predictor import config
from league_predictor.decay import half_life_weight
from league_predictor.predraft.ratings.player_scoring import PILLAR_STATS_BY_ROLE
from league_predictor.predraft.ratings.stat_context.vocab import (
    _PILLAR_INDEX,
    _STAT_INDEX,
    PILLARS,
    PRIOR_HALF_LIFE_DAYS,
    PRIOR_SHRINKAGE_K,
    ContextZ,
    PlayerPrior,
    _assert_scaled_weight_span,
)
from league_predictor.vocab.roles import ROLE_ORDER

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import date
    from pathlib import Path


def team_sum_weights_path() -> Path:
    """Resolve the fit-time weights artifact through the runtime config seam."""
    return config.DATA_DIR / "stat_context_weights.json"


# ── Aggregation ──────────────────────────────────────────────────────────────


def _weighted_aggregates(
    matrix: np.ndarray,
    players: np.ndarray,
    weights: np.ndarray,
    n_players: int,
):
    """(z-bar, per-column count) per player over NaN-masked columns."""
    present = ~np.isnan(matrix)
    weighted = np.where(present, matrix, 0.0) * weights[:, None]
    num = np.zeros((n_players, matrix.shape[1]))
    den = np.zeros((n_players, matrix.shape[1]))
    cnt = np.zeros((n_players, matrix.shape[1]))
    np.add.at(num, players, weighted)
    np.add.at(den, players, present * weights[:, None])
    np.add.at(cnt, players, present.astype(float))
    with np.errstate(invalid="ignore"):
        mean = np.where(den > 0, num / np.maximum(den, 1e-300), np.nan)
    return mean, cnt


def aggregate_prior(
    ctx: ContextZ,
    as_of: date,
    *,
    players: set[str] | None = None,
    half_life_days: float = PRIOR_HALF_LIFE_DAYS,
    shrink_k: float = PRIOR_SHRINKAGE_K,
) -> dict[str, PlayerPrior]:
    """Per-player recency-weighted pillar/stat z-bar over games strictly before ``as_of``.

    Args:
        shrink_k: sample-count shrinkage, or zero when the caller applies its own.

    Returns:
        Aggregates for every player with at least one appearance, restricted to
        ``players`` when given.
    """
    mask = ctx.appearance_dates < as_of.toordinal()
    if players is not None:
        wanted = np.array([pid in players for pid in ctx.player_ids], dtype=bool)
        mask &= wanted[ctx.appearance_players]
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {}
    weights = half_life_weight(as_of.toordinal() - ctx.appearance_dates[idx], half_life_days)
    rows = ctx.appearance_players[idx]
    n_players = len(ctx.player_ids)

    pillar_mean, pillar_cnt = _weighted_aggregates(ctx.pillar_z[idx], rows, weights, n_players)
    stat_mean, stat_cnt = _weighted_aggregates(ctx.stat_z[idx], rows, weights, n_players)
    games = np.bincount(rows, minlength=n_players)

    def factor(cnt, j, pidx):
        return cnt[pidx, j] / (cnt[pidx, j] + shrink_k)

    out: dict[str, PlayerPrior] = {}
    for pidx in np.unique(rows):
        pid = ctx.player_ids[int(pidx)]
        pillar = {
            name: float(pillar_mean[pidx, j] * factor(pillar_cnt, j, pidx))
            for name, j in _PILLAR_INDEX.items()
            if not np.isnan(pillar_mean[pidx, j])
        }
        stat = {
            name: float(stat_mean[pidx, j] * factor(stat_cnt, j, pidx))
            for name, j in _STAT_INDEX.items()
            if not np.isnan(stat_mean[pidx, j])
        }
        out[pid] = PlayerPrior(pillar_z=pillar, stat_z=stat, n_games=int(games[pidx]))
    return out


def walk_forward_pillar_aggregates(
    ctx: ContextZ,
    *,
    half_life_days: float = PRIOR_HALF_LIFE_DAYS,
    k: float = PRIOR_SHRINKAGE_K,
) -> np.ndarray:
    """Each appearance's PRE-GAME shrunk pillar z-bar (strictly earlier days only).

    Returns:
        ``[n, len(PILLARS)]`` aligned with the context's appearances; NaN before a
        player's first scored game in a pillar.
    """
    order = np.argsort(ctx.appearance_dates, kind="stable")
    out = np.full((len(order), len(PILLARS)), np.nan)
    if not order.size:
        return out
    n_players = len(ctx.player_ids)
    num = np.zeros((n_players, len(PILLARS)))
    den = np.zeros((n_players, len(PILLARS)))
    cnt = np.zeros((n_players, len(PILLARS)))

    dates = ctx.appearance_dates[order]
    t0 = int(dates[0])
    _assert_scaled_weight_span(dates, t0, half_life_days)
    boundaries = np.flatnonzero(np.diff(dates)) + 1
    for group in np.split(np.arange(len(order)), boundaries):
        rows = order[group]
        players = ctx.appearance_players[rows]
        with np.errstate(invalid="ignore"):
            mean = np.where(
                den[players] > 0, num[players] / np.maximum(den[players], 1e-300), np.nan
            )
        out[rows] = mean * cnt[players] / (cnt[players] + k)
        # The as-of scale factor cancels in num/den, so no decay pass is ever needed.
        scale = 2.0 ** ((int(dates[group[0]]) - t0) / half_life_days)
        z = ctx.pillar_z[rows]
        present = ~np.isnan(z)
        np.add.at(num, players, np.where(present, z, 0.0) * scale)
        np.add.at(den, players, present * scale)
        np.add.at(cnt, players, present.astype(float))
    return out


# ── Team-sum pillar weights ──────────────────────────────────────────────────


def _validate_role_weights(weights: Mapping, path: Path) -> dict[str, dict[str, float]]:
    """One {role: {pillar: weight}} table, finite and on each role's own pillar shape."""
    for role in ROLE_ORDER:
        pillars = set(PILLAR_STATS_BY_ROLE[role])
        if role not in weights or not set(weights[role]) <= pillars:
            raise RuntimeError(
                f"team-sum pillar weights at {path} carry bad pillars for role {role!r}; "
                "retrain with scripts/model/train_player_weights.py"
            )
        if any(not math.isfinite(v) for v in weights[role].values()):
            raise RuntimeError(
                f"team-sum pillar weights at {path} carry non-finite values for role {role!r}; "
                "retrain with scripts/model/train_player_weights.py"
            )
    return {
        role: {pillar: float(value) for pillar, value in role_weights.items()}
        for role, role_weights in weights.items()
    }


def load_team_sum_weights(
    version: int,
    artifact_path: Path | None = None,
    *,
    expected_build: dict | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Load one recipe version's team-sum pillar weights (log-odds scale), per block.

    Args:
        expected_build: when given and the block records its build kwargs, the two must
            match — a stale block silently pricing a different feature definition raises.

    Returns:
        ``{block: {role: {pillar: weight}}}``; a flat single-table artifact reads as the
        one block ``"base"``.

    Raises:
        RuntimeError: missing file, missing version block, malformed weights,
            or a build-provenance mismatch — retrain with scripts/model/train_player_weights.py.
    """
    path = artifact_path or team_sum_weights_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        table = raw["versions"][str(version)]["weights"]
        blocks = {"base": table} if set(table) & set(ROLE_ORDER) else table
        if not blocks or not isinstance(blocks, dict):
            raise ValueError("empty weights table")  # noqa: TRY301 - converted below, not escaping
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        KeyError,
        TypeError,
        AttributeError,
        ValueError,
    ) as exc:
        raise RuntimeError(
            f"team-sum pillar weights at {path} have no valid weights for version {version}; "
            "retrain with scripts/model/train_player_weights.py"
        ) from exc
    recorded_build = raw["versions"][str(version)].get("build")
    if recorded_build is not None:
        recorded_build = dict(recorded_build)
        if recorded_build.get("variant") == "covariate":
            recorded_build.pop("variant")
        if recorded_build.get("champion_offsets") is False:
            recorded_build.pop("champion_offsets")
    if (
        expected_build is not None
        and recorded_build is not None
        and recorded_build != (expected_build)
    ):
        raise RuntimeError(
            f"team-sum pillar weights at {path} were fit under build {recorded_build}, not "
            f"the recipe's {expected_build}; retrain with scripts/model/train_player_weights.py"
        )
    return {name: _validate_role_weights(table, path) for name, table in blocks.items()}


def team_sum_prior(prior: PlayerPrior, role: str, weights: Mapping[str, dict[str, float]]) -> float:
    """The player's prior in game log-odds: their role's weights over their pillar z-bar."""
    role_weights = weights[role] if role in weights else weights["mid"]
    return sum(role_weights.get(pillar, 0.0) * z for pillar, z in prior.pillar_z.items())
