"""Pure per-axis aggregators for the player draft radar; ``player_draft_inputs`` owns the I/O."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True)
class DraftAxis:
    """One radar axis: its payload key, display label, and Overall Quality weight."""

    key: str
    label: str
    weight: float


# The one home for the radar's axes: spoke order, the labels the dashboard draws, and the
# Overall Quality blend weights (Counter Rate is informational, hence weight 0).
RADAR_AXES: tuple[DraftAxis, ...] = (
    DraftAxis("blind_quality", "Blind", 0.30),
    DraftAxis("counter_quality", "Counter Qual", 0.30),
    DraftAxis("counter_rate", "Counter Rate", 0.00),
    DraftAxis("pool_depth", "Pool Depth", 0.20),
    DraftAxis("consistency", "Consistency", 0.20),
)

AXIS_WEIGHTS: dict[str, float] = {axis.key: axis.weight for axis in RADAR_AXES}

# Below this many games, an axis is too noisy to display — return None and let
# the radar dim the spoke.
MIN_GAMES = 10

# Bayesian shrinkage strength for Consistency: shrunk_stddev =
# (n * raw + SHRINK_K * role_prior) / (n + SHRINK_K).
SHRINK_K = 20


@dataclass(frozen=True)
class DraftAxes:
    """Draft axis values (within-role percentile 0-100) plus per-axis sample counts."""

    pool_depth: float | None
    blind_quality: float | None
    counter_quality: float | None
    counter_rate: float | None
    consistency: float | None
    overall_quality: float | None
    samples: dict[str, int]


def compute_pool_depth_raw(season_champions: Iterable[tuple[str, int]]) -> float:
    """Effective pool size: ``sum(min(games / 3, 1))``, so one champion contributes at most 1."""
    return sum(min(games / 3.0, 1.0) for _champ, games in season_champions)


def compute_consistency_raw(
    per_game_pillar_scores: dict[str, list[float]],
    role_mean_stddev: float,
) -> float | None:
    """Mean per-pillar stddev shrunk toward ``role_mean_stddev``; ``None`` below MIN_GAMES."""
    # ``n`` is inferred from the pillar lists, which must all be one entry per game.
    if not per_game_pillar_scores:
        return None
    lengths = {len(v) for v in per_game_pillar_scores.values()}
    if len(lengths) != 1:
        raise ValueError(f"all pillar lists must have equal length; got {lengths}")
    n = lengths.pop()
    if n < MIN_GAMES:
        return None
    raw = mean(pstdev(scores) for scores in per_game_pillar_scores.values())
    shrink_weight = n / (n + SHRINK_K)
    return shrink_weight * raw + (1 - shrink_weight) * role_mean_stddev


def mean_or_none(pick_percentiles: list[float]) -> float | None:
    """Mean draft percentile; ``None`` below ``MIN_GAMES`` picks."""
    if len(pick_percentiles) < MIN_GAMES:
        return None
    return mean(pick_percentiles)


def compute_counter_rate_raw(n_counter: int, n_total: int) -> float | None:
    """Fraction of picks made after the enemy same-role pick; ``None`` below MIN_GAMES."""
    if n_total < MIN_GAMES:
        return None
    return n_counter / n_total


def percentile_rank_within_role(
    raw_values_by_player: Mapping[str, float | None],
    roles_by_player: Mapping[str, str],
    *,
    invert: bool = False,
) -> dict[str, float | None]:
    """Raw axis values to 0-100 within-role percentiles; ``invert=True`` when lower is better."""
    # Degenerate cases: NULL raw or unknown role passes through as NULL, a role with a
    # single player scores 50, and tied raw values share their average rank.
    out: dict[str, float | None] = {}
    # Group players by role, retaining order for determinism.
    by_role: dict[str, list[tuple[str, float]]] = {}
    for player, raw in raw_values_by_player.items():
        if raw is None:
            out[player] = None
            continue
        role = roles_by_player.get(player)
        if role is None:
            out[player] = None
            continue
        by_role.setdefault(role, []).append((player, raw))

    for entries in by_role.values():
        n = len(entries)
        if n == 1:
            out[entries[0][0]] = 50.0
            continue
        # Sorted so "best" lands at the highest rank; ties then share an average rank.
        sorted_entries = sorted(entries, key=lambda pr: pr[1], reverse=invert)
        ranks: dict[str, float] = {}
        i = 0
        while i < n:
            j = i
            while j < n and sorted_entries[j][1] == sorted_entries[i][1]:
                j += 1
            # Indices i..j-1 are tied; +1 makes the rank 1-based.
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[sorted_entries[k][0]] = avg_rank
            i = j
        for player, _ in entries:
            # Map rank (1..n) to percentile (0..100). Lowest rank → 0, highest → 100.
            r = ranks[player]
            out[player] = (r - 1) / (n - 1) * 100.0

    return out


def blend_overall_quality(axes: Mapping[str, float | None]) -> float | None:
    """Weighted blend per ``AXIS_WEIGHTS``, renormalized over non-NULL axes; None when empty."""
    weighted_sum = 0.0
    total_weight = 0.0
    for axis, weight in AXIS_WEIGHTS.items():
        value = axes.get(axis)
        if weight == 0.0 or value is None:
            continue
        weighted_sum += weight * value
        total_weight += weight
    if total_weight == 0.0:
        return None
    return weighted_sum / total_weight


def _classify_blind_or_counter(rows: list[dict]) -> tuple[list[float], list[float], int]:
    """Split pick-score rows into (blind_percentiles, counter_percentiles, n_total)."""
    blind: list[float] = []
    counter: list[float] = []
    for row in rows:
        if row["is_blind"]:
            blind.append(row["percentile"])
        else:
            counter.append(row["percentile"])
    return blind, counter, len(rows)


def compute_player_draft_axes_batch(
    *,
    roles_by_player: dict[str, str],
    season_champions_by_player: Mapping[str, list[tuple[str, int]]],
    per_game_pillars_by_player: Mapping[str, dict[str, list[float]]],
    pick_rows_by_player: Mapping[str, list[dict]],
    role_mean_stddev_by_role: Mapping[str, float],
) -> dict[str, DraftAxes]:
    """One ``DraftAxes`` per player from pre-loaded dicts.

    Returns:
        Only players with at least one pick-scores axis (blind / counter quality or rate).
        The radar reads drafted behavior; a player the draft grader never scored has no
        row rather than a shape drawn from roster stats alone.
    """
    pool_depth_raw: dict[str, float | None] = {}
    blind_quality_raw: dict[str, float | None] = {}
    counter_quality_raw: dict[str, float | None] = {}
    counter_rate_raw: dict[str, float | None] = {}
    consistency_raw: dict[str, float | None] = {}
    samples: dict[str, dict[str, int]] = {}

    for player, role in roles_by_player.items():
        rows = pick_rows_by_player.get(player, [])
        blind_pct, counter_pct, n_picks = _classify_blind_or_counter(rows)
        n_blind = len(blind_pct)
        n_counter = len(counter_pct)

        pool_depth_raw[player] = (
            compute_pool_depth_raw(season_champions_by_player.get(player, []))
            if season_champions_by_player.get(player)
            else None
        )
        blind_quality_raw[player] = mean_or_none(blind_pct)
        counter_quality_raw[player] = mean_or_none(counter_pct)
        counter_rate_raw[player] = compute_counter_rate_raw(n_counter, n_picks)

        pillars = per_game_pillars_by_player.get(player, {})
        role_prior = role_mean_stddev_by_role.get(role, 0.0)
        consistency_raw[player] = compute_consistency_raw(pillars, role_prior) if pillars else None

        samples[player] = {
            "pool_depth": sum(g for _c, g in season_champions_by_player.get(player, [])),
            "blind_quality": n_blind,
            "counter_quality": n_counter,
            "counter_rate": n_picks,
            "consistency": (next(iter({len(v) for v in pillars.values()}), 0) if pillars else 0),
        }

    # Bigger raw is better for every derived axis except Consistency, whose raw stddev
    # is inverted before ranking.
    pool_depth = percentile_rank_within_role(pool_depth_raw, roles_by_player)
    blind_quality = percentile_rank_within_role(blind_quality_raw, roles_by_player)
    counter_quality = percentile_rank_within_role(counter_quality_raw, roles_by_player)
    counter_rate_pct = percentile_rank_within_role(counter_rate_raw, roles_by_player)
    consistency = percentile_rank_within_role(consistency_raw, roles_by_player, invert=True)

    out: dict[str, DraftAxes] = {}
    for player in roles_by_player:
        pick_axes = (
            blind_quality.get(player),
            counter_quality.get(player),
            counter_rate_pct.get(player),
        )
        if all(value is None for value in pick_axes):
            continue
        axes_for_blend = {
            "blind_quality": blind_quality.get(player),
            "counter_quality": counter_quality.get(player),
            "pool_depth": pool_depth.get(player),
            "consistency": consistency.get(player),
            "counter_rate": counter_rate_pct.get(player),
        }
        out[player] = DraftAxes(
            pool_depth=pool_depth.get(player),
            blind_quality=blind_quality.get(player),
            counter_quality=counter_quality.get(player),
            counter_rate=counter_rate_pct.get(player),
            consistency=consistency.get(player),
            overall_quality=blend_overall_quality(axes_for_blend),
            samples=samples[player],
        )
    return out
