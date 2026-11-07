"""DB-side input assembly for the player draft radar, and axis attachment onto snapshot rows."""

from __future__ import annotations

import dataclasses
from datetime import date, timedelta
from typing import TYPE_CHECKING

from league_predictor.analytics.player_draft import (
    DraftAxes,
    compute_player_draft_axes_batch,
)
from league_predictor.predraft.league_tier import LEAGUE_TIER_DEFAULT
from league_predictor.predraft.ratings.player_scoring import (
    compute_pillar_score,
    pillar_stats_for,
    stat_weights_for,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping

    from league_predictor.analytics.player_context import SnapshotContext
    from league_predictor.analytics.row_types import PlayerDailySnapshotRow

#: A baseline standard deviation below this is numerically zero — the stat gets no z-score.
_EPS = 1e-9


def _load_pick_rows_for_year(
    conn: sqlite3.Connection,
    *,
    year_start: date,
    as_of: date,
) -> dict[str, list[dict]]:
    """player_id -> pick_score dicts (is_blind, percentile) over ``[year_start, as_of]``."""
    # Reads ``pick_scores`` unfiltered by model id — one row per slot, any vintage. The
    # upper bound stays walk-forward: games after ``as_of`` must never leak in.
    rows = conn.execute(
        """
        SELECT ps.player_id, ps.percentile,
               (SELECT da.seq FROM draft_actions da
                  WHERE da.game_id = ps.game_id AND da.kind = 'pick'
                    AND da.side = ps.side AND da.role = ps.role
                  LIMIT 1) AS focal_seq,
               (SELECT da2.seq FROM draft_actions da2
                  WHERE da2.game_id = ps.game_id AND da2.kind = 'pick'
                    AND da2.side != ps.side AND da2.role = ps.role
                  LIMIT 1) AS enemy_seq
        FROM pick_scores ps
        JOIN games g ON g.game_id = ps.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) < ?
        """,
        (year_start.isoformat(), (as_of + timedelta(days=1)).isoformat()),
    ).fetchall()
    out: dict[str, list[dict]] = {}
    for player_id, percentile, focal_seq, enemy_seq in rows:
        # Blind iff the enemy same-role pick lands later (or never / no stored draft).
        is_blind = enemy_seq is None or focal_seq is None or enemy_seq > focal_seq
        out.setdefault(player_id, []).append(
            {
                "is_blind": bool(is_blind),
                "percentile": float(percentile),
            }
        )
    return out


# Per-game stats whose z-score against the per-game tier baseline is well-defined. KDA is
# excluded: its baseline is a per-player aggregate and per-game KDA blows up at zero deaths.
_PER_GAME_PILLAR_STATS: tuple[str, ...] = (
    "golddiffat15",
    "csdiffat15",
    "xpdiffat15",
    "cspm",
    "dpm",
    "killparticipation",
    "firstblood",
    "vspm",
    "earnedgoldshare",
)


def _position_baselines(
    tier_baselines: Mapping, tier: int, role: str
) -> tuple[dict[str, float], dict[str, float]] | None:
    """The (means, stdevs) baseline for ``role`` at ``tier``, else any tier that has one."""
    baselines = tier_baselines.get(tier, {}).get(role)
    if baselines is not None:
        return baselines
    # Fall back to any available tier — same fallback the intermediates
    # loop uses when the player's tier baseline is missing.
    for t in (1, 2, 3):
        baselines = tier_baselines.get(t, {}).get(role)
        if baselines is not None:
            return baselines
    return None


def _row_zscores(row, bl_means: Mapping, bl_stdevs: Mapping) -> tuple[dict, dict]:
    """(zscores, present_mask) for one game row against the tier baseline; KDA never present."""
    zscores: dict[str, float] = {}
    present_mask: dict[str, bool] = {}
    for stat in _PER_GAME_PILLAR_STATS:
        val = row[stat]
        sd = bl_stdevs.get(stat, 0.0)
        if val is not None and sd > _EPS:
            zscores[stat] = (val - bl_means.get(stat, 0.0)) / sd
            present_mask[stat] = True
        else:
            zscores[stat] = 0.0
            present_mask[stat] = False
    # KDA explicitly not present per-game.
    zscores["kda"] = 0.0
    present_mask["kda"] = False
    return zscores, present_mask


def _row_pillar_scores(
    role: str, zscores: Mapping[str, float], present_mask: Mapping[str, bool]
) -> dict[str, float] | None:
    """Every pillar's score for one game, or None as soon as one pillar is uncomputable."""
    role_stat_weights = stat_weights_for(role)
    per_pillar: dict[str, float] = {}
    for pname, stats in pillar_stats_for(role).items():
        sw = role_stat_weights.get(pname)
        score, _dq = compute_pillar_score(zscores, stats, present_mask, stat_weights=sw)
        if score is None:
            return None
        per_pillar[pname] = score
    return per_pillar


def _compute_per_game_pillars_for_year(
    conn: sqlite3.Connection,
    *,
    year_start: date,
    as_of: date,
    tier_baselines: Mapping,
    roles_by_player: Mapping[str, str],
    player_tiers: Mapping[str, int],
) -> dict[str, dict[str, list[float]]]:
    """player_id -> pillar_name -> per-game pillar scores over the year window."""
    # Only games where every pillar is computable are kept, so the pillar lists stay
    # equal-length as ``compute_consistency_raw`` requires.
    cols = ", ".join(f"gp.{s}" for s in _PER_GAME_PILLAR_STATS)
    rows = conn.execute(
        f"""
        SELECT gp.player_id, {cols}
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) < ?
          AND gp.player_id IS NOT NULL
          AND gp.role IS NOT NULL
        """,
        (year_start.isoformat(), (as_of + timedelta(days=1)).isoformat()),
    ).fetchall()

    out: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        player_id = row["player_id"]
        # Use the canonical role from the snapshot — gp.role can drift across games
        # for off-role flexes; the snapshot pins one role per player per day.
        role = roles_by_player.get(player_id)
        if role is None:
            continue
        tier = player_tiers.get(player_id, LEAGUE_TIER_DEFAULT)
        position_baselines = _position_baselines(tier_baselines, tier, role)
        if position_baselines is None:
            continue
        bl_means, bl_stdevs = position_baselines

        zscores, present_mask = _row_zscores(row, bl_means, bl_stdevs)
        per_pillar = _row_pillar_scores(role, zscores, present_mask)
        if per_pillar is None:
            continue
        d = out.setdefault(player_id, {})
        for pname, score in per_pillar.items():
            d.setdefault(pname, []).append(score)

    return out


def _role_mean_stddev_for_consistency(
    per_game_pillars_by_player: Mapping[str, dict[str, list[float]]],
    roles_by_player: Mapping[str, str],
) -> dict[str, float]:
    """role -> mean per-player mean pillar stddev (≥2 games); the Consistency shrink prior."""
    from statistics import mean, pstdev

    by_role: dict[str, list[float]] = {}
    for player_id, pillars in per_game_pillars_by_player.items():
        role = roles_by_player.get(player_id)
        if role is None or not pillars:
            continue
        lengths = {len(v) for v in pillars.values()}
        if len(lengths) != 1 or next(iter(lengths)) < 2:  # noqa: PLR2004 - pstdev needs two
            continue
        player_mean_stddev = mean(pstdev(scores) for scores in pillars.values())
        by_role.setdefault(role, []).append(player_mean_stddev)
    return {role: mean(vals) for role, vals in by_role.items() if vals}


def _compute_draft_axes_for_snapshot(
    rows: list[PlayerDailySnapshotRow],
    conn: sqlite3.Connection,
    ctx: SnapshotContext,
    *,
    year_start: date,
    as_of: date,
) -> dict[str, DraftAxes]:
    """Assemble the batch aggregator's inputs and return the per-player ``DraftAxes``."""
    pick_rows = _load_pick_rows_for_year(conn, year_start=year_start, as_of=as_of)
    roles_by_player = {r.player_id: r.role for r in rows if r.role}
    per_game_pillars = _compute_per_game_pillars_for_year(
        conn,
        year_start=year_start,
        as_of=as_of,
        tier_baselines=ctx.tier_baselines,
        roles_by_player=roles_by_player,
        player_tiers=ctx.player_tiers,
    )
    role_mean_stddev = _role_mean_stddev_for_consistency(per_game_pillars, roles_by_player)
    return compute_player_draft_axes_batch(
        roles_by_player=roles_by_player,
        season_champions_by_player=ctx.season_champions,
        per_game_pillars_by_player=per_game_pillars,
        pick_rows_by_player=pick_rows,
        role_mean_stddev_by_role=role_mean_stddev,
    )


def _attach_draft_axes(
    rows: list[PlayerDailySnapshotRow],
    axes_by_player: Mapping[str, DraftAxes],
) -> list[PlayerDailySnapshotRow]:
    """Populate each row's draft-axis fields; players without axes pass through unchanged."""
    out: list[PlayerDailySnapshotRow] = []
    for r in rows:
        axes = axes_by_player.get(r.player_id)
        if axes is None:
            out.append(r)
            continue
        out.append(
            dataclasses.replace(
                r,
                pool_depth=axes.pool_depth,
                blind_quality=axes.blind_quality,
                counter_quality=axes.counter_quality,
                counter_rate=axes.counter_rate,
                consistency=axes.consistency,
                overall_quality=axes.overall_quality,
                pool_depth_n=axes.samples.get("pool_depth"),
                blind_quality_n=axes.samples.get("blind_quality"),
                counter_quality_n=axes.samples.get("counter_quality"),
                counter_rate_n=axes.samples.get("counter_rate"),
                consistency_n=axes.samples.get("consistency"),
            )
        )
    return out
