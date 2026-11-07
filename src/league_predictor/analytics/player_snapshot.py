"""``player_daily_snapshot`` mart builder: pass 1 computes pillars, pass 2 assembles rows."""

from __future__ import annotations

import dataclasses
from datetime import date
from statistics import fmean, pstdev
from typing import TYPE_CHECKING

from league_predictor.analytics._common import (
    _as_of_iso,
    _safe_json,
    tracked_builder,
)
from league_predictor.analytics.player_context import _load_snapshot_context
from league_predictor.analytics.player_draft_inputs import (
    _attach_draft_axes,
    _compute_draft_axes_for_snapshot,
)
from league_predictor.analytics.row_types import PlayerDailySnapshotRow
from league_predictor.db.names import player_name_lookup
from league_predictor.predraft.ratings.player_scoring import (
    apply_shrinkage,
    pillar_stats_for,
    z_to_performance_score,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Mapping


@dataclasses.dataclass(frozen=True, slots=True)
class PlayerIntermediate:
    """Pass-1 output for one player: pillar scores in raw z units, before role normalization."""

    player_id: str
    team_id: str | None
    role: str | None
    games_played: int
    zscores: dict[str, float]
    present_mask: dict[str, bool]
    pillar_scores: dict[str, float]
    pillar_quality: dict[str, str]


def _all_player_latest_team_role(
    conn: sqlite3.Connection,
    as_of_iso: str,
) -> list[tuple[str, str | None, str | None]]:
    """Every player's (player_id, team_id, role) from their latest game up to ``as_of_iso``."""
    # SQLite bare-column rule: with a single MAX(), the bare columns (team_id,
    # role) are taken from the row holding that max — here the player's latest game.
    rows = conn.execute(
        """
        SELECT gp.player_id, gp.team_id, gp.role, MAX(date(s.scheduled_ts)) AS max_date
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE gp.player_id IS NOT NULL AND date(s.scheduled_ts) <= ?
        GROUP BY gp.player_id
        ORDER BY gp.player_id
        """,
        (as_of_iso,),
    ).fetchall()
    return [(str(row["player_id"]), row["team_id"], row["role"]) for row in rows]


def _role_z_stats(
    entries: Iterable[PlayerIntermediate],
) -> dict[str, tuple[float, float]]:
    """Per-role mean and stdev per pillar score, keyed ``"{role}:{pillar_name}"``."""
    by_role: dict[str, list[PlayerIntermediate]] = {}
    for e in entries:
        by_role.setdefault(e.role or "_none", []).append(e)

    result: dict[str, tuple[float, float]] = {}
    for role, players in by_role.items():
        pillar_names: set[str] = set()
        for p in players:
            pillar_names.update(p.pillar_scores.keys())
        for pname in pillar_names:
            vals = [p.pillar_scores[pname] for p in players if pname in p.pillar_scores]
            n = len(vals)
            if n < 2:  # noqa: PLR2004 - a variance needs two observations
                result[f"{role}:{pname}"] = (0.0, 1.0)
                continue
            result[f"{role}:{pname}"] = (fmean(vals), max(pstdev(vals), 1e-6))
    return result


def _compute_player_intermediates(
    conn: sqlite3.Connection,
    as_of: date,
) -> list[PlayerIntermediate]:
    """Pass 1: aggregate each player's context-adjusted per-game z from the one stat home."""
    from league_predictor.predraft.ratings.impact_artifact import production_stat_context_kwargs
    from league_predictor.predraft.ratings.stat_context import (
        aggregate_prior,
        build_game_context_z,
    )

    context = build_game_context_z(conn, as_of, **production_stat_context_kwargs())
    # Unshrunk: pass 2 role-norms and then applies the display games-shrinkage itself.
    priors = aggregate_prior(context, as_of, shrink_k=0)

    intermediates: list[PlayerIntermediate] = []
    for player_id, team_id, role in _all_player_latest_team_role(conn, as_of.isoformat()):
        prior = priors.get(player_id)
        if prior is None:
            continue

        role_pillar_stats = pillar_stats_for(role)
        zscores = dict(prior.stat_z)
        present_mask = {
            stat: stat in zscores for stats in role_pillar_stats.values() for stat in stats
        }
        pillar_scores: dict[str, float] = {}
        pillar_quality: dict[str, str] = {}
        for pname, stats in role_pillar_stats.items():
            value = prior.pillar_z.get(pname)
            if value is None:
                pillar_quality[pname] = "dropped"
                continue
            pillar_scores[pname] = value
            pillar_quality[pname] = "full" if all(present_mask[s] for s in stats) else "partial"

        intermediates.append(
            PlayerIntermediate(
                player_id=player_id,
                team_id=team_id,
                role=role,
                games_played=prior.n_games,
                zscores=zscores,
                present_mask=present_mask,
                pillar_scores=pillar_scores,
                pillar_quality=pillar_quality,
            )
        )

    return intermediates


def _build_snapshot_rows(
    intermediates: list[PlayerIntermediate],
    as_of: date,
    player_name_lookup: Mapping[str, str],
    season_champions: Mapping[str, list[tuple[str, int]]],
) -> list[PlayerDailySnapshotRow]:
    """Pass 2: normalize each pillar within its role, shrink, and assemble the rows."""
    role_stats = _role_z_stats(intermediates)

    as_of_iso = as_of.isoformat()
    rows: list[PlayerDailySnapshotRow] = []
    for entry in intermediates:
        role = entry.role or "_none"
        games = entry.games_played
        zscores = entry.zscores
        present_mask = entry.present_mask
        pillar_scores = entry.pillar_scores
        pillar_quality = entry.pillar_quality

        season_champs = season_champions.get(entry.player_id, [])
        pool_data = {
            "champion_diversity": len(season_champs),
            "games_played": sum(g for _, g in season_champs),
            "top_champions": [c for c, _ in season_champs[:5]],
            "champions": [{"champion_id": c, "games": g} for c, g in season_champs[:5]],
        }

        role_pillar_stats = pillar_stats_for(entry.role)
        pillars_payload: list[dict] = []
        for pname, stats in role_pillar_stats.items():
            if pname in pillar_scores:
                mean, sd = role_stats.get(f"{role}:{pname}", (0.0, 1.0))
                performance_score = z_to_performance_score(
                    apply_shrinkage((pillar_scores[pname] - mean) / sd, games)
                )
                detail = {
                    s: round(zscores.get(s, 0.0), 3) for s in stats if present_mask.get(s, False)
                }
            else:
                performance_score = None
                detail = {}
            pillars_payload.append(
                {
                    "name": pname,
                    "rating": performance_score,
                    "detail": detail,
                    "data_quality": pillar_quality.get(pname, "dropped"),
                }
            )

        rows.append(
            PlayerDailySnapshotRow(
                player_id=entry.player_id,
                date=as_of_iso,
                team_id=entry.team_id,
                role=entry.role,
                display_name=player_name_lookup.get(entry.player_id, entry.player_id),
                champion_pool_json=_safe_json(pool_data),
                games_played=entry.games_played,
                pillars_json=_safe_json(pillars_payload),
            )
        )
    return rows


def refresh_player_daily_snapshot(
    conn: sqlite3.Connection,
    *,
    as_of_date: date | None = None,
    run_log: list[dict],
) -> int:
    """Replace the ``player_daily_snapshot`` slice for ``as_of_date``; returns row count."""
    as_of = date.fromisoformat(_as_of_iso(as_of_date))
    names = player_name_lookup(conn)
    ctx = _load_snapshot_context(conn, as_of)
    intermediates = _compute_player_intermediates(conn, as_of)
    rows = _build_snapshot_rows(intermediates, as_of, names, ctx.season_champions)

    # Draft radar axes only READ ``pick_scores``: the lifecycle draft ``backfill`` verb
    # scores them, and axes come out NULL when nothing has.
    axes_by_player = _compute_draft_axes_for_snapshot(
        rows,
        conn,
        ctx,
        year_start=date(as_of.year, 1, 1),
        as_of=as_of,
    )
    rows = _attach_draft_axes(rows, axes_by_player)

    with tracked_builder(conn, "player_daily_snapshot", as_of.isoformat(), run_log) as result:
        conn.execute("DELETE FROM player_daily_snapshot WHERE date = ?", (as_of.isoformat(),))
        conn.executemany(
            PlayerDailySnapshotRow.insert_sql(),
            [dataclasses.astuple(r) for r in rows],
        )
        result["rows"] = len(rows)
    return result["rows"]
