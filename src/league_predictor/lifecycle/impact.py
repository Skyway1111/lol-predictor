"""The Impact model: per-player map-win% added over a role's average premier starter."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from league_predictor import artifacts
from league_predictor.db import registry
from league_predictor.lifecycle.base import ModelLifecycle
from league_predictor.predraft.ratings import impact_operations as operations
from league_predictor.predraft.ratings.impact_artifact import player_score_rows
from league_predictor.predraft.ratings.player_impact import IMPACT_RECIPE

if TYPE_CHECKING:
    import sqlite3

    from league_predictor.predraft.ratings.impact_history import ImpactHistoryBuild


def _replace_impact_history(conn: sqlite3.Connection, build: ImpactHistoryBuild) -> None:
    if build.influential_games != build.eligible_games:
        raise RuntimeError(
            f"Impact history: {build.eligible_games} eligible games but only "
            f"{build.influential_games} influential games"
        )
    conn.execute("DELETE FROM impact_history")
    conn.executemany(
        "INSERT INTO impact_history (player_id, asof_date, impact_logodds) VALUES (?, ?, ?)",
        ((row.player_id, row.asof_date.isoformat(), row.impact_logodds) for row in build.rows),
    )


class _ImpactLifecycle(ModelLifecycle):
    token = "impact"
    recipe = IMPACT_RECIPE

    def _fit_into(self, stage, conn, *, through, shared) -> None:
        artifact = operations.fit_candidate(conn, through=through)
        stage.primary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    def _prepare(self, conn, *, through, full, shared):
        return operations.build_current_history(conn, through=through)

    def _write_outputs(self, conn, model_id, *, full, through, prepared) -> int:
        rows = player_score_rows(model_id=model_id, artifact_path=artifacts.primary_path("impact"))
        conn.execute("DELETE FROM player_impact_scores WHERE model_id = ?", (model_id,))
        conn.executemany(
            "INSERT INTO player_impact_scores (player_id, model_id, asof_date, role, impact, "
            "stats_pp, results_pp, percentile, grade) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        _replace_impact_history(conn, prepared)
        return len(rows)

    def _prune_superseded(self, conn, model_id) -> None:
        conn.execute(
            "DELETE FROM player_impact_scores WHERE model_id LIKE 'impact:%' AND model_id != ?",
            (model_id,),
        )
        registry.prune_orphaned_models(conn, "impact", (("player_impact_scores", "model_id"),))


LIFECYCLE = _ImpactLifecycle()
