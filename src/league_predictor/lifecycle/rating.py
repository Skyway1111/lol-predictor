"""The rating model: the skill filter's latent player and team strength."""

from __future__ import annotations

from league_predictor import artifacts, clock
from league_predictor.db.writes import drop_foreign_team_state, replace_team_state
from league_predictor.lifecycle import predictions
from league_predictor.lifecycle.base import ModelLifecycle
from league_predictor.predraft.ratings.skill_filter import operations
from league_predictor.predraft.ratings.skill_filter.recipe import RATING_RECIPE


class _RatingLifecycle(ModelLifecycle):
    token = "rating"
    recipe = RATING_RECIPE

    def _fit_into(self, stage, conn, *, through, shared) -> None:
        operations.fit_candidate(conn, output_path=stage.primary, through=through)

    def _write_outputs(self, conn, model_id, *, full, through, prepared) -> int:
        coverage = operations.build_prd_coverage(
            conn,
            model_id,
            artifact_path=artifacts.primary_path("rating"),
            created_at=clock.utc_now_iso(),
        )
        replace_team_state(conn, model_id, list(coverage.team_state_rows))
        drop_foreign_team_state(conn, model_id)
        return predictions.replace_game_predictions(
            conn, model_id, "backfill", list(coverage.game_rows)
        )


LIFECYCLE = _RatingLifecycle()
