"""The served model: pre-draft P(team A wins)."""

from __future__ import annotations

import json

from league_predictor import artifacts, clock
from league_predictor.db import registry
from league_predictor.db.predictions_read import context_row_count
from league_predictor.lifecycle import predictions
from league_predictor.lifecycle.base import ModelLifecycle
from league_predictor.postdraft import coverage as postdraft_coverage
from league_predictor.predraft.pipeline import operations as served_ops
from league_predictor.predraft.pipeline.chosen_model import ServedModel, chosen_model
from league_predictor.predraft.pipeline.chosen_skill_filter import chosen_rating_dependency


def _live_model() -> ServedModel:
    return ServedModel.load(
        artifacts.primary_path("served"), actual_rating_dependency=chosen_rating_dependency()
    )


class _ServedLifecycle(ModelLifecycle):
    token = "served"
    recipe = chosen_model()

    def _fit_state(self, conn, *, through):
        return served_ops.build_context(conn, through=through, months=served_ops.OOS_MONTHS)

    def _fit_into(self, stage, conn, *, through, shared) -> None:
        served_ops.fit_current_model(conn, through=through, ctx=shared).save(stage.primary)
        meta = json.loads(stage.primary.read_text(encoding="utf-8"))
        meta["code_fingerprint"] = registry.code_fingerprint()
        stage.primary.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def _has_history(self, conn, model_id) -> bool:
        return context_row_count(conn, model_id, "oos") > 0

    def _prepare(self, conn, *, through, full, shared):
        return shared if shared is not None else self._fit_state(conn, through=through)

    def _write_outputs(self, conn, model_id, *, full, through, prepared) -> int:
        """The backfill rows, plus the walk-forward ``oos`` rows when ``full``."""
        model = _live_model()
        created_at = clock.utc_now_iso()
        if full:
            backfill, oos = served_ops.build_prd_coverage(
                conn, model, through=through, ctx=prepared
            )
            predictions.replace_game_predictions(
                conn, model_id, "oos", served_ops.prediction_rows(oos, model_id, "oos", created_at)
            )
        else:
            backfill = served_ops.score_backfill(
                conn, model, indexes=prepared.indexes, corpus=prepared.corpus
            )
        rows = served_ops.prediction_rows(backfill, model_id, "backfill", created_at)
        return predictions.replace_game_predictions(conn, model_id, "backfill", rows)

    def _rebuild_derived(self, conn, model_id) -> None:
        postdraft_coverage.rebuild_and_verify(
            conn,
            served_model_id=model_id,
            draft_model_id=postdraft_coverage.current_draft_model_id(conn),
        )


LIFECYCLE = _ServedLifecycle()
