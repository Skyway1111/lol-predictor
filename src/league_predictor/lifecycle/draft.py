"""The draft model: grades a finished draft from champions alone."""

from __future__ import annotations

import logging

from league_predictor import artifacts, clock
from league_predictor.db.registry import code_fingerprint, prune_orphaned_models
from league_predictor.draft_grading import coverage
from league_predictor.draft_grading.training import provenance, set_fit
from league_predictor.lifecycle.base import ModelLifecycle
from league_predictor.postdraft import coverage as postdraft_coverage

logger = logging.getLogger(__name__)

#: The tables whose rows keep a draft ``models`` row alive.
_PRUNE_REFS = (
    ("draft_scores", "model_id"),
    ("pick_scores", "model_id"),
    ("postdraft_scores", "draft_model_id"),
)


def _served_id(conn) -> str:
    served_id = postdraft_coverage.corpus_served_model_id(conn)
    if served_id is None:
        raise RuntimeError("draft needs the served model's outputs; backfill served first")
    return served_id


class _DraftLifecycle(ModelLifecycle):
    token = "draft"

    @property
    def recipe(self):
        from league_predictor.draft_grading import recipe as draft_recipe

        return draft_recipe.DRAFT_RECIPE

    def _fit_into(self, stage, conn, *, through, shared) -> None:
        logger.info("draft train: %s", provenance.knowledge_freshness(clock.utc_today()))
        set_fit.fit_candidate(
            conn,
            stage.root,
            fit_through=through,
            production_probs=provenance.served_oos_probs(conn, _served_id(conn)),
            code_fingerprint=code_fingerprint(),
        )

    def _has_history(self, conn, model_id) -> bool:
        row = conn.execute(
            "SELECT 1 FROM draft_scores WHERE model_id = ? AND context = 'oos' LIMIT 1",
            (model_id,),
        ).fetchone()
        return row is not None

    def _write_outputs(self, conn, model_id, *, full, through, prepared) -> int:
        """Rescore every draft, then rewrite the walk-forward ``oos`` rows when ``full``."""
        counts = coverage.rescore_all(conn, model_id, model_path=artifacts.artifact_root("draft"))
        if full:
            counts["oos"] = coverage.write_oos_walk(
                conn,
                model_id,
                through=through,
                served_id=_served_id(conn),
                code_fingerprint=code_fingerprint(),
            )
        logger.info("draft outputs (id %s): %s", model_id, counts)
        return counts["scored"]

    def _rebuild_derived(self, conn, model_id) -> None:
        postdraft_coverage.rebuild_and_verify(
            conn,
            draft_model_id=model_id,
            served_model_id=postdraft_coverage.corpus_served_model_id(conn),
        )

    def _prune_superseded(self, conn, model_id) -> None:
        prune_orphaned_models(conn, "draft", _PRUNE_REFS)


LIFECYCLE = _DraftLifecycle()
