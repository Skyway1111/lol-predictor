"""The one ``draft_scores`` read rule: one row per (game, model), ``oos`` winning ``backfill``."""

from __future__ import annotations

# The predicate names the alias `ds`, so a caller that aliases draft_scores differently gets
# a loud "no such column: ds" instead of a wrong row set.


def preferred_context_predicate() -> str:
    """SQL true for the walk-forward ``oos`` row of a (game, model), else its ``backfill``."""
    return (
        "ds.context = (SELECT d.context FROM draft_scores d "
        "WHERE d.game_id = ds.game_id AND d.model_id = ds.model_id "
        "ORDER BY d.context = 'oos' DESC LIMIT 1)"
    )
