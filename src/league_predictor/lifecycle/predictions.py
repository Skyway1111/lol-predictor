"""Model registration and per-game ``predictions`` writes."""

from __future__ import annotations

from typing import TYPE_CHECKING

from league_predictor import artifacts
from league_predictor.db import registry, writes
from league_predictor.db.rows import ModelRow, PredictionRow
from league_predictor.db.schema import PREDICTION_CONTEXTS

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable


def register(conn: sqlite3.Connection, token: str) -> str:
    """Register the live artifact's id and return it; no commit."""
    model_id, sha = artifacts.mint(token)
    registry.ensure_registered(conn, ModelRow(model_id=model_id, kind=token, artifact_sha=sha))
    return model_id


def replace_game_predictions(
    conn: sqlite3.Connection,
    model_id: str,
    context: str,
    rows: Iterable[PredictionRow],
) -> int:
    """Replace a model's ``game`` rows of one context; no commit. Returns rows written."""
    if context not in PREDICTION_CONTEXTS:
        raise ValueError(f"context {context!r} is not one of {PREDICTION_CONTEXTS}")
    rows = list(rows)
    for row in rows:
        if row.subject_kind != "game" or row.model_id != model_id or row.context != context:
            raise ValueError(
                f"row ({row.subject_kind!r}, {row.model_id!r}, {row.context!r}) does not "
                f"match the target ('game', {model_id!r}, {context!r})"
            )
    if registry.get_model(conn, model_id) is None:
        raise ValueError(f"model_id {model_id!r} is not registered; register it first")
    conn.execute(
        "DELETE FROM predictions WHERE model_id = ? AND context = ? AND subject_kind = 'game'",
        (model_id, context),
    )
    writes.append_predictions(conn, rows)
    return len(rows)


def prune_superseded(conn: sqlite3.Connection, token: str, keep_model_id: str) -> int:
    """Delete the token's ``backfill``/``oos`` rows under every other id; no commit."""
    artifacts.validate_token(token)
    placeholders = ", ".join("?" for _ in PREDICTION_CONTEXTS)
    cursor = conn.execute(
        f"DELETE FROM predictions WHERE context IN ({placeholders}) "
        "AND model_id LIKE ? AND model_id != ?",
        (*PREDICTION_CONTEXTS, f"{token}:%", keep_model_id),
    )
    return cursor.rowcount
