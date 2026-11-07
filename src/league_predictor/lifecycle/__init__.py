"""Retrain and backfill for the four models; ``get(token)`` imports one lazily."""

from __future__ import annotations

from functools import cache
from importlib import import_module
from typing import TYPE_CHECKING

from league_predictor.artifacts import validate_token

if TYPE_CHECKING:
    from league_predictor.lifecycle.base import ModelLifecycle

#: Dependency order: served reads the rating model, draft reads served's walk-forward priors.
ORDER = ("rating", "served", "draft", "impact")


@cache
def get(token: str) -> ModelLifecycle:
    """The token's lifecycle; the lazy import keeps torch off unrelated commands."""
    validate_token(token)
    return import_module(f"league_predictor.lifecycle.{token}").LIFECYCLE
