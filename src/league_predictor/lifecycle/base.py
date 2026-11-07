"""The two verbs every model shares; each model's math lives behind the hooks."""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from league_predictor import artifacts
from league_predictor.lifecycle import predictions

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterator
    from datetime import date
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class Written:
    """The model id whose outputs were written, and how many rows."""

    model_id: str
    rows: int
    full: bool


class Stage:
    """A private directory the fit writes into; ``publish`` swaps its family into place."""

    def __init__(self, token: str) -> None:
        self.token = token
        self.live_dir = artifacts.token_dir(token)
        self.dir = self.live_dir / f".staging.{os.getpid()}"
        self.backup = self.live_dir / f".backup.{os.getpid()}"
        self._published: list[Path] = []
        self._moved: list[Path] = []

    @property
    def root(self) -> Path:
        """The suffix-less family stem inside the stage."""
        return self.dir / self.token

    @property
    def primary(self) -> Path:
        return self.dir / artifacts.primary_path(self.token).name

    def publish(self) -> None:
        """Move the old family aside and the staged one into place."""
        staged = artifacts.family_paths(self.token, self.primary)
        missing = [path.name for path in staged if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"{self.token} fit wrote an incomplete family: {missing}")
        live = artifacts.primary_path(self.token)
        self.backup.mkdir(parents=True)
        for path in artifacts.family_paths(self.token) if live.is_file() else []:
            path.replace(self.backup / path.name)
            self._moved.append(path)
        for path in staged:
            dest = self.live_dir / path.name
            path.replace(dest)
            self._published.append(dest)

    def restore(self) -> None:
        """Put the old family back; the backup survives if this itself fails."""
        for path in self._published:
            path.unlink(missing_ok=True)
        for path in self._moved:
            (self.backup / path.name).replace(path)
        shutil.rmtree(self.backup, ignore_errors=True)


@contextmanager
def staged(token: str) -> Iterator[Stage]:
    """A stage whose publish is undone if anything in the block raises."""
    stage = Stage(token)
    stage.dir.mkdir(parents=True, exist_ok=True)
    try:
        yield stage
    except BaseException:
        stage.restore()
        raise
    finally:
        shutil.rmtree(stage.dir, ignore_errors=True)
    shutil.rmtree(stage.backup, ignore_errors=True)


class ModelLifecycle:
    """``retrain`` refits and rewrites every output; ``backfill`` applies the current artifact."""

    token: str
    recipe: Any

    def retrain(self, conn: sqlite3.Connection, *, through: date) -> Written:
        """Fit on games before ``through``, replace the artifact, write its full outputs.

        A failure anywhere restores the previous artifact and rolls back the database.
        """
        with staged(self.token) as stage:
            shared = self._fit_state(conn, through=through)
            self._fit_into(stage, conn, through=through, shared=shared)
            stage.publish()
            return self._write(conn, full=True, through=through, shared=shared)

    def backfill(self, conn: sqlite3.Connection, *, through: date) -> Written:
        """Score every game with the current artifact.

        The walk-forward history (served and draft) is written only when this artifact has
        none in the database yet: a fresh database or a newly pulled artifact. That run is
        slower, about 35 minutes instead of 7.
        """
        model_id, _sha = artifacts.mint(self.token)
        full = not self._has_history(conn, model_id)
        return self._write(conn, full=full, through=through, shared=None)

    def _write(self, conn, *, full: bool, through: date, shared) -> Written:
        # Preparation may read on other connections, so it precedes the first write.
        prepared = self._prepare(conn, through=through, full=full, shared=shared)
        try:
            model_id = predictions.register(conn, self.token)
            rows = self._write_outputs(
                conn, model_id, full=full, through=through, prepared=prepared
            )
            self._rebuild_derived(conn, model_id)
            self._prune_superseded(conn, model_id)
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        return Written(model_id, rows, full)

    # ------------------------------------------------------------------- hooks
    def _fit_state(self, conn: sqlite3.Connection, *, through: date):
        """State the fit and the output write share; default None."""

    def _fit_into(self, stage: Stage, conn: sqlite3.Connection, *, through: date, shared) -> None:
        raise NotImplementedError

    def _has_history(self, conn: sqlite3.Connection, model_id: str) -> bool:
        """Whether this model's walk-forward history is already written; default True."""
        return True

    def _prepare(self, conn: sqlite3.Connection, *, through: date, full: bool, shared):
        """Read-only state built before the first write; default None."""

    def _write_outputs(
        self, conn: sqlite3.Connection, model_id: str, *, full: bool, through: date, prepared
    ) -> int:
        raise NotImplementedError

    def _rebuild_derived(self, conn: sqlite3.Connection, model_id: str) -> None:
        """Rebuild outputs composed from this model's; default none."""

    def _prune_superseded(self, conn: sqlite3.Connection, model_id: str) -> None:
        predictions.prune_superseded(conn, self.token, model_id)
