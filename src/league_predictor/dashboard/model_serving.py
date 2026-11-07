"""Live model handles keyed on the artifacts and committed facts."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from league_predictor import artifacts
from league_predictor.asof import AsOf, MatchContext
from league_predictor.db import facts_generation, get_connection
from league_predictor.draft_grading.suggest import (
    DraftSuggestEngine,
    board_after_pick,
    require_legal_draft,
)

if TYPE_CHECKING:
    from datetime import date
    from pathlib import Path

    from league_predictor.draft_grading.suggest import DraftScore

# --------------------------------------------------------------------- served


@dataclass(frozen=True)
class SeriesPrediction:
    """P(team_a) for one matchup: the next map, then each series length from the state scored.

    Attributes:
        state: maps won so far ``(team_a, team_b)``; the pre-series state without a context.
    """

    p_map: float
    p_bo3: float
    p_bo5: float
    as_of: date
    state: tuple[int, int] = (0, 0)


class ServedOracle:
    """Warm-once live handle on the served model for on-demand matchup predictions."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._status = "cold"  # cold | warming | ready | error
        self._error = ""
        self._model = None
        self._indexes = None
        self._conn = None  # outlives the build: the original family's index falls back to it
        self._built_key: tuple[int, int] | None = None

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def _prd_key(self) -> tuple[int, int]:
        assert self._conn is not None
        return artifacts.primary_path("served").stat().st_mtime_ns, facts_generation(self._conn)

    # sightline-ok: 14 - the matchup key (team_a, team_b, as_of) is owned by predraft.serve
    def predict_series(
        self,
        team_a: str,
        team_b: str,
        as_of: date,
        *,
        context: MatchContext | None = None,
    ) -> SeriesPrediction | None:
        """P(team_a beats team_b) as-of a date: the next map, then Bo3/Bo5 from the state.

        Args:
            context: the match's format, stage and maps won so far; ``None`` prices a
                pre-series map under the pair's scheduled match, else neutral.

        Returns:
            ``None`` until warm; a cold or errored handle re-arms the background build,
            and an artifact swap answers ``None`` until the rebuild lands, never the stale
            model. Scoring runs under the handle lock (~0.15s, serialized).
        """
        from league_predictor.predraft.pipeline import serve

        query = AsOf.of(as_of).with_context(context)
        state = (0, 0) if context is None else (context.wins_for(team_a), context.wins_for(team_b))
        with self._lock:
            if self._status == "ready" and self._built_key != self._prd_key():
                self._status = "cold"  # artifact swapped under us -> rebuild
            if self._status in ("cold", "error"):
                self._status = "warming"
                threading.Thread(target=self._build, daemon=True).start()
            if self._status != "ready":
                return None
            assert self._model is not None  # ready => built
            assert self._indexes is not None  # ready => built
            p_map = float(serve.predict(self._model, self._indexes, team_a, team_b, as_of=query))

        return SeriesPrediction(
            p_map=p_map,
            p_bo3=float(serve.series_prob(p_map, "Bo3", state=state)),
            p_bo5=float(serve.series_prob(p_map, "Bo5", state=state)),
            as_of=as_of,
            state=state,
        )

    def _build(self) -> None:
        """Warm the handle: minutes, CPU-heavy; the connection outlives it for the indexes."""
        from league_predictor.predraft.pipeline.live import load_live_oracle

        conn = None
        try:
            conn = get_connection(self._db_path, check_same_thread=False)
            key = (
                artifacts.primary_path("served").stat().st_mtime_ns,
                facts_generation(conn),
            )
            model, indexes = load_live_oracle(conn, artifacts.primary_path("served"))
        except Exception as exc:  # noqa: BLE001 - background warm thread; any failure must land as status='error'
            if conn is not None:
                conn.close()
            with self._lock:
                self._status = "error"
                self._error = f"{type(exc).__name__}: {exc}"
            return
        with self._lock:
            if self._conn is not None:
                self._conn.close()  # the superseded build's connection
            self._model, self._indexes, self._conn = model, indexes, conn
            self._built_key = key
            self._status = "ready"
            self._error = ""


# ---------------------------------------------------------------------- draft


class DraftScorer:
    """Live handle on the draft grader, lazily loading ``draft_grading.suggest``."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._lock = threading.Lock()
        self._engine = None
        self._conn = None
        self._loaded_key: tuple[int, int] | None = None

    def _loaded_engine(self) -> DraftSuggestEngine:
        """The suggest engine, loaded (or reloaded after a retrain) on a fresh connection."""
        if self._conn is None:
            self._conn = get_connection(self._db_path, check_same_thread=False)
        key = (
            artifacts.primary_path("draft").stat().st_mtime_ns,
            facts_generation(self._conn),
        )
        if self._engine is not None and self._loaded_key == key:
            return self._engine
        self._engine = DraftSuggestEngine.load(
            self._conn, artifact=artifacts.artifact_root("draft")
        )
        if facts_generation(self._conn) != key[1]:
            self._engine = None
            raise RuntimeError("Facts changed while loading the draft model; retry the request")
        self._loaded_key = key
        return self._engine

    # sightline-ok: 14 - the board's home is draft_grading.suggest, which these calls forward to
    def score(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
    ) -> DraftScore:
        """Grade the draft as it stands; raises ValueError on an illegal draft, before load."""

        require_legal_draft(blue_picks, red_picks, blue_bans, red_bans)
        with self._lock:
            return self._loaded_engine().score(blue_picks, red_picks, blue_bans, red_bans)

    def assigned_roles(
        self, blue_picks: list[str], red_picks: list[str]
    ) -> dict[str, dict[str, str]]:
        """Both sides' champion -> role maps under the engine's play-count assignment."""
        with self._lock:
            return self._loaded_engine().assigned_roles(blue_picks, red_picks)

    # sightline-ok: 14 - the board's home is draft_grading.suggest, which these calls forward to
    def suggest(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
        *,
        side: str,
        top_n: int = 5,
    ) -> tuple[DraftScore, list[dict[str, Any]]]:
        """Rank ``side``'s next picks (fast one-ply, ~1s): (standing score, ranked lines).

        Raises:
            ValueError: Illegal draft (checked before any artifact load) or unknown ``side``.
        """

        require_legal_draft(blue_picks, red_picks, blue_bans, red_bans)
        with self._lock:
            base, lines = self._loaded_engine().suggest(
                blue_picks, red_picks, blue_bans, red_bans, side=side, top_n=top_n, rollout=False
            )
        assert base is not None  # the no-rollout path always scores the standing draft
        return base, lines

    def rollout_values(
        self,
        blue_picks: list[str],
        red_picks: list[str],
        blue_bans: list[str],
        red_bans: list[str],
        *,
        side: str,
        champions: list[str],
    ) -> list[float]:
        """End-of-draft P(blue) per candidate, aligned to ``champions``; one lock hold per panel.

        Raises:
            ValueError: Illegal draft or candidate (checked before any load), unknown ``side``.
        """

        for champion in champions:
            blue_after, red_after = board_after_pick(blue_picks, red_picks, side, champion)
            require_legal_draft(blue_after, red_after, blue_bans, red_bans)
        with self._lock:
            return self._loaded_engine().rollout_values(
                blue_picks, red_picks, blue_bans, red_bans, side=side, champions=champions
            )
