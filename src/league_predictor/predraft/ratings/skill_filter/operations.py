"""The rating model's fit and coverage."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING

from league_predictor import clock
from league_predictor.asof import AsOf
from league_predictor.db import latest_played_date
from league_predictor.db.registry import code_fingerprint
from league_predictor.db.rows import PredictionRow, TeamStateRow
from league_predictor.predraft.pipeline.splits import default_split
from league_predictor.predraft.ratings.skill_filter import (
    corpus as corpus_module,
)
from league_predictor.predraft.ratings.skill_filter import (
    serving,
    state_export,
    train,
)
from league_predictor.predraft.ratings.skill_filter.index_snapshot import (
    facts_identity,
    save_index_snapshot,
)
from league_predictor.predraft.ratings.skill_filter.recipe import RATING_RECIPE
from league_predictor.probability import sigmoid

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path

token = "rating"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- verbs


def fit_candidate(
    conn: sqlite3.Connection,
    *,
    output_path: Path,
    through: date | None = None,
) -> None:
    """Cold-fit the deployed K=5 recipe, bounded by ``through`` when provided."""
    data_end = _latest_match_date(conn)
    requested_end = min(data_end, through) if through is not None else data_end
    loss_end, val_end = _resolve_cutoffs(conn, requested_end)
    corpus = corpus_module.build_corpus(conn, through=val_end, norm_through=loss_end)
    fits = train.fit_production_bag(corpus, loss_end=loss_end, val_window=(loss_end, val_end))
    _save_bag(
        output_path,
        [fit.model for fit in fits],
        [fit.model.config for fit in fits],
        corpus,
        loss_end=loss_end,
        val_end=val_end,
        member_epochs=[fit.epochs for fit in fits],
        member_val_logloss=[round(float(fit.val_logloss), 4) for fit in fits],
    )


@dataclass(frozen=True, slots=True)
class RatingCoverage:
    """Rows calculated from one complete replay; lifecycle owns replacement."""

    team_state_rows: tuple[TeamStateRow, ...]
    game_rows: tuple[PredictionRow, ...]


def build_prd_coverage(
    conn: sqlite3.Connection,
    prd_id: str,
    *,
    artifact_path: Path,
    created_at: str,
) -> RatingCoverage:
    """Build the rating model's team-state and per-game coverage without writing it.

    Returns:
        Both products from one serving-index build (one corpus build, one replay): the
            ``team_state`` replay rows, and the per-game score
            ``p_a = sigmoid(rating_logit(features_ab, features_ba))`` as canonical
            ``P(team_a)`` ``context='backfill'`` rows. Does NOT commit — the caller owns the
            transaction boundary.
    """
    backend = serving.BaggedSkillFilterBackend(artifact_path, use_index_snapshot=True)
    index = backend.build_index(conn, through=clock.utc_today(), query_dates=[])

    # The same complete, leak-safe replay the backfill products consume. Publishing it makes
    # Served's skill-filter family a verified load instead of a duplicate 25-second replay.
    # sightline-ok: 35 - torch-loading module stays off the cold import path
    from league_predictor.predraft.ratings.skill_filter.artifact import artifact_sha

    if not isinstance(index.through, date):
        raise ValueError(  # noqa: TRY004 - a missing through is incomplete data, not a type bug
            "complete rating replay lacks a through date"
        )
    save_index_snapshot(
        index,
        artifact_path=artifact_path,
        # The serving backend verifies the raw torch artifact SHA.  The lifecycle's family SHA
        # intentionally frames filenames/lengths even for a one-file family and is not equal.
        artifact_sha=artifact_sha(artifact_path),
        source_identity=facts_identity(conn, through=index.through),
    )

    return RatingCoverage(
        team_state_rows=tuple(state_export.team_state_rows(index, prd_id)),
        game_rows=tuple(_score_game_rows(index, _canonical_games(conn), prd_id, now=created_at)),
    )


# --------------------------------------------------------------------------- per-game scoring


def _canonical_games(conn: sqlite3.Connection) -> list[tuple[str, str, str, AsOf, str]]:
    """Every prod game with a scheduled series, as ``(game_id, team_a, team_b, as_of, as_of_iso)``.

    Returns:
        Rows in the ``series`` canonical orientation (``team_a < team_b``, a schema CHECK), so
            a game's matchup probability is scored and stored as canonical ``P(team_a)``.
            ``as_of`` is the game's position; the index reads states strictly before its day.
    """
    rows = conn.execute(
        "SELECT g.game_id, s.team_a, s.team_b, s.scheduled_ts, s.series_id, g.game_no "
        "FROM games g JOIN series s ON s.series_id = g.series_id "
        "WHERE s.scheduled_ts IS NOT NULL ORDER BY g.game_id"
    ).fetchall()
    return [
        (game_id, team_a, team_b, AsOf.game(ts, series_id, game_no), str(ts)[:10])
        for game_id, team_a, team_b, ts, series_id, game_no in rows
    ]


def _score_game_rows(
    index,
    games: list[tuple[str, str, str, AsOf, str]],
    model_id: str,
    *,
    now: str,
) -> list[PredictionRow]:
    """Canonical ``P(team_a)`` ``backfill`` rows for ``games`` from a built rating ``index``.

    Returns:
        One row per game. The raw one-orientation ``rating_logit_diff`` is read for both team
            orderings, ``serve.rating_logit`` symmetrizes them into the native matchup logit
            (one already-symmetric call, no extra ``0.5``), and ``sigmoid`` maps it to
            ``P(team_a)`` — canonical because team A is ``series.team_a``.
    """
    if not games:
        return []
    from league_predictor.predraft.pipeline.serve import rating_logit

    queries_ab = [(team_a, team_b, as_of) for _gid, team_a, team_b, as_of, _iso in games]
    queries_ba = [(team_b, team_a, as_of) for _gid, team_a, team_b, as_of, _iso in games]
    feats_ab = index.derived_features_batch(queries_ab)
    feats_ba = index.derived_features_batch(queries_ba)
    rows: list[PredictionRow] = []
    for (game_id, _a, _b, _as_of, as_of_iso), f_ab, f_ba in zip(
        games, feats_ab, feats_ba, strict=True
    ):
        p_a = sigmoid(rating_logit(f_ab, f_ba))
        rows.append(
            PredictionRow(
                subject_kind="game",
                subject_id=game_id,
                model_id=model_id,
                p_a=p_a,
                asof_date=as_of_iso,
                context="backfill",
                created_at=now,
            )
        )
    return rows


# --------------------------------------------------------------------------- torch/engine seams
# Monkeypatchable wrappers: no torch at module load, and plumbing tests never run a real fit.


def _save_bag(
    path: Path,
    members: list,
    member_configs: list,
    corpus,
    *,
    loss_end: date,
    val_end: date,
    member_epochs: list[int] | None = None,
    member_val_logloss: list[float] | None = None,
) -> None:
    """Write one production bag with its cutoff and convergence provenance."""
    # sightline-ok: 35 - torch-loading module stays off the cold import path
    from league_predictor.predraft.ratings.skill_filter.artifact import save_bag_artifact

    notes = {
        "loss_end": loss_end.isoformat(),
        "val_end": val_end.isoformat(),
        "bag_seeds": len(members),
        "code_fingerprint": code_fingerprint(),  # the runtime source tree this fit ran from
    }
    if member_epochs is not None:
        notes["member_epochs"] = member_epochs
    if member_val_logloss is not None:
        notes["member_val_logloss"] = member_val_logloss
    save_bag_artifact(
        path,
        members,
        member_configs[0],
        corpus.norm,
        corpus.vocab.league_index,
        signature=corpus.signature,
        notes=notes,
        recipe=RATING_RECIPE.to_dict(),
    )


# --------------------------------------------------------------------------- small helpers


def _latest_match_date(conn: sqlite3.Connection) -> date:
    latest = latest_played_date(conn)
    if latest is None:
        raise ValueError("no finished matches to fit the rating model on")
    return latest


def _resolve_cutoffs(conn: sqlite3.Connection, through: date) -> tuple[date, date]:
    """The ``(loss_end, val_end)`` for a rating fit as of ``through``, anchored on real data.

    Returns:
        ``val_end`` is the day after the latest match on or before ``through`` (never a future
            date with no games). The early-stop holdout ``[loss_end, val_end)`` is sized by
            GAME COUNT — at least as many games as the canonical split's val window — but never
            narrower than that window's calendar length, so an off-season lull cannot leave a
            holdout so thin that early stopping fires immediately and undertrains every member.

    Raises:
        RuntimeError: the resolved holdout holds fewer games than the canonical window while
            earlier games exist to widen into. Only a data-limited corpus (nothing earlier to
            widen into) may fit with a thinner holdout.
    """
    split = default_split()
    val_days = (split.val_end - split.train_end).days
    target_games = _scheduled_game_count(conn, split.train_end, split.val_end)
    last = min(through, _latest_match_date(conn))
    val_end = last + timedelta(days=1)
    loss_end = val_end - timedelta(days=val_days)  # the calendar floor
    if target_games > 0:
        loss_end = min(loss_end, _loss_end_for_game_count(conn, val_end, target_games))
        held = _scheduled_game_count(conn, loss_end, val_end)
        if held < target_games and _scheduled_game_count(conn, date(1, 1, 1), loss_end) > 0:
            raise RuntimeError(
                f"rating fit holdout [{loss_end}, {val_end}) holds {held} scheduled maps, fewer "
                f"than the canonical val window's {target_games}, while earlier games exist to "
                "widen into - a game-thin early-stop holdout makes every bag member stop almost "
                "immediately and undertrains the whole ensemble; refusing to fit"
            )
    return loss_end, val_end


def _scheduled_game_count(conn: sqlite3.Connection, start: date, end: date) -> int:
    """The number of scheduled maps with a series date in ``[start, end)``."""
    row = conn.execute(
        "SELECT COUNT(*) FROM games g JOIN series s ON s.series_id = g.series_id "
        "WHERE s.scheduled_ts IS NOT NULL "
        "AND substr(s.scheduled_ts, 1, 10) >= ? AND substr(s.scheduled_ts, 1, 10) < ?",
        (start.isoformat(), end.isoformat()),
    ).fetchone()
    return int(row[0])


def _loss_end_for_game_count(conn: sqlite3.Connection, val_end: date, target: int) -> date:
    """The latest day ``D`` for which ``[D, val_end)`` holds at least ``target`` scheduled maps.

    Returns:
        The date of the ``target``-th most-recent map before ``val_end`` (every map on ``D`` is
            inside the window, so the count is ``>= target``); the earliest scheduled day when
            fewer than ``target`` maps exist; ``val_end`` when there are no maps at all.
    """
    row = conn.execute(
        "SELECT substr(s.scheduled_ts, 1, 10) FROM games g "
        "JOIN series s ON s.series_id = g.series_id "
        "WHERE s.scheduled_ts IS NOT NULL AND substr(s.scheduled_ts, 1, 10) < ? "
        "ORDER BY substr(s.scheduled_ts, 1, 10) DESC LIMIT 1 OFFSET ?",
        (val_end.isoformat(), target - 1),
    ).fetchone()
    if row is None:  # fewer than `target` maps -> widen to all data
        row = conn.execute(
            "SELECT MIN(substr(s.scheduled_ts, 1, 10)) FROM games g JOIN series s "
            "ON s.series_id = g.series_id WHERE s.scheduled_ts IS NOT NULL "
            "AND substr(s.scheduled_ts, 1, 10) < ?",
            (val_end.isoformat(),),
        ).fetchone()
    if row is None or row[0] is None:
        return val_end
    return date.fromisoformat(row[0])
