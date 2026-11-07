"""Serving replay → pinned-skill-filter ``team_state`` rows, and their freshness reconcile."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import TYPE_CHECKING

from league_predictor import clock, db
from league_predictor.db import registry
from league_predictor.db.asof import team_state_coverage
from league_predictor.db.rows import TeamStateRow
from league_predictor.db.writes import drop_foreign_team_state, replace_team_state
from league_predictor.predraft.ratings.skill_filter.serving_frozen import _KEY_STRIDE

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping

    from league_predictor.predraft.ratings.skill_filter.serving import (
        BaggedSkillFilterBackend,
    )
    from league_predictor.predraft.ratings.skill_filter.serving_bag import BaggedSkillFilterIndex

logger = logging.getLogger(__name__)

# The row is queried one day AFTER the appearance day so the day's games are incorporated
# (the end-of-day snapshot), then stamped back to the appearance day.
_ONE_DAY = timedelta(days=1)


def ensure_team_state_current(conn: sqlite3.Connection) -> bool:
    """Reconcile ``team_state`` to the served rating id; True when a rebuild ran."""
    # Stale = served id has no rows, or its newest asof_date predates the corpus's newest
    # completed-game day. A rebuild costs one production rating replay; this function commits.
    from league_predictor.predraft.pipeline.chosen_skill_filter import (
        chosen_skill_filter_backend,
    )
    from league_predictor.predraft.pipeline.model_identity import (
        current_rating_model_id,
        model_row,
    )

    rating_id = current_rating_model_id()
    count, max_asof = team_state_coverage(conn, rating_id)
    latest = db.latest_played_date(conn)
    latest_day = latest.isoformat() if latest is not None else None
    is_current = count > 0 and (
        latest_day is None or (max_asof is not None and max_asof >= latest_day)
    )

    if is_current:
        deleted = drop_foreign_team_state(conn, rating_id)
        if deleted:
            conn.commit()
            logger.info(
                "team_state reconcile: coverage current under %s; dropped %d foreign-id rows",
                rating_id,
                deleted,
            )
        return False

    registry.ensure_registered(conn, model_row("rating"))
    conn.commit()
    rows = export_team_state(conn, chosen_skill_filter_backend(), rating_id)
    replace_team_state(conn, rating_id, rows)
    drop_foreign_team_state(conn, rating_id)
    conn.commit()
    logger.info(
        "team_state reconcile: rebuilt %d rows under %s (was %d rows, max asof %s)",
        len(rows),
        rating_id,
        count,
        max_asof,
    )
    return True


def export_team_state(
    corpus_conn: sqlite3.Connection, backend: BaggedSkillFilterBackend, model_id: str
) -> list[TeamStateRow]:
    """One serving replay of ``backend`` over ``corpus_conn`` → one row per (team, day)."""
    index = backend.build_index(corpus_conn, through=clock.utc_today(), query_dates=[])
    return team_state_rows(index, model_id)


def team_state_rows(index: BaggedSkillFilterIndex, model_id: str) -> list[TeamStateRow]:
    """The (team, appearance-day) rows from a built bag index; reads ``members[0]`` only."""
    member = index.members[0]
    teams, days = _appearance_queries(member._frozen_tl, member._org_index)
    if not teams:
        return []
    now = _member_states(member, teams, [d + _ONE_DAY for d in days])
    return _assemble_rows(teams, days, now, model_id)


def _appearance_queries(frozen, org_index: Mapping) -> tuple[list[str], list[date]]:
    """Flat parallel (team_id, day) lists from the frozen timelines; played days only."""
    # team_key packs key = org_id * _KEY_STRIDE + day ordinal; orgs absent from the vocab skip.
    team_of = {oi: t for t, oi in org_index.items()}
    keys = frozen.team_key.tolist()
    offsets = frozen.team_offset.tolist()
    teams: list[str] = []
    days: list[date] = []
    for oi in range(len(offsets) - 1):
        team = team_of.get(oi)
        if team is None:
            continue
        for key in keys[offsets[oi] : offsets[oi + 1]]:
            teams.append(team)
            days.append(date.fromordinal(key - oi * _KEY_STRIDE))
    return teams, days


def _assemble_rows(
    teams: list[str],
    days: list[date],
    now: list[float],
    model_id: str,
) -> list[TeamStateRow]:
    """Build one end-of-day strength row per team appearance."""
    rows: list[TeamStateRow] = []
    for team, day, state in zip(teams, days, now, strict=True):
        rows.append(
            TeamStateRow(
                team_id=team,
                model_id=model_id,
                asof_date=day.isoformat(),
                strength=state,
            )
        )
    return rows


def _member_states(member, teams: list[str], query_dates: list[date]) -> list[float]:
    """Read representative-member self-matchup strengths in one batched pass."""
    import torch

    with torch.no_grad():
        state = member._batched_team_state(torch, teams, query_dates)
        f, _hist = member._batched_pair_strength(torch, state, state)
        strength = f[:, 0]  # [K]  self-pair => f[:,0] == f[:,1]
    return strength.tolist()
