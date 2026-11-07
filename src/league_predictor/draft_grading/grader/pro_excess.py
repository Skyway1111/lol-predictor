"""Walk-forward team-form and impact baselines for the decayed champion stats."""

from __future__ import annotations

import math
from bisect import bisect_left
from datetime import datetime
from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.decay import half_life_rate
from league_predictor.draft_grading.grader.corpus_memo import memo
from league_predictor.probability import logit, sigmoid

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Sequence


# Bayesian shrinkage pseudo-count for the residualized pro stat — pulls
# thin-sample champions toward 0.0 excess (no evidence of over/under-performance).
SHRINKAGE_K: float = 20.0

# Exponential time-decay half-life for BulkStatsCache's excess and pick-rate stats.
HALF_LIFE_DAYS: float = 30.0

# Team-form debiasing baseline: each result is compared against the team's decayed win rate
# strictly before that game, shrunk toward 0.5. Slower half-life than the champion stat.
TEAM_FORM_HALF_LIFE_DAYS: float = 90.0
TEAM_FORM_SHRINKAGE_K: float = 10.0

# The two pro-excess debiasing baselines; "team_form" is the default. Selection is an
# explicit parameter (the artifact metadata records the selected baseline).
PRO_EXCESS_BASELINES: tuple[str, ...] = ("team_form", "impact_bt")

# The residual targets an outcome fit may ride as the boosters' init_score
# (SetTrainConfig.outcome_offset).
OUTCOME_OFFSETS: tuple[str, ...] = ("none", "team_form")


def _build_team_form(
    match_rows,
    half_life_days: float = TEAM_FORM_HALF_LIFE_DAYS,
) -> dict[tuple[str, str], float]:
    """Decayed trailing win rate for each ``(team_id, game_id)``, strictly before that game.

    Args:
        match_rows: ``(game_id, series_start, series_id, game_no, team1, team2, winner)``.
            Earlier series and earlier maps of the same series are visible.

    Returns:
        Form per ``(team_id, match_id)``, shrunk toward 0.5, so a team with no prior
        games has form 0.5.
    """
    decay_rate = half_life_rate(half_life_days)
    k = TEAM_FORM_SHRINKAGE_K

    parsed = [row for row in match_rows if ordinal_or_none(row[1]) is not None]
    parsed.sort(key=lambda row: (row[1], row[2], row[3]))
    state: dict[str, tuple[int, float, float]] = {}
    form: dict[tuple[str, str], float] = {}
    for stamp, games in groupby(parsed, key=lambda row: row[1]):
        ordinal = datetime.fromisoformat(stamp).toordinal()
        own_maps: dict[tuple[str, str], tuple[float, int]] = {}
        pending: dict[str, tuple[float, int]] = {}
        for match_id, _, series_id, _, team1, team2, winner in games:
            for team in (team1, team2):
                if team is None:
                    continue
                last, wins, total = state.get(team, (ordinal, 0.0, 0.0))
                decay = math.exp(-decay_rate * (ordinal - last))
                wins, total = wins * decay, total * decay
                state[team] = (ordinal, wins, total)
                own_wins, own_total = own_maps.get((series_id, team), (0.0, 0))
                form[(team, match_id)] = (wins + own_wins + k * 0.5) / (total + own_total + k)
                result = float(winner == team)
                own_maps[(series_id, team)] = (own_wins + result, own_total + 1)
                added_wins, added_total = pending.get(team, (0.0, 0))
                pending[team] = (added_wins + result, added_total + 1)
        for team, (wins, total) in pending.items():
            _, prior_wins, prior_total = state[team]
            state[team] = (ordinal, prior_wins + wins, prior_total + total)
    return form


def _team_form(db: sqlite3.Connection) -> dict[tuple[str, str], float]:
    """Memoized trailing form for each team and game in the visible facts state."""

    def build():
        rows = db.execute(
            "SELECT g.game_id, s.scheduled_ts, s.series_id, g.game_no, s.team_a, s.team_b, g.winner "
            "FROM games g JOIN series s ON s.series_id = g.series_id "
            "WHERE s.scheduled_ts IS NOT NULL AND g.winner IS NOT NULL"
        ).fetchall()
        return _build_team_form(rows)

    return memo(db, "team_form", build)


def team_form_offsets(db: sqlite3.Connection, game_ids: Sequence[str]) -> np.ndarray:
    """Walk-forward team-strength log-odds per game, row-aligned to ``game_ids``.

    Returns:
        ``logit(form_blue) - logit(form_red)`` over the decayed trailing win rate strictly
        before each series plus its own earlier maps. A side with no history reads
        form 0.5 and so contributes exactly 0.0.

    Raises:
        RuntimeError: a game id resolves no blue/red form entry — the corpus join drifted
            under the fit, never a silently offset-free row.
    """
    form = _team_form(db)
    sides = {
        row[0]: (row[1], row[2])
        for row in db.execute("SELECT game_id, blue_team, red_team FROM games")
    }
    pairs = [(game_id, sides.get(game_id, (None, None))) for game_id in game_ids]
    missing = [g for g, (blue, red) in pairs if (blue, g) not in form or (red, g) not in form]
    if missing:
        raise RuntimeError(
            f"team-form offsets: {len(missing)} rows have no dated blue/red form entry "
            f"(first: {missing[:5]}) — the corpus join drifted under the fit."
        )
    return np.array(
        [logit(form[(blue, g)]) - logit(form[(red, g)]) for g, (blue, red) in pairs],
        dtype=np.float64,
    )


def require_impact_history(db: sqlite3.Connection) -> None:
    """RAISE unless ``impact_history`` exists and has rows — the ``impact_bt`` gate.

    Raises:
        RuntimeError: when the table is missing or empty.
    """
    table = db.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'impact_history'"
    ).fetchone()
    n = db.execute("SELECT COUNT(*) FROM impact_history").fetchone()[0] if table else 0
    if not n:
        raise RuntimeError(
            "pro_excess_baseline='impact_bt' needs a populated impact_history table, but it "
            f"is {'EMPTY' if table else 'MISSING'} in this database — serving anyway "
            "would silently collapse every debias baseline to 0.5 (a model-meaning "
            "change, not a degrade). Fix: run `lol-predict backfill impact` to build the "
            "walk-forward Impact grid."
        )


def _lineups_by_game(lineup_rows) -> dict[str, tuple[int, dict[str, list[str]]]]:
    """``game_id -> (date ordinal, team_id -> player_ids)``; games with an unparseable date drop."""
    games: dict[str, tuple[int, dict[str, list[str]]]] = {}
    for game_id, team_id, player_id, date_str in lineup_rows:
        entry = games.get(game_id)
        if entry is None:
            ordinal = ordinal_or_none(date_str)
            if ordinal is None:
                continue
            entry = (ordinal, {})
            games[game_id] = entry
        entry[1].setdefault(team_id, []).append(player_id)
    return games


def _build_impact_baseline(impact_rows, lineup_rows) -> dict[tuple[str, str], float]:
    """Impact Bradley-Terry expectation for each ``(team_id, game_id)``, walk-forward.

    Args:
        impact_rows: ``(player_id, asof_date, impact_logodds)`` from ``impact_history``; for a
            game on date D only rows STRICTLY before D count, an unrated player 0.0.
        lineup_rows: ``(game_id, team_id, player_id, date_str)`` for every game's two lineups.

    Returns:
        ``sigmoid(Σ impact(team) - Σ impact(enemy))`` per ``(team_id, game_id)``; 0.5 when
        neither side has a rated player.
    """
    by_player: dict[str, list[tuple[int, float]]] = {}
    for player_id, asof_date, logodds in impact_rows:
        ordinal = ordinal_or_none(asof_date)
        if ordinal is None:
            continue
        by_player.setdefault(player_id, []).append((ordinal, logodds))
    history: dict[str, tuple[list[int], list[float]]] = {}
    for player_id, entries in by_player.items():
        entries.sort()
        history[player_id] = ([e[0] for e in entries], [e[1] for e in entries])

    def impact_before(player_id: str, ordinal: int) -> float:
        entry = history.get(player_id)
        if entry is None:
            return 0.0
        dates, values = entry
        idx = bisect_left(dates, ordinal)  # entries [0, idx) are strictly before
        return values[idx - 1] if idx else 0.0

    baseline: dict[tuple[str, str], float] = {}
    for game_id, (ordinal, teams) in _lineups_by_game(lineup_rows).items():
        strengths = {
            team: sum(impact_before(pid, ordinal) for pid in players)
            for team, players in teams.items()
        }
        total = sum(strengths.values())
        for team, own in strengths.items():
            baseline[(team, game_id)] = sigmoid(own - (total - own))
    return baseline


def _impact_baseline(db: sqlite3.Connection) -> dict[tuple[str, str], float]:
    """Memoized Impact BT expectation for each team and game in the visible facts state.

    Raises:
        RuntimeError: via :func:`require_impact_history` when the Impact grid is missing or empty.
    """

    def build():
        require_impact_history(db)
        impact_rows = db.execute(
            "SELECT player_id, asof_date, impact_logodds FROM impact_history"
        ).fetchall()
        lineup_rows = db.execute(
            "SELECT gp.game_id, gp.team_id, gp.player_id, date(s.scheduled_ts) "
            "FROM game_players gp JOIN games g ON g.game_id = gp.game_id "
            "JOIN series s ON s.series_id = g.series_id "
            "WHERE s.scheduled_ts IS NOT NULL"
        ).fetchall()
        return _build_impact_baseline(impact_rows, lineup_rows)

    return memo(db, "impact_baseline", build)


def _baseline_map(db: sqlite3.Connection, baseline: str) -> dict[tuple[str, str], float]:
    """Selected ``(team_id, game_id) -> debias`` map; each baseline reads only its own tables."""
    if baseline == "team_form":
        return _team_form(db)
    if baseline == "impact_bt":
        return _impact_baseline(db)
    raise ValueError(f"unknown pro_excess_baseline {baseline!r}; known: {PRO_EXCESS_BASELINES}")


def ordinal_or_none(value: str) -> int | None:
    """Parse a facts date; missing or malformed dates carry no history."""
    try:
        return datetime.fromisoformat(value).toordinal()
    except (ValueError, TypeError):
        return None
