"""Reusable full-history facts corpus for player Impact fits."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER

if TYPE_CHECKING:
    import sqlite3


@dataclass(frozen=True, slots=True)
class ImpactGame:
    """One Impact-eligible game reduced to its lineups, outcome, and age."""

    team1_players: tuple[str, ...]
    team2_players: tuple[str, ...]
    team1_won: int
    days_before_asof: float
    #: +1 when team1 played blue, -1 when team2 did, 0 when the facts name no blue side.
    blue_sign: int


@dataclass(frozen=True, slots=True)
class ImpactCorpus:
    """Factorized full-history games plus compact appearance-role arrays."""

    player_ids: tuple[str, ...]
    role_values: tuple[str, ...]
    series_ids: tuple[str, ...]
    game_nos: np.ndarray
    team_a: tuple[str, ...]
    team_b: tuple[str, ...]
    winners: tuple[str | None, ...]
    blue_teams: tuple[str | None, ...]
    scheduled_instants: np.ndarray
    scheduled_dates: np.ndarray
    raw_dates: np.ndarray
    league_ids: tuple[str | None, ...]
    lineups: np.ndarray
    lineup_counts: np.ndarray
    appearance_games: np.ndarray
    appearance_players: np.ndarray
    appearance_roles: np.ndarray
    source_appearances: int
    rejected_appearances: int

    @property
    def n_appearances(self) -> int:
        """Number of canonical-side appearances eligible for Impact influence."""
        return int(self.appearance_players.size)


def blue_sign(blue_team: str | None, team_a: str, team_b: str) -> int:
    """+1 when ``team_a`` played blue, -1 when ``team_b`` did, 0 when the facts name neither."""
    if blue_team == team_a:
        return 1
    if blue_team == team_b:
        return -1
    return 0


def _readonly(values: list[int] | list[tuple[int, int]], dtype) -> np.ndarray:
    """Return a compact immutable ndarray."""
    out = np.asarray(values, dtype=dtype)
    out.flags.writeable = False
    return out


_ROLE_RANK = {role: rank for rank, role in enumerate(ROLE_ORDER)}


def _scheduled_instant(value: str | None, *, game_id: str) -> int:
    """Parse one locked timestamp to UTC microseconds or fail with game context."""
    try:
        parsed = datetime.fromisoformat(value) if value is not None else None
    except ValueError as exc:
        raise RuntimeError(f"game {game_id!r} has invalid scheduled_ts {value!r}") from exc
    if parsed is None:
        raise RuntimeError(f"game {game_id!r} has invalid scheduled_ts {value!r}")
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return int(parsed.astimezone(UTC).timestamp() * 1_000_000)


def _dominant_role(counts: Counter[str]) -> str:
    """Resolve role frequency ties with the locked semantic role order."""
    highest = max(counts.values())
    tied = (role for role, count in counts.items() if count == highest)
    return min(tied, key=lambda role: _ROLE_RANK[role])


def _intern(key: str, index: dict[str, int], values: list[str]) -> int:
    """The index of ``key``, appending it to ``values`` on first sight."""
    position = index.get(key)
    if position is None:
        position = len(values)
        index[key] = position
        values.append(key)
    return position


def _flush_lineup(
    game_id,
    side_players: tuple[list[int], list[int]],
    lineup_rows,
    lineup_count_rows: list[tuple[int, int]],
) -> None:
    """Pack one game's side rosters into a fixed ``(2, 5)`` row; a no-op before the first game."""
    if game_id is None:
        return
    counts = (len(side_players[0]), len(side_players[1]))
    if max(counts) > N_ROLES:
        raise RuntimeError(
            f"game {game_id!r} has lineup width {max(counts)}; locked maximum {N_ROLES}"
        )
    packed = np.full((2, 5), -1, dtype=np.int32)
    for side in (0, 1):
        packed[side, : counts[side]] = side_players[side]
    lineup_rows.append(packed)
    lineup_count_rows.append(counts)


def build_impact_corpus(  # noqa: PLR0915 - linear column accumulation, one chronological pass
    conn: sqlite3.Connection,
) -> ImpactCorpus:
    """Stream and factorize the complete joined Impact appearance history once."""
    rows = conn.execute(
        """
        SELECT gp.game_id, s.series_id, g.game_no, gp.player_id, gp.team_id, gp.role,
               s.team_a, s.team_b, g.winner, g.blue_team, s.scheduled_ts,
               date(s.scheduled_ts), s.league_id
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        ORDER BY datetime(s.scheduled_ts), s.series_id, g.game_no, gp.player_id
        """
    )

    player_ids: list[str] = []
    player_index: dict[str, int] = {}
    role_values: list[str] = []
    role_index: dict[str, int] = {}
    series_ids: list[str] = []
    game_nos: list[int] = []
    team_a_values: list[str] = []
    team_b_values: list[str] = []
    winners: list[str | None] = []
    blue_teams: list[str | None] = []
    scheduled_instants: list[int] = []
    scheduled_dates: list[int] = []
    raw_dates: list[int] = []
    league_ids: list[str | None] = []
    lineup_rows: list[np.ndarray] = []
    lineup_count_rows: list[tuple[int, int]] = []
    appearance_games: list[int] = []
    appearance_players: list[int] = []
    appearance_roles: list[int] = []
    source_appearances = 0
    rejected_appearances = 0

    current_game: str | None = None
    current_index = -1
    side_players: tuple[list[int], list[int]] = ([], [])

    for row in rows:
        (
            game_id,
            series_id,
            game_no,
            player_id,
            team_id,
            role,
            team_a,
            team_b,
            winner,
            blue_team,
            scheduled_ts,
            scheduled_date,
            league_id,
        ) = row
        source_appearances += 1
        if game_id != current_game:
            _flush_lineup(current_game, side_players, lineup_rows, lineup_count_rows)
            current_game = game_id
            current_index += 1
            side_players = ([], [])
            series_ids.append(series_id)
            game_nos.append(game_no)
            team_a_values.append(team_a)
            team_b_values.append(team_b)
            winners.append(winner)
            blue_teams.append(blue_team)
            scheduled_instants.append(_scheduled_instant(scheduled_ts, game_id=game_id))
            league_ids.append(league_id)
            try:
                scheduled_dates.append(date.fromisoformat(scheduled_date).toordinal())
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"game {game_id!r} has invalid calendar date from scheduled_ts {scheduled_ts!r}"
                ) from exc
            try:
                raw_dates.append(date.fromisoformat(scheduled_ts[:10]).toordinal())
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"game {game_id!r} has invalid scheduled_ts {scheduled_ts!r}"
                ) from exc

        if team_id == team_a:
            side = 0
        elif team_id == team_b:
            side = 1
        else:
            rejected_appearances += 1
            continue

        pidx = _intern(player_id, player_index, player_ids)
        ridx = -1
        if role:
            if role not in _ROLE_RANK:
                raise RuntimeError(
                    f"game {game_id!r} player {player_id!r} has unsupported role {role!r}"
                )
            ridx = _intern(role, role_index, role_values)
        appearance_games.append(current_index)
        appearance_players.append(pidx)
        appearance_roles.append(ridx)
        side_players[side].append(pidx)

    _flush_lineup(current_game, side_players, lineup_rows, lineup_count_rows)
    lineups = np.stack(lineup_rows) if lineup_rows else np.empty((0, 2, 5), dtype=np.int32)
    lineups.flags.writeable = False
    return ImpactCorpus(
        player_ids=tuple(player_ids),
        role_values=tuple(role_values),
        series_ids=tuple(series_ids),
        game_nos=_readonly(game_nos, np.int16),
        team_a=tuple(team_a_values),
        team_b=tuple(team_b_values),
        winners=tuple(winners),
        blue_teams=tuple(blue_teams),
        scheduled_instants=_readonly(scheduled_instants, np.int64),
        scheduled_dates=_readonly(scheduled_dates, np.int32),
        raw_dates=_readonly(raw_dates, np.int32),
        league_ids=tuple(league_ids),
        lineups=lineups,
        lineup_counts=_readonly(lineup_count_rows, np.uint8),
        appearance_games=_readonly(appearance_games, np.int32),
        appearance_players=_readonly(appearance_players, np.int32),
        appearance_roles=_readonly(appearance_roles, np.int16),
        source_appearances=source_appearances,
        rejected_appearances=rejected_appearances,
    )


def prepare_impact_window(
    corpus: ImpactCorpus,
    *,
    as_of_date: date,
    window_months: int | None,
) -> tuple[list[ImpactGame], dict[str, str]]:
    """Prepare a legacy-equivalent trailing window, or all prior games when unbounded."""
    window_start = (
        0
        if window_months is None
        else (as_of_date - timedelta(days=window_months * 30)).toordinal()
    )
    cutoff = as_of_date.toordinal()
    selected = (corpus.scheduled_dates >= window_start) & (corpus.scheduled_dates < cutoff)
    role_counts: dict[str, Counter[str]] = {}
    for game_idx, player_idx, role_idx in zip(
        corpus.appearance_games,
        corpus.appearance_players,
        corpus.appearance_roles,
        strict=True,
    ):
        if not selected[int(game_idx)] or role_idx < 0:
            continue
        player_id = corpus.player_ids[int(player_idx)]
        role = corpus.role_values[int(role_idx)]
        role_counts.setdefault(player_id, Counter())[role] += 1

    games: list[ImpactGame] = []
    for game_idx in np.flatnonzero(selected):
        count_a, count_b = (int(v) for v in corpus.lineup_counts[game_idx])
        winner = corpus.winners[game_idx]
        if (
            not count_a
            or not count_b
            or winner not in (corpus.team_a[game_idx], corpus.team_b[game_idx])
        ):
            continue
        raw_date = int(corpus.raw_dates[game_idx])
        if raw_date < 0:
            continue
        team1 = tuple(corpus.player_ids[int(idx)] for idx in corpus.lineups[game_idx, 0, :count_a])
        team2 = tuple(corpus.player_ids[int(idx)] for idx in corpus.lineups[game_idx, 1, :count_b])
        games.append(
            ImpactGame(
                team1_players=team1,
                team2_players=team2,
                team1_won=1 if winner == corpus.team_a[game_idx] else 0,
                days_before_asof=float(max(0, cutoff - raw_date)),
                blue_sign=blue_sign(
                    corpus.blue_teams[game_idx],
                    corpus.team_a[game_idx],
                    corpus.team_b[game_idx],
                ),
            )
        )

    player_roles = {player_id: _dominant_role(counts) for player_id, counts in role_counts.items()}
    return games, player_roles
