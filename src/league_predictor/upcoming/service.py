"""Upcoming-pipeline verbs: sync, predict, pair; writes upcoming_matches, never predictions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from league_predictor import artifacts
from league_predictor.asof import AsOf, MatchContext
from league_predictor.upcoming.matching import league_to_oe, resolve_teams
from league_predictor.upcoming.pandascore import UPCOMING_HORIZON_DAYS, PandaMatch

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Mapping

    #: What ``predict_unfinished`` needs from its loader: a per-map P(team_a) scorer at a
    #: query point (its context carries the match's format, stage and score) + a model id.
    ServedPredictor = tuple[Callable[[str, str, AsOf], float], str]

_PAIR_WINDOW_DAYS = 1.5  # max |scheduled gap| between a PandaScore match and its OE series

#: Maps that decide a series, by format; a row whose score reaches it is finished whatever the
#: feed still says (a finished match leaves the running feed before the ingest pairs it).
_WINS_NEEDED = {1: 1, 2: 2, 3: 2, 5: 3}


@dataclass(frozen=True)
class SyncReport:
    """One schedule sync: rows written, feed-dropped rows canceled, teams needing a hand map."""

    upserted: int
    canceled: int
    unmatched: list[str]


def sync_schedule(
    conn: sqlite3.Connection,
    matches: list[PandaMatch],
    *,
    now: datetime,
) -> SyncReport:
    """Upsert the fetched schedule into ``upcoming_matches`` and cancel rows the feed dropped."""
    now_iso = now.isoformat()
    resolved = resolve_teams(conn, (t for m in matches for t in m.teams), now=now_iso)
    for match in matches:
        _upsert_match(conn, match, resolved, now_iso)
    # Lower bound now+1h: a match dropped just before start is starting, not canceled. Only
    # the fetch horizon is scanned, and an empty feed yields NOT IN (NULL), cancelling nothing.
    canceled = conn.execute(
        "UPDATE upcoming_matches SET ps_status = 'canceled'"
        " WHERE ps_status = 'not_started' AND scheduled_ts BETWEEN ? AND ?"
        f" AND ps_match_id NOT IN ({','.join('?' * len(matches)) or 'NULL'})",
        [
            (now + timedelta(hours=1)).isoformat(),
            (now + timedelta(days=UPCOMING_HORIZON_DAYS)).isoformat(),
        ]
        + [m.ps_match_id for m in matches],
    ).rowcount
    unmatched = sorted(
        {
            f"{team.name} (ps_team_id {team.ps_team_id})"
            for match in matches
            for team in match.teams
            if resolved[team.ps_team_id] is None
        }
    )
    return SyncReport(upserted=len(matches), canceled=canceled, unmatched=unmatched)


def _upsert_match(
    conn: sqlite3.Connection,
    match: PandaMatch,
    resolved: Mapping[int, str | None],
    now_iso: str,
) -> None:
    """Write one match row, preserving its prediction unless the matchup changed."""
    ids = [resolved[t.ps_team_id] for t in match.teams]
    score_a = score_b = None
    if len(ids) == 2 and all(ids):  # noqa: PLR2004 - a match has exactly two teams
        if ids[0] == ids[1]:  # corrupt identity data — every run would hit it
            raise ValueError(
                f"PandaScore teams {[t.ps_team_id for t in match.teams]} both map to"
                f" {ids[0]!r} — fix one with `lol-predict upcoming-map-team"
                " <ps_team_id> <team_id>`"
            )
        wins = dict(match.wins)
        names = (t.name for t in match.teams)
        maps_won = (wins.get(t.ps_team_id, 0) for t in match.teams)
        (team_a, a_name, score_a), (team_b, b_name, score_b) = sorted(
            zip(ids, names, maps_won, strict=True)  # type: ignore[arg-type]
        )
    else:
        team_a = team_b = None
        a_name = match.teams[0].name if match.teams else None
        b_name = match.teams[1].name if len(match.teams) > 1 else None
    league_id = league_to_oe(conn, match.league_name)
    status = match.status
    needed = _WINS_NEEDED.get(match.best_of or 1, 1)
    if score_a is not None and max(score_a, score_b or 0) >= needed:
        status = "finished"  # decided by its games, whatever the feed still says

    values = {
        "ps_match_id": match.ps_match_id,
        "league_name": match.league_name,
        "league_id": league_id,
        "best_of": match.best_of,
        "stage": match.tournament_name,
        "scheduled_ts": match.scheduled_ts,
        "ps_status": status,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_name": a_name,
        "team_b_name": b_name,
        "score_a": score_a,
        "score_b": score_b,
        "fetched_at": now_iso,
    }
    updates = [f"{key} = excluded.{key}" for key in values if key != "ps_match_id"]
    same_teams = "team_a IS excluded.team_a AND team_b IS excluded.team_b"
    updates.extend(
        f"{key} = CASE WHEN {same_teams} THEN {key} END"
        for key in ("p_a", "p_a_map", "model_id", "predicted_at", "series_id")
    )
    conn.execute(
        f"INSERT INTO upcoming_matches ({', '.join(values)}) "
        f"VALUES ({', '.join('?' for _ in values)}) "
        f"ON CONFLICT(ps_match_id) DO UPDATE SET {', '.join(updates)}",
        tuple(values.values()),
    )


def pair_realized(conn: sqlite3.Connection, *, now: datetime) -> int:
    """Link matched rows to their ingested OE series: nearest scheduled time in the window."""
    # Each series pairs once, and candidates run oldest first: a same-pair rematch tomorrow must
    # never steal yesterday's series.
    rows = conn.execute(
        "SELECT ps_match_id, team_a, team_b, scheduled_ts FROM upcoming_matches"
        " WHERE series_id IS NULL AND team_a IS NOT NULL"
        "   AND ps_status != 'canceled' AND scheduled_ts <= ?"
        " ORDER BY scheduled_ts",
        (now.isoformat(),),
    ).fetchall()
    paired = 0
    for row in rows:
        series = conn.execute(
            "SELECT s.series_id FROM series s"
            " WHERE s.team_a = ? AND s.team_b = ?"
            "   AND abs(julianday(s.scheduled_ts) - julianday(?)) <= ?"
            "   AND s.series_id NOT IN"
            "     (SELECT series_id FROM upcoming_matches WHERE series_id IS NOT NULL)"
            " ORDER BY abs(julianday(s.scheduled_ts) - julianday(?)) LIMIT 1",
            (
                row["team_a"],
                row["team_b"],
                row["scheduled_ts"],
                _PAIR_WINDOW_DAYS,
                row["scheduled_ts"],
            ),
        ).fetchone()
        if series is None:
            continue
        conn.execute(
            "UPDATE upcoming_matches SET series_id = ? WHERE ps_match_id = ?",
            (series["series_id"], row["ps_match_id"]),
        )
        paired += 1
    return paired


def predict_unfinished(
    conn: sqlite3.Connection,
    *,
    now: datetime,
    load_predictor: Callable[[], ServedPredictor],
) -> int:
    """(Re)score every matched match that is not yet started, or running, with the served model.

    Returns:
        The number of rows scored. A running match is priced from its state: ``p_a_map`` is
        the next map at the current score, ``p_a`` the series from that state. A decided or
        paired row is frozen.
    """
    # ``load_predictor`` is called at most once, and not at all when nothing needs scoring.
    from league_predictor.predraft.pipeline.serve import series_prob

    now_iso = now.isoformat()
    candidates = conn.execute(
        "SELECT ps_match_id, team_a, team_b, best_of, stage, score_a, score_b, ps_status"
        " FROM upcoming_matches"
        " WHERE team_a IS NOT NULL AND series_id IS NULL AND ("
        "   ps_status = 'running' OR (ps_status = 'not_started' AND scheduled_ts > ?))",
        (now_iso,),
    ).fetchall()
    if not candidates:
        return 0
    predict_fn, model_id = load_predictor()
    for row in candidates:
        match_format = f"Bo{row['best_of']}" if row["best_of"] else None
        state = (int(row["score_a"] or 0), int(row["score_b"] or 0))
        context = MatchContext.of(
            match_format or "Bo1",
            str(row["stage"] or "regular"),
            {row["team_a"]: state[0], row["team_b"]: state[1]},
        )
        p_map = float(predict_fn(row["team_a"], row["team_b"], AsOf.of(now).with_context(context)))
        conn.execute(
            "UPDATE upcoming_matches SET p_a = ?, p_a_map = ?, model_id = ?, predicted_at = ?"
            " WHERE ps_match_id = ?",
            (
                series_prob(p_map, match_format, state=state),
                p_map,
                model_id,
                now_iso,
                row["ps_match_id"],
            ),
        )
    return len(candidates)


def live_served_predictor(conn: sqlite3.Connection) -> ServedPredictor:
    """The real ``load_predictor``: the served model over live as-of feature indexes."""
    # COST: the index build takes MINUTES of CPU and may materialize missing team-snapshot
    # cache rows onto ``conn``'s database. Registers the served id into ``models``.
    from league_predictor.lifecycle.predictions import register
    from league_predictor.predraft.pipeline import serve
    from league_predictor.predraft.pipeline.live import load_live_oracle

    model_id = register(conn, "served")
    model, indexes = load_live_oracle(conn, artifacts.primary_path("served"))

    def predict_fn(team_a: str, team_b: str, as_of: AsOf) -> float:
        return float(serve.predict(model, indexes, team_a, team_b, as_of=as_of))

    return predict_fn, model_id
