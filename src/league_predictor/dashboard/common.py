"""Shared helpers for the FastAPI dashboard service."""

from __future__ import annotations

import dataclasses
import datetime
import typing

from league_predictor import clock
from league_predictor.predraft.league_tier import INTERNATIONAL_LEAGUES, MAJOR_LEAGUES

if typing.TYPE_CHECKING:
    import sqlite3

# The dashboard's "major only" lens: domestic majors PLUS internationals (MSI/Worlds/
# FST/EWC), which the training-population MAJOR_LEAGUES set deliberately excludes.
DASHBOARD_MAJOR_LEAGUES: tuple[str, ...] = (*MAJOR_LEAGUES, *INTERNATIONAL_LEAGUES)


def default_date_range() -> tuple[str, str]:
    """Year-to-date window from Jan 1 of the current year through today (ISO)."""
    today = clock.utc_today()
    return f"{today.year}-01-01", today.isoformat()


@dataclasses.dataclass(frozen=True)
class FilterScope:
    """Resolved list/aggregate filters over the ``series`` facts.

    Attributes:
        sd: Resolved ISO start day, inclusive.
        ed: Resolved ISO end day, inclusive; also the as-of bound for Impact joins.
        clauses: WHERE fragments to AND together, assuming ``series`` joined under the alias.
        params: Bind values positionally matching ``clauses``.
    """

    sd: str
    ed: str
    clauses: list[str]
    params: list[object]

    def where(self, *extra: str) -> str:
        """AND-join the scope clauses with any caller-supplied extra clauses."""
        return " AND ".join([*self.clauses, *extra])


def parse_league_csv(leagues: str | None) -> list[str]:
    """Split the shared ``leagues`` multiselect CSV into trimmed, non-empty ids."""
    return [x.strip() for x in (leagues or "").split(",") if x.strip()]


# sightline-ok: 14 - FilterScope is the clump's home; the callers are FastAPI query params
def parse_filters(
    start_date: str | None,
    end_date: str | None,
    leagues: str | None,
    *,
    major_only: bool,
    series_alias: str = "s",
) -> FilterScope:
    """Turn the shared filter params into WHERE fragments scoped on a series row.

    Args:
        start_date: ISO day, inclusive; defaults to Jan 1 of the current year.
        end_date: ISO day, inclusive; defaults to today.
        leagues: CSV of league ids; non-empty wins and silences ``major_only``.
        major_only: Restrict to the dashboard major set only when ``leagues`` is empty.
        series_alias: SQL alias the caller joined ``series`` under.
    """
    default_sd, default_ed = default_date_range()
    sd, ed = start_date or default_sd, end_date or default_ed
    day = f"date({series_alias}.scheduled_ts)"
    clauses = [f"{day} >= ?", f"{day} <= ?"]
    params: list[object] = [sd, ed]
    league_ids = parse_league_csv(leagues)
    if league_ids:
        placeholders = ",".join("?" * len(league_ids))
        clauses.append(f"{series_alias}.league_id IN ({placeholders})")
        params.extend(league_ids)
    elif major_only:
        placeholders = ",".join("?" * len(DASHBOARD_MAJOR_LEAGUES))
        clauses.append(f"{series_alias}.league_id IN ({placeholders})")
        params.extend(DASHBOARD_MAJOR_LEAGUES)
    return FilterScope(sd=sd, ed=ed, clauses=clauses, params=params)


# Display-name suffixes marking sub-rosters excluded from the main team/player lists.
ACADEMY_DISPLAY_NAME_SUFFIXES: tuple[str, ...] = (
    "Academy",
    "Challengers",
    "Rookies",
    "Youth",
    "Trainees",
    "Scholars",
)


def academy_exclusion_clauses(column: str) -> list[str]:
    """Return `<col> NOT LIKE '%X%'` clauses for every academy suffix."""
    return [f"{column} NOT LIKE '%{suffix}%'" for suffix in ACADEMY_DISPLAY_NAME_SUFFIXES]


def major_region_clauses(region_col: str, name_col: str) -> tuple[list[str], list[object]]:
    """Return (clauses, params) restricting a list to the major-league regions."""
    placeholders = ",".join("?" * len(MAJOR_LEAGUES))
    clauses = [f"{region_col} IN ({placeholders})", *academy_exclusion_clauses(name_col)]
    return clauses, list(MAJOR_LEAGUES)


# Whitelist of tables latest_snapshot_date may read. The name is interpolated into SQL,
# so a table reaching it from a route must be one of these.
_ALLOWED_SNAPSHOT_TABLES: frozenset[str] = frozenset(
    {"team_daily_snapshot", "player_daily_snapshot", "power_rating_snapshots"}
)


def latest_snapshot_date(
    conn: sqlite3.Connection,
    table: str,
    *,
    end_date: str | None = None,
) -> str | None:
    """Latest `date` in `table`, optionally bounded by `end_date`."""
    if table not in _ALLOWED_SNAPSHOT_TABLES:
        raise ValueError(f"Invalid snapshot table: {table}")
    if end_date:
        row = conn.execute(
            f"SELECT MAX(date) AS max_date FROM {table} WHERE date <= ?",
            (end_date,),
        ).fetchone()
    else:
        row = conn.execute(f"SELECT MAX(date) AS max_date FROM {table}").fetchone()
    return str(row["max_date"]) if row and row["max_date"] else None


def _league_in_clause(column: str, league_ids: list[str] | None) -> tuple[str, list[object]]:
    """An optional ``AND <column> IN (...)`` fragment for a league restriction."""
    if not league_ids:
        return "", []
    placeholders = ",".join("?" * len(league_ids))
    return f" AND {column} IN ({placeholders})", list(league_ids)


# Rankings-population gate: a team is "currently active" if it played within this many
# days of the snapshot date shown — wide enough to span the LCK/LEC offseason.
TEAM_ACTIVITY_WINDOW_DAYS = 180


def active_team_ids_sql(sd: str, ed: str, league_ids: list[str] | None = None) -> tuple[str, list]:
    """Return (sub-query, params) for team_ids with played games in [sd, ed].

    Args:
        league_ids: Restrict to series in these leagues; the daily snapshots carry no league.
    """
    clause, params = _league_in_clause("league_id", league_ids)
    # Played games only: scheduled-but-gameless series rows (upcoming-sync
    # residue) must never count as activity.
    half = (
        f"WHERE date(scheduled_ts) >= ? AND date(scheduled_ts) <= ?{clause} "
        "AND EXISTS (SELECT 1 FROM games g WHERE g.series_id = series.series_id)"
    )
    sql = f"""(
        SELECT team_a AS team_id FROM series {half}
        UNION
        SELECT team_b FROM series {half}
    )"""
    return sql, [sd, ed, *params, sd, ed, *params]


def active_player_ids_sql(
    sd: str, ed: str, league_ids: list[str] | None = None
) -> tuple[str, list]:
    """Return (sub-query, params) for player_ids who themselves played in [sd, ed]."""
    # Raw ISO bounds, ed exclusive at the next midnight, so idx_series_scheduled applies:
    # a date() wrap is non-sargable and forces a full game_players scan (~30x slower).
    ed_next = (datetime.date.fromisoformat(ed[:10]) + datetime.timedelta(days=1)).isoformat()
    clause, params = _league_in_clause("s.league_id", league_ids)
    sql = f"""(
        SELECT DISTINCT gp.player_id
        FROM series s
        JOIN games g ON g.series_id = s.series_id
        JOIN game_players gp ON gp.game_id = g.game_id
        WHERE s.scheduled_ts >= ? AND s.scheduled_ts < ?{clause}
    )"""
    return sql, [sd, ed_next, *params]
