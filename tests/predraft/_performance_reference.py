"""Row-wise reference for the production player-prefix parity checks."""

import sqlite3
from datetime import UTC, date, datetime

from league_predictor.decay import half_life_weight
from league_predictor.predraft.ratings.player_performance import (
    OPTIONAL_STATS,
    PerformanceAggregate,
    _as_of_datetime,
    _parse_match_datetime,
    assemble_performance_profile,
)
from league_predictor.predraft.ratings.stat_context.vocab import _AT10_SCALE

PLAYER_PERFORMANCE_COLUMNS = """
            gp.player_id, gp.champion_id, gp.result, gp.kills, gp.deaths, gp.assists,
            gp.teamkills, gp.teamdeaths, gp.damageshare, gp.killparticipation,
            gp.vspm, gp.controlwardsbought, gp.golddiffat15, gp.csdiffat15, gp.xpdiffat15,
            gp.golddiffat10, gp.csdiffat10, gp.xpdiffat10, gp.dpm, gp.cspm,
            gp.earnedgoldshare, gp.firstblood, gp.firsttower, gp.firstdragon, gp.firstbaron,
            g.duration_s AS gamelength, gp.wpm, gp.wcpm, gp.wardsplaced, gp.wardskilled,
            gp.damagetakenperminute, gp.damagemitigatedperminute,
            gp.firstbloodkill, gp.firstbloodassist,
            gp.killsat15, gp.assistsat15, gp.deathsat15,
            s.scheduled_ts AS match_date"""

PERFORMANCE_PROFILE_HALF_LIFE_DAYS = 60.0


def _as_cutoff_string(as_of_date: date | datetime) -> str:
    """Cutoff text in ``series.scheduled_ts`` shape: 'T'-separated, naive UTC.

    Returns:
        A space separator would sort below same-day 'T' timestamps, dropping that day.
    """
    if isinstance(as_of_date, datetime):
        if as_of_date.tzinfo is not None:
            as_of_date = as_of_date.astimezone(UTC).replace(tzinfo=None)
        return as_of_date.isoformat(sep="T")
    return as_of_date.isoformat()


def compute_player_performance_profile(
    conn: sqlite3.Connection,
    player_id: str,
    as_of_date: date,
    *,
    half_life: float = PERFORMANCE_PROFILE_HALF_LIFE_DAYS,
    tier: int | None = None,
) -> dict | None:
    """Recency-weighted performance profile for a player; sets ``conn.row_factory``.

    Args:
        tier: when set, restricts games to that tier's leagues.

    Returns:
        The ``compute_performance_profile_from_rows`` dict, or None with no game data.
    """
    conn.row_factory = sqlite3.Row
    tier_filter = ""
    params: list = [player_id, _as_cutoff_string(as_of_date)]
    if tier is not None:
        from league_predictor.predraft.ratings.player_scoring import leagues_for_tier

        tier_leagues = leagues_for_tier(tier)
        if tier_leagues:
            placeholders = ",".join("?" for _ in tier_leagues)
            tier_filter = f"AND s.league_id IN ({placeholders})"
            params.extend(tier_leagues)

    query = f"""
        SELECT {PLAYER_PERFORMANCE_COLUMNS}
        FROM game_players gp
        JOIN games g ON g.game_id = gp.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE gp.player_id = ?
          AND s.scheduled_ts < ?
          {tier_filter}
        ORDER BY date(s.scheduled_ts) DESC, s.series_id DESC, g.game_no DESC
    """
    rows = conn.execute(query, params).fetchall()
    return compute_performance_profile_from_rows(rows, as_of_date, half_life)


def _laning_terms(
    row: sqlite3.Row, w: float, *, has_at10: bool
) -> tuple[float, float, float, float, int, int]:
    """One game's weighted laning contribution: ``(weight, gd, cs, xp, full, fallback)``.

    Returns:
        All zeros when neither window is complete, so a game with no timeline data cannot
        dilute the pillar. The @10 window is read only when the row schema carries it, and
        is scaled onto the @15 scale.
    """
    gd15, cs15, xp15 = row["golddiffat15"], row["csdiffat15"], row["xpdiffat15"]
    if gd15 is not None and cs15 is not None and xp15 is not None:
        return w, w * float(gd15), w * float(cs15), w * float(xp15), 1, 0
    if has_at10:
        gd10, cs10, xp10 = row["golddiffat10"], row["csdiffat10"], row["xpdiffat10"]
        if gd10 is not None and cs10 is not None and xp10 is not None:
            return (
                w,
                w * float(gd10) * _AT10_SCALE,
                w * float(cs10) * _AT10_SCALE,
                w * float(xp10) * _AT10_SCALE,
                0,
                1,
            )
    return 0.0, 0.0, 0.0, 0.0, 0, 0


def compute_performance_profile_from_rows(  # noqa: PLR0915 - linear weighted stat sums
    rows: list[sqlite3.Row],
    as_of_date: date,
    half_life: float = PERFORMANCE_PROFILE_HALF_LIFE_DAYS,
) -> dict | None:
    """Recency-weighted performance profile from pre-fetched game rows; None when empty."""
    if not rows:
        return None

    total_weight = 0.0
    w_golddiff15 = 0.0
    w_csdiff15 = 0.0
    w_xpdiff15 = 0.0
    w_cspm = 0.0
    w_dpm = 0.0
    w_damageshare = 0.0
    w_earnedgoldshare = 0.0
    w_killparticipation = 0.0
    w_kda_numerator = 0.0
    w_deaths = 0.0
    w_vspm = 0.0
    w_controlwards = 0.0
    w_firstblood = 0.0
    w_firsttower = 0.0
    w_firstdragon = 0.0
    w_firstbaron = 0.0

    champion_games: dict[str, int] = {}
    champion_ranks: dict[str, int] = {}
    # data_quality is the BEST laning quality among a player's games; laning averages only over
    # games with @15/@10 data, so a few missing-timeline games cannot zero-dilute the pillar.
    full_laning_games = 0
    fallback_laning_games = 0
    laning_weight = 0.0
    # Partial-coverage stats carry their own weight: averaged only over non-NULL games, so a
    # league with 0% coverage leaves weight=0 → marked absent in the present dict.
    w_opt: dict[str, float] = dict.fromkeys(OPTIONAL_STATS, 0.0)
    weight_opt: dict[str, float] = dict.fromkeys(OPTIONAL_STATS, 0.0)
    as_of_dt = _as_of_datetime(as_of_date)

    # Detect schema once — rows from compute_player_performance_profile always include
    # at10 fallback columns, but external callers may not.
    sample_keys = set(rows[0].keys())
    has_at10 = {"golddiffat10", "csdiffat10", "xpdiffat10"} <= sample_keys
    has_opt = {s: s in sample_keys for s in OPTIONAL_STATS}

    def _val(v, default=0.0):
        return float(v) if v is not None else default

    for row_index, row in enumerate(rows):
        match_dt = _parse_match_datetime(row["match_date"])

        days_ago = max(0.0, (as_of_dt - match_dt).total_seconds() / 86400.0)
        w = half_life_weight(days_ago, half_life)
        total_weight += w

        lan_w, gd, cs, xp, full, fallback = _laning_terms(row, w, has_at10=has_at10)
        laning_weight += lan_w
        w_golddiff15 += gd
        w_csdiff15 += cs
        w_xpdiff15 += xp
        full_laning_games += full
        fallback_laning_games += fallback
        w_cspm += w * _val(row["cspm"])
        w_dpm += w * _val(row["dpm"])
        w_damageshare += w * _val(row["damageshare"])
        w_earnedgoldshare += w * _val(row["earnedgoldshare"])
        w_killparticipation += w * _val(row["killparticipation"])
        kills = _val(row["kills"])
        assists = _val(row["assists"])
        deaths = _val(row["deaths"])
        w_kda_numerator += w * (kills + assists)
        w_deaths += w * deaths
        w_vspm += w * _val(row["vspm"])
        w_controlwards += w * _val(row["controlwardsbought"])
        w_firstblood += w * _val(row["firstblood"])
        w_firsttower += w * _val(row["firsttower"])
        w_firstdragon += w * _val(row["firstdragon"])
        w_firstbaron += w * _val(row["firstbaron"])

        for stat in OPTIONAL_STATS:
            if has_opt[stat]:
                v = row[stat]
                if v is not None:
                    weight_opt[stat] += w
                    w_opt[stat] += w * float(v)

        champ = row["champion_id"] or "Unknown"
        champion_games[champ] = champion_games.get(champ, 0) + 1
        # Input rows are newest-first; a larger value means a more recent first sighting.
        champion_recency = len(rows) - row_index
        champion_ranks.setdefault(champ, champion_recency)

    if total_weight == 0:
        return None

    return assemble_performance_profile(
        PerformanceAggregate(
            total_weight=total_weight,
            laning_weight=laning_weight,
            sums={
                "cspm": w_cspm,
                "dpm": w_dpm,
                "damageshare": w_damageshare,
                "earnedgoldshare": w_earnedgoldshare,
                "killparticipation": w_killparticipation,
                "kda_numerator": w_kda_numerator,
                "deaths": w_deaths,
                "vspm": w_vspm,
                "controlwardsbought": w_controlwards,
                "firstblood": w_firstblood,
                "firsttower": w_firsttower,
                "firstdragon": w_firstdragon,
                "firstbaron": w_firstbaron,
            },
            laning_sums={
                "golddiffat15": w_golddiff15,
                "csdiffat15": w_csdiff15,
                "xpdiffat15": w_xpdiff15,
            },
            optional_sums=w_opt,
            optional_weights=weight_opt,
            champions={
                champion: (count, champion_ranks[champion])
                for champion, count in champion_games.items()
            },
            games=len(rows),
            full_laning_games=full_laning_games,
            fallback_laning_games=fallback_laning_games,
        )
    )
