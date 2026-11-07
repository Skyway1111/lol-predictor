"""Player scoring, grades, and position baselines."""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from league_predictor.predraft.league_tier import LEAGUE_TIER_DEFAULT, LEAGUE_TIER_MAP

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping

__all__ = [
    "PILLAR_STATS_BY_ROLE",
    "STAT_WEIGHTS_BY_ROLE",
    "apply_shrinkage",
    "compute_pillar_score",
    "compute_position_baselines",
    "leagues_for_tier",
    "performance_score_to_grade",
    "pillar_stats_for",
    "resolve_tournament_tier",
    "stat_weights_for",
    "z_to_performance_score",
]


# ── Constants ──

#: A summed stat weight below this is numerically zero — the pillar has no usable support.
_EPS = 1e-9

# Per-role pillars: stats selected by deconfounded point-biserial |r| > 0.07 with outcome;
# jng combat drops killparticipation (r=+0.01), sup vision is kept for role identity (r=+0.04).
PILLAR_STATS_BY_ROLE: dict[str, dict[str, list[str]]] = {
    "top": {
        "laning": ["golddiffat15", "csdiffat15", "xpdiffat15", "cspm"],
        "combat": ["dpm", "killparticipation", "kda", "firstblood"],
        "vision": ["vspm"],
    },
    "jng": {
        "laning": ["golddiffat15", "csdiffat15", "xpdiffat15", "cspm"],
        "combat": ["dpm", "kda", "firstblood"],
        "vision": ["vspm"],
    },
    "mid": {
        "laning": ["golddiffat15", "csdiffat15", "xpdiffat15", "cspm"],
        "combat": ["dpm", "killparticipation", "kda", "firstblood"],
        "vision": ["vspm"],
    },
    "bot": {
        "laning": ["golddiffat15", "csdiffat15", "xpdiffat15", "cspm"],
        "combat": ["dpm", "killparticipation", "kda", "firstblood"],
        "vision": ["vspm"],
    },
    "sup": {
        "lane": ["golddiffat15", "xpdiffat15"],
        "playmaking": ["kda", "killparticipation", "dpm", "firstblood"],
        "utility": ["earnedgoldshare"],
        "vision": ["vspm"],
    },
}

# Within-pillar stat weights: proportional to deconfounded |r| with game outcome, normalized
# to sum to 1.0 within each pillar. Source: all-tiers full-history correlation cut (892K games).
STAT_WEIGHTS_BY_ROLE: dict[str, dict[str, dict[str, float]]] = {
    "top": {
        "laning": {"golddiffat15": 0.33, "csdiffat15": 0.21, "xpdiffat15": 0.28, "cspm": 0.18},
        "combat": {"dpm": 0.32, "killparticipation": 0.15, "kda": 0.41, "firstblood": 0.12},
        "vision": {"vspm": 1.0},
    },
    "jng": {
        "laning": {"golddiffat15": 0.35, "csdiffat15": 0.18, "xpdiffat15": 0.27, "cspm": 0.20},
        "combat": {"dpm": 0.35, "kda": 0.44, "firstblood": 0.21},
        "vision": {"vspm": 1.0},
    },
    "mid": {
        "laning": {"golddiffat15": 0.36, "csdiffat15": 0.21, "xpdiffat15": 0.28, "cspm": 0.15},
        "combat": {"dpm": 0.33, "killparticipation": 0.14, "kda": 0.40, "firstblood": 0.13},
        "vision": {"vspm": 1.0},
    },
    "bot": {
        "laning": {"golddiffat15": 0.39, "csdiffat15": 0.18, "xpdiffat15": 0.32, "cspm": 0.11},
        "combat": {"dpm": 0.31, "killparticipation": 0.16, "kda": 0.40, "firstblood": 0.13},
        "vision": {"vspm": 1.0},
    },
    "sup": {
        "lane": {"golddiffat15": 0.62, "xpdiffat15": 0.38},
        "playmaking": {"kda": 0.39, "killparticipation": 0.18, "dpm": 0.24, "firstblood": 0.19},
        "utility": {"earnedgoldshare": 1.0},
        "vision": {"vspm": 1.0},
    },
}


def pillar_stats_for(role: str | None) -> dict[str, list[str]]:
    """Pillar -> stat-names map for a role. Unknown / None falls back to mid."""
    if role in PILLAR_STATS_BY_ROLE:
        return PILLAR_STATS_BY_ROLE[role]
    return PILLAR_STATS_BY_ROLE["mid"]


def stat_weights_for(role: str | None) -> dict[str, dict[str, float]]:
    """Pillar -> {stat: weight} map for a role. Falls back to mid."""
    if role in STAT_WEIGHTS_BY_ROLE:
        return STAT_WEIGHTS_BY_ROLE[role]
    return STAT_WEIGHTS_BY_ROLE.get("mid", {})


def compute_pillar_score(
    zscores: Mapping[str, float],
    pillar_stats: list[str],
    present_mask: Mapping[str, bool],
    stat_weights: dict[str, float] | None = None,
) -> tuple[float | None, str]:
    """Weighted mean z-score over present constituent stats, attenuated for partial data.

    Args:
        stat_weights: None means equal weights. Attenuation uses the weight fraction, not
            the count fraction, so missing a high-weight stat penalizes more.
    """
    present = [s for s in pillar_stats if present_mask.get(s, False)]
    if not present:
        return None, "dropped"

    if stat_weights is None:
        stat_weights = dict.fromkeys(pillar_stats, 1.0)

    w_present = sum(stat_weights.get(s, 1.0) for s in present)
    w_expected = sum(stat_weights.get(s, 1.0) for s in pillar_stats)

    if w_present < _EPS:
        return None, "dropped"

    score = sum(stat_weights.get(s, 1.0) * zscores.get(s, 0.0) for s in present) / w_present

    if w_present < w_expected - _EPS:
        score *= (w_present / w_expected) ** 0.5

    dq = "full" if len(present) == len(pillar_stats) else "partial"
    return score, dq


def leagues_for_tier(tier: int) -> list[str]:
    """Return league names that map to the given tier. International (None) included in Tier 1."""
    return [
        league for league, t in LEAGUE_TIER_MAP.items() if t == tier or (t is None and tier == 1)
    ]


def resolve_tournament_tier(tournament: str) -> int:
    """Map a tournament name to its tier. International (None) -> 1. Unknown -> default."""
    tier = LEAGUE_TIER_MAP.get(tournament)
    if tier is None:
        return 1 if tournament in LEAGUE_TIER_MAP else LEAGUE_TIER_DEFAULT
    return tier


# ── Team-snapshot helpers ──


# ── Player analytics ──

#: Descending display cuts: the first cut a value clears names its band.
_GRADE_CUTS: tuple[tuple[float, str], ...] = ((80.0, "S"), (65.0, "A"), (50.0, "B"), (35.0, "C"))


def apply_shrinkage(z: float, games_played: int, k: float = 20.0) -> float:
    """James-Stein shrinkage: z * (n / (n + k))."""
    return z * (games_played / (games_played + k))


def z_to_performance_score(z: float) -> float:
    """Map z-score to 0-100 scale: 50 + 10*z, clipped [1, 99]."""
    return max(1.0, min(99.0, 50.0 + 10.0 * z))


def performance_score_to_grade(score: float) -> str:
    """Map a 0-100 player performance score to S/A/B/C/D."""
    return next((grade for cut, grade in _GRADE_CUTS if score >= cut), "D")


# ── Position baselines ──


def _kda_baseline_per_player(
    conn: sqlite3.Connection,
    cutoff: str,
    as_of_date: date,
    *,
    tier: int | None,
    min_games: int,
) -> dict[str, tuple[float, float]]:
    """Per-role (mean, stdev) of each player's aggregate KDA = sum(k+a)/sum(deaths+eps)."""
    tier_filter, tier_params = _tier_clause(tier)
    params = [cutoff, as_of_date.isoformat(), *tier_params, min_games]
    rows = conn.execute(
        f"""
        SELECT role,
               AVG(p_kda) AS mean_kda,
               AVG(p_kda * p_kda) AS sq_kda
        FROM (
            SELECT pg.role AS role,
                   SUM(COALESCE(pg.kills, 0) + COALESCE(pg.assists, 0)) * 1.0
                       / (SUM(COALESCE(pg.deaths, 0)) + 0.001) AS p_kda
            FROM game_players pg
            JOIN games g ON g.game_id = pg.game_id
            JOIN series s ON s.series_id = g.series_id
            WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) < ?
              AND pg.role IS NOT NULL {tier_filter}
            GROUP BY pg.role, pg.player_id
            HAVING COUNT(*) >= ?
        )
        GROUP BY role
        """,
        params,
    ).fetchall()
    out: dict[str, tuple[float, float]] = {}
    for r in rows:
        m = float(r["mean_kda"])
        var = max(0.0, float(r["sq_kda"]) - m * m)
        out[r["role"]] = (m, var**0.5)
    return out


def compute_position_baselines(
    conn: sqlite3.Connection,
    as_of_date: date,
    *,
    trailing_days: int = 180,
    min_games: int = 5,
    tier: int | None = None,
) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    """Per-position mean and stdev for each pillar stat over trailing window."""
    cutoff = (as_of_date - timedelta(days=trailing_days)).isoformat()

    tier_filter, tier_params = _tier_clause(tier)
    params = [cutoff, as_of_date.isoformat(), *tier_params, min_games]

    # Partial-coverage stats use bare AVG (NULL-tolerant) like the diff stats, so a league
    # with 0% coverage on a column cannot collapse the major-league baseline toward zero.
    _NULL_TOL_STATS = (
        "golddiffat15",
        "csdiffat15",
        "xpdiffat15",
        "wpm",
        "wcpm",
        "damagetakenperminute",
        "damagemitigatedperminute",
        "firstbloodkill",
        "firstbloodassist",
        "firstblood",
        "killsat15",
        "assistsat15",
        "deathsat15",
    )
    _COALESCED_STATS = (
        "dpm",
        "damageshare",
        "killparticipation",
        "vspm",
        "controlwardsbought",
        "cspm",
        "earnedgoldshare",
    )

    null_tol_sql = ",\n               ".join(
        f"AVG(pg.{s}) AS mean_{s}, AVG(pg.{s} * pg.{s}) AS sq_{s}" for s in _NULL_TOL_STATS
    )
    coalesced_sql = ",\n               ".join(
        f"AVG(COALESCE(pg.{s}, 0.0)) AS mean_{s}, "
        f"AVG(COALESCE(pg.{s}, 0.0) * COALESCE(pg.{s}, 0.0)) AS sq_{s}"
        for s in _COALESCED_STATS
    )

    rows = conn.execute(
        f"""
        SELECT pg.role,
               {null_tol_sql},
               {coalesced_sql}
        FROM game_players pg
        JOIN games g ON g.game_id = pg.game_id
        JOIN series s ON s.series_id = g.series_id
        WHERE date(s.scheduled_ts) >= ? AND date(s.scheduled_ts) < ?
          AND pg.role IS NOT NULL
          {tier_filter}
        GROUP BY pg.role
        HAVING COUNT(*) >= ?
        """,
        params,
    ).fetchall()

    stat_names = [*_NULL_TOL_STATS, *_COALESCED_STATS]
    baselines: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for row in rows:
        role = row["role"]
        means: dict[str, float] = {}
        stdevs: dict[str, float] = {}
        for stat in stat_names:
            m = float(row[f"mean_{stat}"] or 0.0)  # None when a role has no real diffs
            sq = float(row[f"sq_{stat}"] or 0.0)
            variance = max(0.0, sq - m * m)
            means[stat] = m
            stdevs[stat] = max(variance**0.5, 1e-6)
        baselines[role] = (means, stdevs)

    # KDA must be a per-PLAYER aggregate (sum(k+a)/sum(deaths)); a per-game average explodes
    # on zero-death games and makes KDA a dead z-score feature for everyone.
    for role, (kmean, ksd) in _kda_baseline_per_player(
        conn,
        cutoff,
        as_of_date,
        tier=tier,
        min_games=min_games,
    ).items():
        if role in baselines:
            baselines[role][0]["kda"] = kmean
            baselines[role][1]["kda"] = max(ksd, 1e-6)
    return baselines


def _tier_clause(tier: int | None) -> tuple[str, list[str]]:
    leagues = leagues_for_tier(tier) if tier is not None else []
    return (
        (f"AND s.league_id IN ({','.join('?' for _ in leagues)})", leagues) if leagues else ("", [])
    )
