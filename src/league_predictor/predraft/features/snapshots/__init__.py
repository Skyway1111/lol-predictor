"""Pre-draft feature orchestration: two team snapshots and a pair into one flat row."""

import sqlite3
from collections.abc import Mapping
from datetime import date, datetime

from league_predictor.asof import AsOf
from league_predictor.predraft.features.manifest import (
    dead_weight,
    diff_keys,
    diff_only_bases,
)
from league_predictor.predraft.features.snapshots.stream import SnapshotStore
from league_predictor.predraft.league_tier import (
    LEAGUE_TIER_DEFAULT,
    LEAGUE_TIER_MAP,
)

__all__ = [
    "SnapshotStore",
    "build_pre_draft_features",
]

# Caps diff_kda outliers from broken rows (raw min ~-20000). A monotone transform
# would be a no-op on tree splits; winsorization genuinely collapses the bad-tail ranking.
_KDA_WINSOR_CAP = 3.0

# Feature membership (diff bases, dropped sides, dead weight) is
# declared once in features/manifest.py; the prune-key set derives from it at module load.
_DIFF_ONLY_PRUNE_KEYS = frozenset(
    f"{prefix}{base}" for prefix in ("t1_", "t2_") for base in diff_only_bases()
)


# --- Pruning rules ---
def _drop_diff_only_redundant(features: Mapping) -> dict:
    """Prune pass 1: drop the t1_/t2_ sides of the manifest's ``diff_only_bases``."""
    return {k: v for k, v in features.items() if k not in _DIFF_ONLY_PRUNE_KEYS}


def _drop_low_importance(features: Mapping) -> dict:
    """Prune pass 2: drop the manifest's ``dead_weight`` columns (<3% combined importance)."""
    dead = dead_weight()
    return {k: v for k, v in features.items() if k not in dead}


# Snapshot groups whose keys copy through verbatim under the t1_/t2_ prefixes. The first set is
# always present; the extended set is optional per snapshot.
_CORE_GROUPS = ("momentum", "objective", "pace", "variance")
_EXTENDED_GROUPS = (
    "schedule",
    "roster_stability",
    "upset",
    "side",
    "win_margin",
    "champ_pool",
    "game_phase",
    "ema_wr",
    "bayesian_wr",
    "draft",
)


def _add_snapshot_groups(features: dict, t1_snapshot: Mapping, t2_snapshot: Mapping) -> None:
    """Copy both snapshots' per-team feature groups into ``features`` under their side prefix."""
    sides = (("t1_", t1_snapshot), ("t2_", t2_snapshot))
    for group in _CORE_GROUPS:
        for prefix, snap in sides:
            for k, v in snap[group].items():
                features[f"{prefix}{k}"] = v
    for prefix, snap in sides:
        for group in _EXTENDED_GROUPS:
            for k, v in snap.get(group, {}).items():
                features[f"{prefix}{k}"] = v


# --- Orchestrator ---
def build_pre_draft_features(
    conn: sqlite3.Connection,
    team1_id: str,
    team2_id: str,
    match_format: str,
    stage: str,
    as_of: AsOf | date | datetime,
    *,
    series_wins: tuple[int, int] = (0, 0),
    prune: bool = True,
    store: SnapshotStore | None = None,
) -> dict[str, float]:
    """Build a flat feature dict for the pre-draft (team strength) model.

    Args:
        match_format: 'Bo1' / 'Bo3' / 'Bo5'.
        stage: Tournament stage (e.g. 'group', 'playoff').
        as_of: All reads are walk-forward relative to this query; a bare date is the start
            of that day.
        series_wins: Maps ``(team1, team2)`` have already won in this series.
        prune: Run both prune passes. False returns the full pre-pruning dict, which a
            per-feature ablation needs; it is also the original family's index path.
        store: The snapshot store to read; ``None`` loads the two teams' timelines for this
            one row (the live fallback). A bulk build passes one store for every row.

    Returns:
        Flat feature dict (pruned set when `prune`, else the full set).
    """
    as_of = AsOf.of(as_of)
    if store is None:
        store = SnapshotStore(conn, teams=(team1_id, team2_id))
    t1_snapshot = store.team_snapshot(team1_id, as_of)
    t2_snapshot = store.team_snapshot(team2_id, as_of)

    # Team composite ratings (role-weighted)
    t1_composite = t1_snapshot["composite"]
    t2_composite = t2_snapshot["composite"]

    def _sub(composite, outer_key, inner_key, default=0.0):
        return composite.get(outer_key, {}).get(inner_key, default)

    # Recent form
    t1_form = t1_snapshot["form"]
    t2_form = t2_snapshot["form"]

    # Synergy / roster snapshot
    t1_synergy = t1_snapshot["synergy"]
    t2_synergy = t2_snapshot["synergy"]

    t1_region = t1_snapshot["region"]
    t2_region = t2_snapshot["region"]
    features = {
        # Team 1 laning
        "t1_golddiffat15": _sub(t1_composite, "avg_laning", "golddiffat15"),
        "t1_csdiffat15": _sub(t1_composite, "avg_laning", "csdiffat15"),
        "t1_xpdiffat15": _sub(t1_composite, "avg_laning", "xpdiffat15"),
        "t1_cspm": _sub(t1_composite, "avg_laning", "cspm"),
        # Team 1 teamfighting
        "t1_dpm": _sub(t1_composite, "avg_teamfighting", "dpm"),
        "t1_damageshare": _sub(t1_composite, "avg_teamfighting", "damageshare"),
        "t1_killparticipation": _sub(t1_composite, "avg_teamfighting", "killparticipation"),
        "t1_kda": _sub(t1_composite, "avg_teamfighting", "kda"),
        "t1_earnedgoldshare": _sub(t1_composite, "avg_teamfighting", "earnedgoldshare"),
        # Team 1 vision
        "t1_vspm": _sub(t1_composite, "avg_vision", "vspm"),
        "t1_controlwardsbought": _sub(t1_composite, "avg_vision", "controlwardsbought"),
        # Team 1 form
        "t1_win_rate_10": t1_form["win_rate_10"],
        "t1_win_rate_20": t1_form["win_rate_20"],
        "t1_win_streak": t1_form["win_streak"],
        "t1_loss_streak": t1_form["loss_streak"],
        # Team 1 synergy
        "t1_games_together": t1_synergy["games_together"],
        "t1_lineup_win_rate": t1_synergy["lineup_win_rate"],
        "t1_weighted_win_rate": t1_synergy["weighted_win_rate"],
        # Team 2 laning
        "t2_golddiffat15": _sub(t2_composite, "avg_laning", "golddiffat15"),
        "t2_csdiffat15": _sub(t2_composite, "avg_laning", "csdiffat15"),
        "t2_xpdiffat15": _sub(t2_composite, "avg_laning", "xpdiffat15"),
        "t2_cspm": _sub(t2_composite, "avg_laning", "cspm"),
        # Team 2 teamfighting
        "t2_dpm": _sub(t2_composite, "avg_teamfighting", "dpm"),
        "t2_damageshare": _sub(t2_composite, "avg_teamfighting", "damageshare"),
        "t2_killparticipation": _sub(t2_composite, "avg_teamfighting", "killparticipation"),
        "t2_kda": _sub(t2_composite, "avg_teamfighting", "kda"),
        "t2_earnedgoldshare": _sub(t2_composite, "avg_teamfighting", "earnedgoldshare"),
        # Team 2 vision
        "t2_vspm": _sub(t2_composite, "avg_vision", "vspm"),
        "t2_controlwardsbought": _sub(t2_composite, "avg_vision", "controlwardsbought"),
        # Team 2 form
        "t2_win_rate_10": t2_form["win_rate_10"],
        "t2_win_rate_20": t2_form["win_rate_20"],
        "t2_win_streak": t2_form["win_streak"],
        "t2_loss_streak": t2_form["loss_streak"],
        # Team 2 synergy
        "t2_games_together": t2_synergy["games_together"],
        "t2_lineup_win_rate": t2_synergy["lineup_win_rate"],
        "t2_weighted_win_rate": t2_synergy["weighted_win_rate"],
        # Series state: maps each side has taken so far in this series.
        "t1_series_wins": float(series_wins[0]),
        "t2_series_wins": float(series_wins[1]),
    }

    _add_snapshot_groups(features, t1_snapshot, t2_snapshot)

    # Head-to-head features, pairwise rather than per-team
    features.update(store.h2h(team1_id, team2_id, as_of))

    # Match-level contextual features (format, stage, playoff flag)
    stage_lower = (stage or "").lower()
    features["is_playoffs"] = (
        1.0
        if any(kw in stage_lower for kw in ("playoff", "final", "elimination", "bracket"))
        else 0.0
    )
    format_map = {"Bo1": 1.0, "Bo3": 3.0, "Bo5": 5.0}
    features["format_bo"] = format_map.get(match_format, 1.0)

    # League tier context (1=premier, 2=regional, 3=minor)
    t1_tier = (
        LEAGUE_TIER_MAP.get(t1_region, LEAGUE_TIER_DEFAULT) if t1_region else LEAGUE_TIER_DEFAULT
    )
    t2_tier = (
        LEAGUE_TIER_MAP.get(t2_region, LEAGUE_TIER_DEFAULT) if t2_region else LEAGUE_TIER_DEFAULT
    )
    # Exact domestic leagues preserve the tier distinction between premier and feeder circuits.
    assert t1_tier is not None
    assert t2_tier is not None
    features["league_tier_avg"] = (t1_tier + t2_tier) / 2.0
    features["is_cross_tier"] = 1.0 if t1_tier != t2_tier else 0.0

    # Pairwise difference features (the diff bases are declared in features/manifest.py)
    for _k in diff_keys():
        _t1 = features.get(f"t1_{_k}", 0.0)
        _t2 = features.get(f"t2_{_k}", 0.0)
        features[f"diff_{_k}"] = _t1 - _t2

    # Winsorize the KDA diff (see the _KDA_WINSOR_CAP comment).
    features["diff_kda"] = max(-_KDA_WINSOR_CAP, min(_KDA_WINSOR_CAP, features["diff_kda"]))

    # Blue side indicator: team1 in canonical ordering is always blue side.
    # Training preparation overrides this for swapped rows.
    features["is_t1_blue_side"] = 1.0

    if not prune:
        return features
    return _drop_low_importance(_drop_diff_only_redundant(features))
