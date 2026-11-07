"""Declarative schema, pruning rules, and compute-site homes for the served feature columns."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The version families of a served artifact's feature schema. Bump when a manifest edit
#: changes what the ``original`` family emits or prunes; the fingerprint tracks finer edits.
SCHEMA_VERSION = 3

_OTHER_FAMILY_PREFIXES = ("rating_", "lineup_", "recency_")


@dataclass(frozen=True)
class FeatureSpec:
    """One original-family base (or context key) and how it behaves.

    Attributes:
        base: the base name (``golddiffat15``) for a sided feature, or the full key
            (``is_playoffs``) for a context feature.
        dead: the emitted columns in the low-importance prune — a subset of
            ``('t1', 't2', 'diff')`` for a sided base, or ``('self',)`` for a context key.
        note: a locked quirk (a definition or unit the served artifact froze).
    """

    base: str
    context: bool = False  # a bare context key (emitted once), not a sided t1_/t2_ base
    diff: bool = False  # emits diff_<base> (sided bases only)
    drop_sides: bool = False  # prune pass 1 drops t1_/t2_<base>
    dead: tuple[str, ...] = ()  # emitted columns in the dead-weight prune (see class docstring)
    note: str = ""  # locked-quirk explanation

    def emitted(self) -> tuple[str, ...]:
        """The fully-qualified column names this spec produces pre-prune."""
        if self.context:
            return (self.base,)
        names = [f"t1_{self.base}", f"t2_{self.base}"]
        if self.diff:
            names.append(f"diff_{self.base}")
        return tuple(names)

    def dead_names(self) -> tuple[str, ...]:
        """The subset of :meth:`emitted` that is low-importance dead weight."""
        if self.context:
            return (self.base,) if self.dead else ()
        table = {"t1": f"t1_{self.base}", "t2": f"t2_{self.base}", "diff": f"diff_{self.base}"}
        return tuple(table[tag] for tag in self.dead)


# The manifest — the original family + its context keys, one row per base/key,
# grouped by compute site in snapshot-builder emission order.
_SIDED = [
    # Laning / teamfighting / vision — role-weighted, recency-decayed player averages.
    FeatureSpec("golddiffat15", diff=True, drop_sides=True),
    FeatureSpec("csdiffat15", diff=True, drop_sides=True),
    FeatureSpec("xpdiffat15", diff=True, drop_sides=True),
    FeatureSpec("cspm", diff=True, drop_sides=True),
    FeatureSpec("dpm", diff=True, drop_sides=True),
    FeatureSpec("damageshare", diff=True, drop_sides=True),
    FeatureSpec("killparticipation", diff=True, drop_sides=True),
    FeatureSpec("kda", diff=True, drop_sides=True),
    FeatureSpec("earnedgoldshare", diff=True, drop_sides=True),
    FeatureSpec("vspm", diff=True, drop_sides=True),
    FeatureSpec("controlwardsbought", diff=True, drop_sides=True),
    # Synergy — how long/well the fielded five have played together.
    FeatureSpec("games_together", diff=True, drop_sides=True),
    FeatureSpec("lineup_win_rate", diff=True, drop_sides=True),
    FeatureSpec("weighted_win_rate", diff=True, drop_sides=True),
    # Form — keep sides (a 60% team is a different prior than two 60% teams).
    FeatureSpec("win_rate_10", diff=True, dead=("t1", "t2", "diff")),
    FeatureSpec(
        "win_rate_20",
        diff=True,
        dead=("diff",),
        note="win_rate_20 is a LEGACY key name: it is the win rate over the FULL recent "
        "form window, and the served config passes window=15 — so it counts 15 games, "
        "not 20. The name is frozen by the persisted snapshot schema and the served "
        "feature set; changing it requires a served retrain.",
    ),
    FeatureSpec("win_streak", diff=True, dead=("t1", "t2", "diff")),
    FeatureSpec("loss_streak", diff=True, dead=("t1", "t2", "diff")),
    # Momentum.
    FeatureSpec("win_rate_5", diff=True, dead=("t1", "t2", "diff")),
    FeatureSpec("performance_slope", diff=True, dead=("diff",)),
    # Objective control — diff-only.
    FeatureSpec("tower_rate", diff=True, drop_sides=True),
    FeatureSpec("first_dragon_rate", diff=True, drop_sides=True),
    FeatureSpec("first_herald_rate", diff=True, drop_sides=True),
    FeatureSpec("grub_rate", diff=True, drop_sides=True, dead=("diff",)),
    # Pace — diff-only.
    FeatureSpec("avg_game_duration", diff=True, drop_sides=True),
    FeatureSpec("kills_per_min", diff=True, drop_sides=True),
    FeatureSpec("first_blood_rate", diff=True, drop_sides=True),
    # Performance variance — diff-only.
    FeatureSpec("gamedur_variance", diff=True, drop_sides=True),
    FeatureSpec("kpm_variance", diff=True, drop_sides=True),
    # Extended: schedule / roster / upset / side / margin / champ-pool / game-phase.
    FeatureSpec("games_7d", diff=True, drop_sides=True),
    FeatureSpec("games_14d", diff=True, drop_sides=True, dead=("diff",)),
    FeatureSpec("roster_days_stable", diff=True, drop_sides=True),
    FeatureSpec("roster_total_games", diff=True, drop_sides=True),
    FeatureSpec("upset_rate", diff=True, drop_sides=True),
    FeatureSpec("favorite_rate", diff=True, drop_sides=True, dead=("diff",)),
    FeatureSpec("blue_wr", diff=True, drop_sides=True),
    FeatureSpec("red_wr", diff=True, drop_sides=True),
    FeatureSpec("avg_gold_diff_end", diff=True, drop_sides=True),
    FeatureSpec(
        "stomp_rate",
        diff=True,
        drop_sides=True,
        dead=("diff",),
        note="stomp_rate and close_game_rate divide by ALL rows: a row with missing "
        "end-game gold counts as neither a stomp nor a close game (not dropped from "
        "the denominator). The served artifact trained on this definition; changing "
        "it requires a served retrain.",
    ),
    FeatureSpec("close_game_rate", diff=True, drop_sides=True),
    FeatureSpec("unique_champs", diff=True, drop_sides=True),
    FeatureSpec("avg_champ_diversity", diff=True, drop_sides=True),
    FeatureSpec("early_game_index", diff=True, drop_sides=True),
    FeatureSpec("late_game_index", diff=True, drop_sides=True),
    # EMA / Bayesian win rate — diff-only.
    FeatureSpec("ema_wr_fast", diff=True, drop_sides=True, dead=("t1", "t2", "diff")),
    FeatureSpec("ema_wr_medium", diff=True, drop_sides=True, dead=("t1", "t2", "diff")),
    FeatureSpec("ema_wr_slow", diff=True, drop_sides=True),
    FeatureSpec("bayesian_wr", diff=True, drop_sides=True),
    # Draft tendency (from PRIOR drafts) — diff-only.
    FeatureSpec("draft_score_avg", diff=True, drop_sides=True),
    FeatureSpec("draft_grade_avg", diff=True, drop_sides=True, dead=("t1", "t2", "diff")),
    FeatureSpec("draft_win_rate", diff=True, drop_sides=True, dead=("t1", "t2", "diff")),
    FeatureSpec("draft_wr_5", diff=True, drop_sides=True, dead=("t1", "t2", "diff")),
    FeatureSpec("draft_wr_10", diff=True, drop_sides=True, dead=("t1", "t2", "diff")),
    # Sided bases with NO diff column. Those with drop_sides vanish entirely under prune.
    FeatureSpec("baron_rate", drop_sides=True),
    FeatureSpec("dragon_rate", drop_sides=True),
    FeatureSpec("first_baron_rate", drop_sides=True),
    FeatureSpec("first_tower_rate", drop_sides=True),
    FeatureSpec("side_diff", drop_sides=True),
    FeatureSpec("sample_size_30d", drop_sides=True),
    # Kept sides, no diff, no prune: staleness input.
    FeatureSpec("days_since_last"),
    # Series state: maps each side has taken so far in this series (0-0 before map one).
    FeatureSpec("series_wins", diff=True),
]

_CONTEXT = [
    # Match context — assembled inline by the snapshot orchestrator.
    FeatureSpec("is_playoffs", context=True, dead=("self",)),
    FeatureSpec("format_bo", context=True),
    FeatureSpec("league_tier_avg", context=True),
    FeatureSpec("is_cross_tier", context=True),
    FeatureSpec("is_t1_blue_side", context=True),
    # Head-to-head (pairwise).
    FeatureSpec("h2h_win_rate", context=True),
    FeatureSpec("h2h_recency_wr", context=True),
    FeatureSpec("h2h_games", context=True),
    FeatureSpec("h2h_days_since", context=True),
]

MANIFEST: tuple[FeatureSpec, ...] = tuple(_SIDED + _CONTEXT)


# --- import-time integrity (cheap asserts, safety doctrine) ---------------------------- #
def _validate_manifest() -> None:
    """Reject a malformed manifest at import: dup bases, bad dead tags, context clash, no home."""
    bases = [s.base for s in MANIFEST]
    dups = {base for base, count in Counter(bases).items() if count > 1}
    if dups:
        raise ValueError(f"duplicate feature bases in the manifest: {sorted(dups)}")
    for s in MANIFEST:
        allowed = ("self",) if s.context else ("t1", "t2", "diff")
        bad = set(s.dead) - set(allowed)
        if bad:
            raise ValueError(f"{s.base!r}: dead tags {sorted(bad)} not in {allowed}")
        if s.context and (s.diff or s.drop_sides):
            raise ValueError(f"{s.base!r}: a context key emits no sides/diff")


_validate_manifest()


# --- derived views ----------------------------------------------------------------------- #
def diff_keys() -> tuple[str, ...]:
    """Return bases that emit an order-stable ``diff_<base>`` column."""
    return tuple(s.base for s in MANIFEST if s.diff)


def diff_only_bases() -> frozenset[str]:
    """Return bases whose side columns drop after diffing."""
    return frozenset(s.base for s in MANIFEST if s.drop_sides)


def dead_weight() -> frozenset[str]:
    """Return fully-qualified columns in the low-importance prune."""
    return frozenset(name for s in MANIFEST for name in s.dead_names())


def full_feature_universe() -> frozenset[str]:
    """Every column the ``original`` family emits pre-prune (all sided + diff + context keys)."""
    return frozenset(name for s in MANIFEST for name in s.emitted())


# --- served-set cross-check + schema stamp --------------------------------------------- #
def assert_served_features_covered(served_features: Iterable[str]) -> None:
    """Assert every served feature is claimed by a family, at import.

    Raises:
        ValueError: a name absent from the ``original`` universe and not owned by another
            family prefix — a typo in ``SERVED_FEATURES`` or the manifest.
    """
    universe = full_feature_universe()
    stray = sorted(
        f for f in served_features if not f.startswith(_OTHER_FAMILY_PREFIXES) and f not in universe
    )
    if stray:
        raise ValueError(
            f"served features not in the original-family manifest universe and not owned by "
            f"another family ({', '.join(_OTHER_FAMILY_PREFIXES)}): {stray} — a typo in "
            "SERVED_FEATURES or predraft/features/manifest.py. Fix one."
        )


def feature_schema_fingerprint() -> str:
    """A 16-hex sha256 of emitted and pruned features, excluding documentation."""
    payload = sorted(
        (s.base, s.context, s.diff, s.drop_sides, tuple(sorted(s.dead))) for s in MANIFEST
    )
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()[:16]
