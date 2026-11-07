"""Independent comparisons of stored champion knowledge and reviewed declarations."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from league_predictor.draft_grading.champion_attributes.ability_schema import (
    ABILITY_SLOTS,
    ALL_FEATURE_KEYS,
)
from league_predictor.draft_grading.champion_attributes.ability_store import AbilityFeatureStore
from scripts.champion_attributes.ability_features import AGREEMENT_RTOL
from scripts.champion_attributes.paths import DATA_DIR
from scripts.champion_attributes.reviewed_cc import (
    HARD_CC_TYPE_DIMS,
    SOFT_CC_DIMS,
    UNREPRESENTED_KINDS,
    VARIES_PER_PART,
)
from scripts.champion_attributes.reviewed_kit import (
    SOURCEABLE_DIMS,
    UNREPRESENTED_MECHANICS,
    ReviewedKit,
    resolved_contracts,
)

#: Row key to the line the agreement table prints. The key is what the audit branches
#: on, so renaming a label can never change which slots a class checks.
_ROW_LABELS: dict[str, str] = {
    "none": "reviewed: no control",
    "per_part": "reviewed: control varies per part",
    "blocking": "reviewed: action-blocking control",
    "no_dim": "reviewed: a kind this schema carries no dim for",
    "other": "reviewed: non-blocking control",
    "off_slot": "slots outside P/Q/W/E/R",
}


def load_reviewed_cc() -> dict:
    """Read the vendored control declarations used by the independent knowledge audit."""
    return json.loads((DATA_DIR / "reviewed_cc.json").read_text(encoding="utf-8"))


def _row_class(kind: str, blocking: frozenset[str]) -> str:
    """Which agreement row one reviewed kind belongs to."""
    if kind == "none":
        return "none"
    if kind == VARIES_PER_PART:
        return "per_part"
    if kind in blocking:
        return "blocking"
    if kind in UNREPRESENTED_KINDS:
        return "no_dim"
    return "other"


def audit_store(store: AbilityFeatureStore, reviewed: dict | None = None) -> CcAudit:
    """Compare every reviewed slot declaration with what the store carries for that slot."""
    reviewed = reviewed or load_reviewed_cc()
    blocking = frozenset(reviewed["action_blocking"])
    # A displacement states how far, not how long, so it is exempt from the seconds read.
    displacement = frozenset(reviewed["displacement"])
    unresolved: list[str] = []
    rows: dict[str, list[tuple[str, str, str]]] = {"blocking": [], "untimed": [], "control": []}
    hard_where_none: list[tuple[str, str]] = []
    counts: dict[str, int] = {}

    for champion, slots in reviewed["declarations"].items():
        resolved = store.resolve_name(champion)
        if resolved is None:
            unresolved.append(champion)
            continue
        features = store.get(resolved)
        for slot, kind in slots.items():
            known_slot = slot in ABILITY_SLOTS
            row = _row_class(kind, blocking) if known_slot else "off_slot"
            counts[row] = counts.get(row, 0) + 1
            if not known_slot:
                continue
            kinds = any(features.get(f"{slot}_{dim}", 0.0) > 0 for dim in HARD_CC_TYPE_DIMS)
            seconds = features.get(f"{slot}_cc_hard_seconds", 0.0)
            soft = any(features.get(f"{slot}_{dim}", 0.0) > 0 for dim in SOFT_CC_DIMS)
            if row == "none" and (kinds or seconds > 0):
                hard_where_none.append((champion, slot))
            elif row == "blocking":
                if not (kinds or seconds > 0):
                    rows["blocking"].append((champion, slot, kind))
                elif seconds <= 0 and kind not in displacement:
                    rows["untimed"].append((champion, slot, kind))
            elif row == "other" and not (kinds or seconds > 0 or soft):
                rows["control"].append((champion, slot, kind))

    return CcAudit(
        unresolved=tuple(sorted(unresolved)),
        blocking_missing=tuple(sorted(rows["blocking"])),
        blocking_untimed=tuple(sorted(rows["untimed"])),
        control_missing=tuple(sorted(rows["control"])),
        hard_where_none=tuple(sorted(hard_where_none)),
        counts=counts,
    )


@dataclass(frozen=True)
class CcAudit:
    """One comparison of the store against the reviewed declarations."""

    unresolved: tuple[str, ...]
    blocking_missing: tuple[tuple[str, str, str], ...]
    blocking_untimed: tuple[tuple[str, str, str], ...]
    control_missing: tuple[tuple[str, str, str], ...]
    hard_where_none: tuple[tuple[str, str], ...]
    counts: dict[str, int]

    def table(self) -> str:
        """The agreement table, one line per row class."""
        lines = [f"{v:5d}  {_ROW_LABELS[k]}" for k, v in sorted(self.counts.items())]
        lines += [f"       {kind}: {why}" for kind, why in sorted(UNREPRESENTED_KINDS.items())]
        lines += [
            f"{len(self.unresolved):5d}  champions absent from the store",
            f"{len(self.blocking_missing):5d}  action-blocking kinds with no kind flag (gate: 0)",
            f"{len(self.blocking_untimed):5d}  action-blocking kinds with no cc_hard_seconds",
            f"{len(self.control_missing):5d}  other reviewed kinds with no control at all",
            f"{len(self.hard_where_none):5d}  hard CC where the reviewed slot declares none",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class KitAudit:
    """One comparison of the prose reading against the dims the contract sources."""

    unresolved: tuple[str, ...]
    disagree: tuple[tuple[str, str, float, float], ...]
    prose_zero: tuple[tuple[str, str, float], ...]
    prose_only: tuple[tuple[str, str, float], ...]
    counts: dict[str, int]

    def table(self) -> str:
        """The agreement table, one line per row class."""
        lines = [f"{v:5d}  {k}" for k, v in sorted(self.counts.items())]
        lines += [f"       {m}: {why}" for m, why in sorted(UNREPRESENTED_MECHANICS.items())]
        lines += [
            f"{len(self.unresolved):5d}  champions the prose reading has no row for",
            f"{len(self.disagree):5d}  dims both sources state and disagree on",
            f"{len(self.prose_zero):5d}  dims the contract sources and prose reads zero",
            f"{len(self.prose_only):5d}  dims prose reads and the contract sources none",
        ]
        return "\n".join(lines)


def audit_prose(
    prose: Mapping[str, Mapping[str, float]], kit: ReviewedKit | None = None
) -> KitAudit:
    """Compare the prose-and-label reading of every champion with its contract.

    Args:
        prose: champion name to the dims `ability_features` assembles from the
            wiki export, BEFORE precedence. The finished store is not the honest
            pair: a store that sources a dim from the contract would be audited
            against its own input.
    """
    contracts, unresolved = resolved_contracts(prose, kit)
    disagree: list[tuple[str, str, float, float]] = []
    prose_zero: list[tuple[str, str, float]] = []
    prose_only: list[tuple[str, str, float]] = []
    counts = {"agree": 0, "dims compared": 0}
    readable = SOURCEABLE_DIMS & set(ALL_FEATURE_KEYS)

    for champion, sourced in contracts.items():
        read = prose[champion]
        for dim, value in sourced.items():
            counts["dims compared"] += 1
            held = float(read.get(dim, 0.0))
            if math.isclose(held, value, rel_tol=AGREEMENT_RTOL, abs_tol=AGREEMENT_RTOL):
                counts["agree"] += 1
            elif held == 0.0:
                prose_zero.append((champion, dim, value))
            else:
                disagree.append((champion, dim, value, held))
        prose_only.extend(
            (champion, dim, float(read[dim]))
            for dim in sorted(readable)
            if read.get(dim, 0.0) and dim not in sourced
        )

    return KitAudit(
        unresolved=unresolved,
        disagree=tuple(sorted(disagree)),
        prose_zero=tuple(sorted(prose_zero)),
        prose_only=tuple(sorted(prose_only)),
        counts=counts,
    )
