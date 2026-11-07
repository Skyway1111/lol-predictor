"""Control vocabulary shared by the champion-data readers and knowledge audits."""

from __future__ import annotations

#: The store's lockdown KIND evidence. Its magnitude is `cc_hard_seconds`, audited
#: separately: a stun the reader flagged but could not time is a different failure.
HARD_CC_TYPE_DIMS: tuple[str, ...] = (
    "cc_type_knockup",
    "cc_type_stun",
    "cc_type_root",
    "cc_type_suppress",
    "cc_type_silence",
    "cc_type_charm",
    "cc_type_taunt",
    "cc_type_fear",
    "cc_type_sleep",
    "cc_type_knockback",
    "cc_type_pull",
    "cc_type_ground",
    "cc_type_disarm",
    "cc_type_polymorph",
)

#: The store's soft-control evidence: a slow's duration or its sourced strength.
SOFT_CC_DIMS: tuple[str, ...] = ("cc_slow_duration", "cc_slow_pct")

#: A reviewed kind this schema carries no dim for, with the reason.
UNREPRESENTED_KINDS: dict[str, str] = {
    "blind": "blind blocks no action and denies no position; no dim expresses it",
}

#: The calculator's sentinel for a slot whose control kind varies per damage part. It
#: asserts no control on any particular part, so it cannot be a recall claim.
VARIES_PER_PART = "per_part"
