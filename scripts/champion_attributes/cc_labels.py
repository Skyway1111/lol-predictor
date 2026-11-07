"""The closed vocabulary of wiki leveling labels that state crowd control, read exactly."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

#: Units a percent-valued label may carry, compared after `strip().lower()` because the
#: export's unit column carries stray whitespace ("  seconds").
_PERCENT_UNITS = frozenset({"%"})
#: A duration row states seconds, sometimes with the unit column left blank.
_SECONDS_UNITS = frozenset({"seconds", "second", ""})

#: Unit spellings a ruled label may carry that source nothing, with the reason. A ruled
#: label arriving with any OTHER unit raises: the whitelist must not fail open.
IGNORED_UNITS: dict[str, str] = {
    "% per 100 ap": "the AP coefficient beside the base value, not the control's strength",
    "% per 100 poppy's bonus health": "the bonus-health coefficient beside Poppy's base slow",
}

#: Exact label to the dims it sources, with the unit kind the value must carry.
#: This table is the reviewed declaration: editing it changes the artifact's meaning.
LABEL_RULES: dict[str, tuple[tuple[str, ...], frozenset[str]]] = {
    "Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Maximum Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Minimum Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Total Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Enhanced Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Total Enhanced Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Increased Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Empowered Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Subsequent Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Reduced Slow": (("cc_slow_pct",), _PERCENT_UNITS),
    "Cripple Strength": (("cc_slow_pct",), _PERCENT_UNITS),
    "Maximum Cripple": (("cc_slow_pct",), _PERCENT_UNITS),
    "Slow Duration": (("cc_slow_duration",), _SECONDS_UNITS),
    "Stun Duration": (("cc_hard_seconds", "cc_type_stun"), _SECONDS_UNITS),
    "Root Duration": (("cc_hard_seconds", "cc_type_root"), _SECONDS_UNITS),
    "Total Root Duration": (("cc_hard_seconds", "cc_type_root"), _SECONDS_UNITS),
    "Empowered Root Duration": (("cc_hard_seconds", "cc_type_root"), _SECONDS_UNITS),
    "Fear Duration": (("cc_hard_seconds", "cc_type_fear"), _SECONDS_UNITS),
    "Taunt Duration": (("cc_hard_seconds", "cc_type_taunt"), _SECONDS_UNITS),
    "Silence Duration": (("cc_hard_seconds", "cc_type_silence"), _SECONDS_UNITS),
    "Knock Up Duration": (("cc_hard_seconds", "cc_type_knockup"), _SECONDS_UNITS),
    "Maximum Knock up Duration": (("cc_hard_seconds", "cc_type_knockup"), _SECONDS_UNITS),
    "Airborne Duration": (("cc_hard_seconds", "cc_type_knockup"), _SECONDS_UNITS),
    # A generic disable: the row states the seconds, never which control it is.
    "Disable Duration": (("cc_hard_seconds",), _SECONDS_UNITS),
    # Berserk blocks controlled action, so it is lockdown seconds with no type flag.
    "Berserk Duration": (("cc_hard_seconds",), _SECONDS_UNITS),
}

#: Exact label to why it sources nothing. A CC-family label lands here or in
#: LABEL_RULES; anything else raises rather than reading as zero.
IGNORED_LABELS: dict[str, str] = {
    "Slow Strength Increase": "an increment applied to a slow stated elsewhere",
    "Additional Slow Per Second": "a ramp step, not the slow the ability applies",
    "Additional Cripple Per Second": "a ramp step, not the cripple the ability applies",
    "Root Duration Increase": "an increment applied to a root stated elsewhere",
    "Monster Disable Duration": "a jungle-camp disable, never applied to a champion",
    "Turret Disable Duration": "a structure disable, never applied to a champion",
    "Knock Back Distance": "units of displacement, not a control duration",
    "Blind Duration": "blind blocks no action and has no dim in this schema",
    "Increased Blind Duration": "blind blocks no action and has no dim in this schema",
}

#: Every dim any rule can source. The artifact's key list must contain each one.
SOURCED_DIMS: tuple[str, ...] = tuple(
    dict.fromkeys(dim for dims, _units in LABEL_RULES.values() for dim in dims)
)

#: Word stems that make a label a control claim. A label matching one must be ruled
#: or ignored, so a renamed or new wiki row raises instead of silently sourcing zero.
CC_FAMILY_PATTERN = re.compile(
    r"slow|stun|root|snare|charm|taunt|fear|sleep|drowsy|polymorph|silence|suppress"
    r"|knock|airborne|pull|ground|disarm|blind|nearsight|cripple|berserk|disable"
    r"|immobiliz|stasis|entangl",
    re.IGNORECASE,
)


class UnmappedCcLabel(RuntimeError):
    """A control-family leveling label is in neither LABEL_RULES nor IGNORED_LABELS."""


class UnmappedCcUnit(RuntimeError):
    """A ruled label carries a unit that is neither its expected one nor a declared drop."""


def is_unruled(attr: str) -> bool:
    """Whether a leveling label claims control but no table has decided what it sources."""
    return (
        attr not in LABEL_RULES
        and attr not in IGNORED_LABELS
        and bool(CC_FAMILY_PATTERN.search(attr))
    )


def cc_from_labels(entries: Iterable[tuple[str, float, str]]) -> dict[str, float]:
    """Control dims one ability's leveling rows state outright: magnitude, then its flags.

    Raises:
        UnmappedCcLabel: a control-family label carries no rule and no ignore reason.
        UnmappedCcUnit: a ruled label carries an undeclared unit, which would read zero.
    """
    sourced = dict.fromkeys(SOURCED_DIMS, 0.0)
    for attr, value, unit in entries:
        rule = LABEL_RULES.get(attr)
        if rule is None:
            if is_unruled(attr):
                raise UnmappedCcLabel(_unmapped_message({attr: []}))
            continue
        dims, units = rule
        normalized = unit.strip().lower()
        if normalized not in units:
            if normalized not in IGNORED_UNITS:
                raise UnmappedCcUnit(
                    f"{attr!r} carries unit {unit!r}, which is neither its expected unit "
                    f"({sorted(units)}) nor a declared drop (cc_labels.IGNORED_UNITS). "
                    "Rule it or record why it sources nothing."
                )
            continue
        magnitude, *flags = dims
        sourced[magnitude] = max(sourced[magnitude], value)
        for flag in flags:
            sourced[flag] = 1.0
    return sourced


def audit_family_labels(export: Mapping[str, Any]) -> None:
    """Sweep the whole champion export once, naming every unruled control-family label.

    Raises:
        UnmappedCcLabel: names each offending label and the champions carrying it.
    """
    offenders: dict[str, list[str]] = {}
    for name, champion in export.items():
        for abilities in (champion.get("abilities") or {}).values():
            for ability in abilities or []:
                for effect in ability.get("effects") or []:
                    for leveling in effect.get("leveling") or []:
                        attr = str(leveling.get("attribute", ""))
                        if is_unruled(attr):
                            offenders.setdefault(attr, []).append(name)
    if offenders:
        raise UnmappedCcLabel(_unmapped_message(offenders))


def _unmapped_message(offenders: Mapping[str, list[str]]) -> str:
    """Name each unruled label and up to five champions carrying it."""
    lines = [
        f"  {label!r} on {', '.join(sorted(set(champs))[:5]) or '(direct call)'}"
        for label, champs in sorted(offenders.items())
    ]
    return (
        "control-family leveling labels with no rule and no ignore reason "
        "(cc_labels.LABEL_RULES / IGNORED_LABELS):\n" + "\n".join(lines)
    )
