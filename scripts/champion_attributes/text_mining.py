"""Ability feature values mined from wiki effect text: CC, slows, mechanic flags, grants."""

# Build-time contract: extraction thresholds are fit against these patterns' exact
# matching behavior, so editing a pattern requires regenerating the artifact.

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

#: Airborne wording. The excluded gap words keep "knocked back up to 300 units" a
#: knockback and "knock down" a stun; "suspend" is the wiki's other airborne word.
_KNOCKUP_VERBS = (
    r"knock(?:ed|ing|s)?(?:\s+(?!back\b|down\b|away\b|aside\b)\w+){0,3}\s+up\b"
    r"|knockup|airborne|suspend(?:ed|ing|s)?"
)

# The hard-CC verb alternation — one definition, interpolated into the three
# patterns below (a capturing group, so the matched verb is available).
_HARD_CC_VERBS = (
    r"(stun(?:ned|ning|s)?|" + _KNOCKUP_VERBS + r"|root(?:ed|ing|s)?|"
    r"suppress(?:ed|es|ing)?|fear(?:ed|ing|s)?|silenc(?:e|ed|es|ing)|"
    r"charm(?:ed|ing|s)?|taunt(?:ed|ing|s)?|sleep|polymorph(?:ed|ing|s)?)"
)
#: A stated duration: per-rank or per-level numbers, then an optional parenthetical
#: before the unit ("0.75 / 1 / 1.25 / 1.5 (based on level) seconds").
_SECONDS = r"(?P<value>\d+(?:\.\d+)?(?:\s*[:/]\s*\d+(?:\.\d+)?)*)\s*(?:\([^)]*\)\s*)?seconds?"
# "<value> seconds ... <hard-CC verb>" — duration precedes the CC term.
HARD_CC_PATTERN = re.compile(_SECONDS + r".{0,32}?" + _HARD_CC_VERBS, re.IGNORECASE)
# "<hard-CC verb> ... <value> seconds" — the reverse order.
HARD_CC_REVERSE_PATTERN = re.compile(_HARD_CC_VERBS + r".{0,32}?" + _SECONDS, re.IGNORECASE)
# Any hard-CC verb at all — the cheap "is this sentence about hard CC" filter.
HARD_CC_TERM_PATTERN = re.compile(_HARD_CC_VERBS, re.IGNORECASE)

# A sentence mentioning only these targets is CC on jungle camps/waves, not
# champions; each builder pairs this with its own champion-relevant term list.
NON_CHAMPION_ONLY_TERMS = ("minion", "minions", "monster", "monsters")

DASH_TERMS = (
    "dash",
    "dashes",
    "blink",
    "blinks",
    "leap",
    "leaps",
    "vault",
    "vaults",
    "lunges",
    "lunge",
    "teleport",
    "teleports",
)

# Sentence boundaries: ./!/? not inside a number (2.5), plus ;/newlines.
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<!\d)[.!?](?!\d)|[;\n]+")

#: A negated term names what an effect does NOT apply ("cleanses non- airborne crowd
#: control", "the nearest non- sleeping enemy"). The wiki's hyphen may trail a space.
NEGATED_TERM_PATTERN = re.compile(r"\bnon-\s*\w+")

#: An exemption clause names who is spared; an "against minions" clause adjusts what
#: they take. Neither names a recipient, and both end at a comma, period or bracket.
EXCLUSION_CLAUSE_PATTERN = re.compile(
    r"\b(?:excluding|except|but not)\b[^,.;)]*"
    r"|\bagainst\s+(?:\w+\s+){0,2}?(?:minions?|monsters?)\b",
    re.IGNORECASE,
)

#: One word of wiki prose. `\w` alone stops at the possessive in "Janna's abilities";
#: U+2019 covers the export's curly apostrophe.
_WORD = r"[\w'\u2019-]+"

#: Control participles, for the clause forms that name a control without applying one.
_CC_PARTICIPLES = (
    r"immobiliz\w*|grounded|silenced|stunned|rooted|feared|charmed|taunted|asleep"
    r"|airborne|suppressed|polymorphed|slowed|disarmed|knocked\s+\w+"
)

#: A condition covers the participle and its coordinated siblings only, so an applied
#: control later in the same sentence survives (Lee Sin R: "while it is airborne").
_CONDITION_TAIL = (
    r"(?:" + _CC_PARTICIPLES + r")"
    r"(?:(?:\s*,\s*|\s+)(?:or\s+|and\s+|nor\s+)?(?:" + _CC_PARTICIPLES + r"))*"
)
#: State condition, cancel condition, trigger, attributive participle — the four forms
#: that name a control someone is already under.
CONDITION_CLAUSE_PATTERN = re.compile(
    r"\bwhile\s+(?:"
    + _WORD
    + r"\s+){0,3}?"
    + _CONDITION_TAIL
    + r"|\bbecom(?:e|es|ing)\s+"
    + _CONDITION_TAIL
    + r"|\bwhenever\b(?:\s+"
    + _WORD
    + r"){0,20}"
    + r"|\b(?:enem(?:y|ies)|champions?|targets?|units?)\s+(?:"
    + _CC_PARTICIPLES
    + r")\s+by\b",
    re.IGNORECASE,
)


def ability_effect_texts(ability: Mapping[str, Any]) -> list[str]:
    """The blurb + per-effect description texts of one wiki ability entry."""
    texts: list[str] = []
    if ability.get("blurb"):
        texts.append(str(ability["blurb"]))
    texts.extend(
        str(effect["description"])
        for effect in ability.get("effects") or []
        if isinstance(effect, Mapping) and effect.get("description")
    )
    return texts


#: Enemy-recipient terms, matched over a whole effect description (Riot names the victim
#: once, then says "them"). Word-bounded: "untargetable" must not read as "target".
_ENEMY_RECIPIENT_PATTERN = re.compile(r"\benem(?:y|ies)\b|\bchampions?\b|\btargets?\b")

#: "champion" is the only term that rescues a chunk naming minions or monsters: "enemy"
#: and "target" both qualify them ("enemy minions", "the target minion").
_CHAMPION_TERM = "champion"

#: Chunk-level CC-type detections. Ground matches only the canonical "grounded"; fear
#: requires an inflected verb or a direct object ("Taste Their Fear" is an ability name).
_CC_TYPE_PATTERNS: dict[str, re.Pattern] = {
    "cc_type_charm": re.compile(r"\bcharm(?:s|ed|ing)?\b"),
    "cc_type_taunt": re.compile(r"\btaunt(?:s|ed|ing)?\b"),
    "cc_type_fear": re.compile(
        r"\bfear(?:s|ed|ing)\b|\bfear\b(?=\s+(?:them|all|nearby|enem|target|champion))"
    ),
    "cc_type_sleep": re.compile(r"\basleep\b|\bsleeps?\b|\bsleeping\b"),
    # Riot writes a knockback four ways: back, away, aside, or a bare unit distance.
    "cc_type_knockback": re.compile(
        r"\bknock(?:s|ed|ing)?(?:\s+\w+){0,3}\s+(?:back|away|aside)\b"
        r"|\bknockback\b"
        r"|\bknock(?:s|ed|ing)?(?:\s+\w+){0,4}?\s+\d[\d,.]*\s*units\b"
    ),
    "cc_type_pull": re.compile(r"\bpull(?:s|ed|ing)?\b"),
    "cc_type_ground": re.compile(r"\bgrounded\b"),
    "cc_type_disarm": re.compile(r"\bdisarm(?:s|ed|ing)?\b"),
    "cc_type_polymorph": re.compile(r"\bpolymorph(?:s|ed|ing)?\b"),
}


def _is_non_champion_only(lowered: str) -> bool:
    """True when the chunk names only minions/monsters — "enemy minions" must not pass as CC."""
    return _CHAMPION_TERM not in lowered and any(t in lowered for t in NON_CHAMPION_ONLY_TERMS)


def _book_cc_types(result: dict[str, float], lowered: str) -> None:
    """Raise the schema-v3 ``cc_type_*`` flag of every ``_CC_TYPE_PATTERNS`` term in the chunk."""
    for key, pattern in _CC_TYPE_PATTERNS.items():
        if pattern.search(lowered):
            result[key] = 1.0


#: The individual numbers inside a matched duration, so a per-rank list reads its top.
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


def _hard_cc_seconds(lowered: str) -> float:
    """Longest hard-CC duration the chunk states; unnumbered "brief" reads as 0.5s."""
    duration = 0.0
    for pattern in (HARD_CC_REVERSE_PATTERN, HARD_CC_PATTERN):
        for match in pattern.finditer(lowered):
            duration = max(
                duration, *(float(n) for n in _NUMBER_PATTERN.findall(match.group("value")))
            )
    if duration <= 0.0 and "brief" in lowered:
        return 0.5
    return duration


def _book_hard_cc_types(result: dict[str, float], lowered: str) -> None:
    """Raise the chunk's hard-CC type flags; airborne wording wins the stun co-flag."""
    # The wiki's "knocks up and stuns" is one CC ("knock down"/"knock back" are not
    # airborne); schema v3 keeps silence to 'silence', the rest carry own cc_type_* keys.
    if re.search(_KNOCKUP_VERBS, lowered):
        result["cc_type_knockup"] = 1.0
    elif "stun" in lowered:
        result["cc_type_stun"] = 1.0
    if "root" in lowered:
        result["cc_type_root"] = 1.0
    if "suppress" in lowered:
        result["cc_type_suppress"] = 1.0
    if "silence" in lowered:
        result["cc_type_silence"] = 1.0


def champion_cc_chunks(texts: Iterable[str]) -> Iterator[str]:
    """The sentences of one ability that may state control applied to a champion.

    Yields:
        Masked sentences of a description that names an enemy recipient, minus those
        whose only recipients are minions or monsters. Every CC reader shares this path.
    """
    for text in texts:
        masked = text.lower()
        for pattern in (NEGATED_TERM_PATTERN, EXCLUSION_CLAUSE_PATTERN, CONDITION_CLAUSE_PATTERN):
            masked = pattern.sub(" ", masked)
        if not _ENEMY_RECIPIENT_PATTERN.search(masked):
            continue
        for chunk in SENTENCE_SPLIT_PATTERN.split(masked):
            lowered = chunk.strip()
            if lowered and not _is_non_champion_only(lowered):
                yield lowered


def extract_cc_from_text(texts: Iterable[str]) -> dict[str, float]:
    """Hard-CC seconds and type flags; only the text names the recipient, never ``affects``."""
    result: dict[str, float] = {
        "cc_hard_seconds": 0.0,
        "cc_type_knockup": 0.0,
        "cc_type_stun": 0.0,
        "cc_type_root": 0.0,
        "cc_type_suppress": 0.0,
        "cc_type_silence": 0.0,
        **dict.fromkeys(_CC_TYPE_PATTERNS, 0.0),
    }
    for lowered in champion_cc_chunks(texts):
        _book_cc_types(result, lowered)
        if not HARD_CC_TERM_PATTERN.search(lowered):
            continue
        result["cc_hard_seconds"] = max(result["cc_hard_seconds"], _hard_cc_seconds(lowered))
        _book_hard_cc_types(result, lowered)
    return result


#: "slows them by 40% for 2 seconds", "slowing them by 70% decaying over 2 seconds". The
#: gap stays inside one sentence, so an unrelated later duration never becomes the slow's.
_SLOW_PATTERN = re.compile(
    r"slow(?:s|ed|ing)?\b[^.;]{0,60}?(?:for|over)\s+(?P<dur>\d+(?:\.\d+)?)\s*seconds?",
    re.IGNORECASE,
)
_SLOW_REVERSE_PATTERN = re.compile(
    r"for\s+(?P<dur>\d+(?:\.\d+)?)\s*seconds?,?\s+"
    r"(?:\w+\s+){0,5}slow(?:s|ed|ing)?",
    re.IGNORECASE,
)


def extract_slow_duration(texts: Iterable[str]) -> float:
    """Longest slow the ability states on a champion recipient."""
    best = 0.0
    for lowered in champion_cc_chunks(texts):
        for pattern in (_SLOW_PATTERN, _SLOW_REVERSE_PATTERN):
            for match in pattern.finditer(lowered):
                best = max(best, float(match.group("dur")))
    return best


_STEALTH_TERMS = ("invisible", "camouflage", "stealth", "unseen")
_TERRAIN_TERMS = ("terrain", "impassable", "pillar", "creates a wall")
_EXECUTE_TERMS = ("missing health", "execute", "current health")
_STACKING_TERMS = ("stack", "stacks")
_PERSISTENT_AOE_TERMS = (
    "per second",
    "leaving a trail",
    "zone that",
    "area that",
    "controllable pet",
    "remains on the field",
)

#: Deployable/pet context: a sentence naming one of these describes the summoned unit's
#: state, not the champion's.
_SUMMON_CONTEXT_TERMS = (
    "summon",
    "deploy",
    "spawn",
    "trap",
    "clone",
    "sentry",
    "tentacle",
    "faerie",
    "bloblet",
    "box is",
    "shadow",
)

#: Denial-family terms, chunk-level. A chunk with summon/deployable context never
#: fires them — the summoned unit is the recipient, not the champion.
_DENIAL_TERMS: dict[str, tuple[str, ...]] = {
    "has_untargetable": ("untargetable",),
    "has_invulnerable": ("invulnerable", "invulnerability"),
    "has_stasis": ("stasis",),
    "has_damage_reduction": (
        "damage reduction",
        "reduces incoming damage",
        "damage taken is reduced",
    ),
}

#: has_revive: "revive*" and "resurrect(s|ed)" are owned revives; the state-noun
#: "resurrection" counts only alongside a fatal/lethal-damage grant.
_REVIVE_PATTERN = re.compile(r"\brevive|\bresurrect(?:s|ed)\b")
_REVIVE_GRANT_TERMS = ("fatal damage", "lethal damage")

#: has_spellshield: grants self/ally a hostile-effect-blocking OR cc-immunity buff. The
#: cc-immunity terms also require grant context, so a bare immunity aside never fires.
_SPELLSHIELD_TERMS = ("spell shield", "spellshield")
_CC_IMMUNITY_TERMS = ("crowd control immunity", "immunity to crowd control")
_GRANT_CONTEXT_TERMS = ("grant", "gain")

_PROJECTILE_BLOCK_VERBS = ("block", "destroy", "intercept")

#: Minimum % bonus for a movement-speed steroid; below it, incidental ms ticks are
#: ubiquitous enough (110 of 173 champions) to stop discriminating.
_MS_BOOST_MIN_PERCENT = 20.0
_MS_BOOST_PATTERN = re.compile(r"(\d+(?:\.\d+)?)%[^.;%]{0,40}?bonus (?:total )?movement speed")

#: Text fallbacks for grants the wiki states only in prose; leveling entries win via
#: max-merge when both exist.
_SHIELD_HP_PCT_PATTERN = re.compile(
    r"shield[^.;]{0,60}?(\d+(?:\.\d+)?)% of (?:his|her|their|its) maximum health"
)
_DAMAGE_REDUCTION_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)% damage reduction")
_ARMOR_SHRED_PCT_PATTERN = re.compile(r"(\d+(?:\.\d+)?)% armor reduction")

_TANK_TEXT_PATTERNS: dict[str, re.Pattern] = {
    "shield_hp_pct": _SHIELD_HP_PCT_PATTERN,
    "damage_reduction_pct": _DAMAGE_REDUCTION_PCT_PATTERN,
    "armor_shred_pct": _ARMOR_SHRED_PCT_PATTERN,
}


def extract_tank_text_grants(texts: Iterable[str]) -> dict[str, float]:
    """Prose-only grant magnitudes (``_TANK_TEXT_PATTERNS``), max per key."""
    result = dict.fromkeys(_TANK_TEXT_PATTERNS, 0.0)
    for text in texts:
        lowered = text.lower()
        for key, pattern in _TANK_TEXT_PATTERNS.items():
            for match in pattern.finditer(lowered):
                result[key] = max(result[key], float(match.group(1)))
    return result


def _book_chunk_mechanic_flags(flags: dict[str, float], lowered: str) -> None:
    """Raise one chunk's per-sentence mechanic flags; summon context denies the denial family."""
    if any(float(m.group(1)) >= _MS_BOOST_MIN_PERCENT for m in _MS_BOOST_PATTERN.finditer(lowered)):
        flags["has_ms_boost"] = 1.0
    if any(t in lowered for t in _SUMMON_CONTEXT_TERMS):
        return
    for key, terms in _DENIAL_TERMS.items():
        if any(t in lowered for t in terms):
            flags[key] = 1.0
    if _REVIVE_PATTERN.search(lowered) or (
        "resurrect" in lowered and any(t in lowered for t in _REVIVE_GRANT_TERMS)
    ):
        flags["has_revive"] = 1.0
    if any(t in lowered for t in _SPELLSHIELD_TERMS) or (
        any(t in lowered for t in _CC_IMMUNITY_TERMS)
        and any(t in lowered for t in _GRANT_CONTEXT_TERMS)
    ):
        flags["has_spellshield"] = 1.0
    if "projectile" in lowered and any(v in lowered for v in _PROJECTILE_BLOCK_VERBS):
        flags["has_projectile_block"] = 1.0


def extract_mechanic_flags(texts: list[str]) -> dict[str, float]:
    """Boolean mechanic flags from description text; denial and ms-boost flags read per sentence."""
    combined = " ".join(texts).lower()
    flags = {
        "has_dash": 1.0 if any(t in combined for t in DASH_TERMS) else 0.0,
        "has_unstoppable": 1.0
        if "unstoppable" in combined or "displacement immun" in combined
        else 0.0,
        "has_stealth": 1.0 if any(t in combined for t in _STEALTH_TERMS) else 0.0,
        "has_terrain": 1.0 if any(t in combined for t in _TERRAIN_TERMS) else 0.0,
        "has_execute": 1.0 if any(t in combined for t in _EXECUTE_TERMS) else 0.0,
        "has_stacking": 1.0 if any(t in combined for t in _STACKING_TERMS) else 0.0,
        "has_persistent_aoe": 1.0 if any(t in combined for t in _PERSISTENT_AOE_TERMS) else 0.0,
        **dict.fromkeys(_DENIAL_TERMS, 0.0),
        "has_revive": 0.0,
        "has_spellshield": 0.0,
        "has_projectile_block": 0.0,
        "has_ms_boost": 0.0,
    }
    for text in texts:
        for chunk in SENTENCE_SPLIT_PATTERN.split(text):
            lowered = chunk.lower().strip()
            if lowered:
                _book_chunk_mechanic_flags(flags, lowered)
    return flags
