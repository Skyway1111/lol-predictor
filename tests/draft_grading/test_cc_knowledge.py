"""How the champion store learns crowd control: sourced labels, prose, and the drift gate."""

import pytest

from league_predictor.draft_grading.champion_attributes.ability_schema import ABILITY_FEATURE_KEYS
from league_predictor.draft_grading.champion_attributes.ability_store import (
    AbilityFeatureStore,
)
from league_predictor.draft_grading.champion_attributes.wiki_export import (
    DEFAULT_WIKI_EXPORT_PATH,
    load_wiki_champion_export,
)
from scripts.champion_attributes.ability_features import extract_all_features
from scripts.champion_attributes.cc_labels import (
    SOURCED_DIMS,
    UnmappedCcLabel,
    UnmappedCcUnit,
    audit_family_labels,
    cc_from_labels,
)
from scripts.champion_attributes.reviewed_kit import load_reviewed_kit
from scripts.champion_attributes.text_mining import (
    extract_cc_from_text,
    extract_slow_duration,
)
from tests.draft_grading._knowledge_audit import audit_store, load_reviewed_cc


@pytest.fixture(scope="module")
def wiki_data():
    return load_wiki_champion_export(DEFAULT_WIKI_EXPORT_PATH)


#: Reviewed slows neither the shipped export nor the reviewed contract states a
#: duration for, so no dim can carry them.
UNSEEN_SLOWS: dict[tuple[str, str], str] = {
    ("Ashe", "W"): "applies the named buff 'Critical Slow' with no number at all",
    ("Bel'Veth", "R"): "slows 'for the duration' of a consume with no stated seconds",
    ("Miss Fortune", "E"): "states the percent and the field's life, never the slow's",
    ("Mordekaiser", "R"): "slows 'over the cast time', which the text does not number",
    ("Nunu & Willump", "R"): "the slow scales with channel time and states no seconds",
    ("Rumble", "R"): "a zone slow that refreshes while inside, so it names no duration",
    ("Vladimir", "W"): "a pool slow that names its percent and no duration",
    ("Xin Zhao", "W"): "the export's W text names no slow at all",
}

#: Reviewed lockdowns whose kind the store carries but whose seconds it cannot read.
#: Displacements are exempt by construction: they state a distance, not a duration.
BLOCKING_WITHOUT_SECONDS: dict[tuple[str, str], str] = {
    ("Pyke", "E"): "its '1.25 (+ 0.1 per 10 Lethality) seconds' sits past the reader's window",
    ("Sett", "R"): "suppresses 'while dashing to their location'; the text states no seconds",
    ("Tahm Kench", "R"): "suppressed 'during the cast time and while attached', never numbered",
}

#: Slots we flag as hard CC where the reviewed module declares none: six controls the
#: ability really applies (MODULE_CC records one kind), four the reader still misreads.
HARD_WHERE_REVIEWED_NONE: dict[tuple[str, str], str] = {
    ("Ahri", "W"): "reader: a targeting-priority clause names Charm, another ability",
    ("Akshan", "E"): "reader: 'any immobilizing or polymorphing crowd control' is a category",
    ("Alistar", "P"): "reader: a stack trigger counts champions his OTHER abilities stun",
    ("Briar", "R"): "ours: R does fear on arrival; the module records no kind for it",
    ("Garen", "Q"): "ours: Q silences for 1.5s; the module records no kind for it",
    ("Jarvan IV", "Q"): "ours: the Q into Standard combo knocks up, per Q's own text",
    ("Ornn", "E"): "ours: E knocks up and stuns on a terrain collision",
    ("Quinn", "Q"): "reader: the disarm rider applies to non-champions only",
    ("Riven", "Q"): "ours: the third cast knocks back, per Q's own text",
    ("Vex", "P"): "ours: Doom fears for 0.75s to 1.5s by level",
}


@pytest.fixture(scope="module")
def audit():
    return audit_store(AbilityFeatureStore())


def test_both_vendored_readings_receipt_one_calculator_revision():
    """The oracle and the kit must read one revision, or the gate audits B against A."""
    control, kit = load_reviewed_cc()["module_sha256"], load_reviewed_kit().module_sha256
    assert sorted(control) == sorted(kit)
    assert [name for name in sorted(control) if control[name] != kit[name]] == []


def test_every_reviewed_champion_is_in_the_store(audit):
    """A reviewed champion the store cannot resolve would audit as an all-zero vector."""
    assert audit.unresolved == ()


def test_every_reviewed_lockdown_reaches_the_store(audit):
    """The strict half of the gate: no action-blocking control may be missing."""
    assert audit.blocking_missing == (), audit.table()


def test_the_lockdowns_the_store_cannot_time_are_the_declared_ones(audit):
    """A kind flag with no seconds is its own failure; it must not hide behind the flag."""
    assert {(champion, slot) for champion, slot, _kind in audit.blocking_untimed} == set(
        BLOCKING_WITHOUT_SECONDS
    ), audit.table()


def test_the_slows_the_export_cannot_state_are_the_declared_ones(audit):
    """A new unseen slow, or a fixed one, moves this table rather than passing quietly."""
    assert {(champion, slot) for champion, slot, _kind in audit.control_missing} == set(
        UNSEEN_SLOWS
    ), audit.table()


def test_the_hard_cc_the_reviewed_set_denies_is_the_declared_set(audit):
    """The precision budget is an exact set, so drift in either direction fails."""
    assert set(audit.hard_where_none) == set(HARD_WHERE_REVIEWED_NONE), audit.table()


def test_every_control_label_in_the_export_is_ruled(wiki_data):
    """The shipped export carries no control-family label the table has not decided."""
    audit_family_labels(wiki_data)


def test_an_unruled_control_label_raises_instead_of_sourcing_zero(wiki_data):
    """A renamed wiki row must stop the build, never quietly leave the dim at zero."""
    champion = next(iter(wiki_data))
    injected = {
        champion: {
            "abilities": {
                "Q": [{"effects": [{"leveling": [{"attribute": "Rooting Grip Duration"}]}]}]
            }
        }
    }
    with pytest.raises(UnmappedCcLabel, match="Rooting Grip Duration"):
        audit_family_labels(injected)


def test_the_artifact_carries_the_sourced_slow_percent(wiki_data):
    """Malphite Q's 40% slow reaches the feature vector; the prose never states it."""
    features = extract_all_features("Malphite", wiki_data["Malphite"])
    assert features["Q_cc_slow_pct"] == 40.0


def test_every_sourced_dim_is_a_schema_key():
    """A rule sourcing a dim the key list lacks would write into nothing."""
    assert set(SOURCED_DIMS) <= set(ABILITY_FEATURE_KEYS)


def test_a_slow_label_sources_its_percent():
    """Malphite Q states its slow only in the leveling row, never in the blurb."""
    sourced = cc_from_labels([("Magic Damage", 270.0, ""), ("Slow", 40.0, "%")])
    assert sourced["cc_slow_pct"] == 40.0


def test_a_slow_ap_coefficient_is_not_the_slow():
    """Nami E carries 'Slow' twice: 35 '%' and 5 '% per 100 AP'. Only the percent counts."""
    sourced = cc_from_labels([("Slow", 35.0, "%"), ("Slow", 5.0, "% per 100 AP")])
    assert sourced["cc_slow_pct"] == 35.0


def test_an_undeclared_unit_on_a_ruled_label_raises():
    """A ruled label whose unit column changes must stop the build, not source zero."""
    with pytest.raises(UnmappedCcUnit, match="Slow"):
        cc_from_labels([("Slow", 35.0, "% per 200 AP")])


def test_a_stun_duration_label_sources_seconds_and_kind():
    """A sourced duration row is stronger evidence than any sentence about it."""
    sourced = cc_from_labels([("Stun Duration", 1.5, " seconds")])
    assert sourced["cc_hard_seconds"] == 1.5
    assert sourced["cc_type_stun"] == 1.0


def test_a_level_scaled_duration_reads_its_longest_rank(wiki_data):
    """The export writes '1 : 3.5 (based on distance travelled) seconds'; max rank is 3.5."""
    assert extract_all_features("Ashe", wiki_data["Ashe"])["R_cc_hard_seconds"] == 3.5
    assert extract_all_features("Vex", wiki_data["Vex"])["P_cc_hard_seconds"] == 1.5


def test_knock_back_up_to_a_distance_is_not_a_knockup():
    """'knocked back up to 300 units' states a knockback distance, never airborne CC."""
    cc = extract_cc_from_text(
        ["Enemies are knocked back up to 300 units based on their proximity."]
    )
    assert cc["cc_type_knockback"] == 1.0
    assert cc["cc_type_knockup"] == 0.0


def test_the_target_of_a_minion_clause_is_not_a_champion():
    """Nunu Q devours a minion; 'the target minion' must not read as champion CC."""
    cc = extract_cc_from_text(
        [
            "If Consume would kill the target minion or small or medium monster, "
            "Willump devours the target, stunning and pulling them towards him."
        ]
    )
    assert cc["cc_type_stun"] == 0.0
    assert cc["cc_type_pull"] == 0.0


def test_a_self_or_ally_recipient_raises_no_enemy_cc():
    """Ryze R roots Ryze and his allies; self-applied control is not a draft's lockdown."""
    cc = extract_cc_from_text(
        [
            "Ryze and all allied units within the portal will blink to the location "
            "and become rooted, disarmed, silenced and untargetable for 0.75 seconds."
        ]
    )
    assert cc["cc_type_root"] == 0.0
    assert cc["cc_type_silence"] == 0.0
    assert cc["cc_hard_seconds"] == 0.0


def test_an_exemption_in_brackets_does_not_swallow_the_cc_after_it():
    """Rell Q states its stun after '(excluding the shields of monsters)', with no comma."""
    cc = extract_cc_from_text(
        [
            "Rell destroys the shields of all enemies hit (excluding the shields of "
            "monsters) before dealing them magic damage and stunning them for 0.65 seconds."
        ]
    )
    assert cc["cc_type_stun"] == 1.0
    assert cc["cc_hard_seconds"] == 0.65


def test_a_recipient_named_once_carries_across_the_description():
    """Warwick R names its victim in sentence one and suppresses 'them' in sentence two."""
    cc = extract_cc_from_text(
        [
            "Warwick leaps in the target direction, stopping upon hitting an enemy champion. "
            "He then knocks them down and channels for up to 1.5 seconds to suppress them."
        ]
    )
    assert cc["cc_type_suppress"] == 1.0


def test_an_ally_only_description_applies_no_enemy_cc():
    """Kalista R pulls her Oathsworn; a description naming no enemy applies no enemy CC."""
    cc = extract_cc_from_text(
        [
            "Kalista cleanses the Oathsworn from all crowd control and renders them "
            "invulnerable and untargetable for the duration, and pulls them to her."
        ]
    )
    assert cc["cc_type_pull"] == 0.0


def test_a_self_declared_affects_field_does_not_deny_stated_enemy_cc(wiki_data):
    """Poppy W is filed 'Self' yet knocks up enemies; the sentence outranks the field."""
    features = extract_all_features("Poppy", wiki_data["Poppy"])
    assert features["W_cc_type_knockup"] == 1.0


def test_knocking_a_target_away_or_aside_is_a_knockback(wiki_data):
    """Anivia W, Camille R and Taliyah R say away/aside, never the word 'back'."""
    for champion, slot in (("Anivia", "W"), ("Camille", "R"), ("Taliyah", "R")):
        features = extract_all_features(champion, wiki_data[champion])
        assert features[f"{slot}_cc_type_knockback"] == 1.0, f"{champion} {slot}"


def test_a_stated_displacement_distance_is_a_knockback():
    """Thresh E knocks enemies a distance without naming a direction word."""
    cc = extract_cc_from_text(
        ["Enemy champions hit are dealt damage and knocked 200 units in the target direction."]
    )
    assert cc["cc_type_knockback"] == 1.0


def test_suspending_an_enemy_is_airborne(wiki_data):
    """Nami Q suspends; the wiki's airborne wording for that ability is never 'knock up'."""
    features = extract_all_features("Nami", wiki_data["Nami"])
    assert features["Q_cc_type_knockup"] == 1.0


def test_a_decaying_slow_states_its_duration():
    """Braum Q and Lulu Q write 'decaying over N seconds', never 'for N seconds'."""
    texts = [
        "Braum deals magic damage to the first enemy hit, "
        "slowing them by 70% decaying over 2 seconds."
    ]
    assert extract_slow_duration(texts) == 2.0


def test_a_slow_with_a_named_area_states_its_duration():
    """Jhin E puts a noun phrase between the verb and its percent."""
    assert extract_slow_duration(["Traps slow enemies within the area by 35% for 2 seconds"]) == 2.0


def test_a_slow_on_minions_only_is_not_a_champion_slow():
    """Darius R and Aatrox R slow the jungle and the wave; a draft reads nothing from that."""
    assert (
        extract_slow_duration(
            [
                "Darius fears nearby minions and monsters for 3 seconds, during which "
                "they are gradually slowed by up to 99% over the duration."
            ]
        )
        == 0.0
    )


def test_a_damage_modifier_against_minions_leaves_the_slow_standing():
    """Lulu Q names minions only to halve their damage; the slow still hits champions."""
    texts = [
        "Each bolt deals magic damage to enemies hit, reduced to 70% against minions, "
        "and slows them by 80% decaying over 2 seconds."
    ]
    assert extract_slow_duration(texts) == 2.0


def test_a_slow_on_the_caster_is_not_a_champion_slow():
    """Blitzcrank W slows Blitzcrank when it ends; self-cost is not draft lockdown."""
    texts = ["When Overdrive ends, Blitzcrank is slowed for 1.5 seconds."]
    assert extract_slow_duration(texts) == 0.0


def test_a_silence_alone_is_hard_cc():
    """Blitzcrank R only silences; a silence with no other control still locks a fight down."""
    cc = extract_cc_from_text(
        ["It deals enemy champions magic damage and silences them for 0.5 seconds."]
    )
    assert cc["cc_type_silence"] == 1.0
    assert cc["cc_hard_seconds"] == 0.5


def test_an_excluded_monster_does_not_make_a_sentence_non_champion():
    """Blitzcrank R names monsters only to exempt them; its silence still hits champions."""
    cc = extract_cc_from_text(
        [
            "Blitzcrank deals damage to all nearby enemies, excluding monsters, "
            "and silences them for 0.5 seconds."
        ]
    )
    assert cc["cc_type_silence"] == 1.0


def test_a_state_condition_is_not_an_applied_cc():
    """Bard E names grounding as a restriction on casting it, not an effect it applies."""
    cc = extract_cc_from_text(
        ["A champion cannot take the corridor while immobilized or grounded."]
    )
    assert cc["cc_type_ground"] == 0.0
    assert cc["cc_type_root"] == 0.0


def test_a_trigger_clause_is_not_an_applied_cc():
    """Janna E refunds cooldown when her OTHER abilities knock up; E knocks nobody up."""
    cc = extract_cc_from_text(
        [
            "Whenever Janna's abilities slow or knock up at least one enemy champion, "
            "20% of Eye of the Storm's cooldown is refunded."
        ]
    )
    assert cc["cc_type_knockup"] == 0.0


def test_a_state_condition_does_not_swallow_the_cc_that_follows_it():
    """Lee Sin R names airborne as a condition, then applies a knock-up in the same breath."""
    cc = extract_cc_from_text(
        [
            "Enemy champions that collide with the displaced enemy while it is airborne "
            "take the same damage and are knocked up for 1 second."
        ]
    )
    assert cc["cc_type_knockup"] == 1.0
    assert cc["cc_hard_seconds"] == 1.0


def test_a_cancel_condition_on_the_caster_is_not_an_applied_cc():
    """Kayn E, Akshan E and Quinn R all list what interrupts them, not what they inflict."""
    cc = extract_cc_from_text(
        [
            "The effect on the enemy champion cancels immediately if Kayn performs a "
            "basic attack or becomes immobilized or polymorphed."
        ]
    )
    assert cc["cc_type_polymorph"] == 0.0
    assert cc["cc_type_ground"] == 0.0


def test_an_attributive_participle_is_not_an_applied_cc():
    """Sejuani P marks enemies another ability stunned; the passive stuns nobody."""
    cc = extract_cc_from_text(["Enemy champions stunned by Sejuani are marked Frozen."])
    assert cc["cc_type_stun"] == 0.0


def test_a_negated_cc_term_is_not_an_applied_cc():
    """Zoe W targets the nearest non-sleeping enemy; the negation applies no sleep."""
    cc = extract_cc_from_text(["She shoots at the nearest non- sleeping enemy champion in range."])
    assert cc["cc_type_sleep"] == 0.0


def test_a_duration_outside_the_cc_clause_is_not_cc_seconds():
    """Trundle E's 6 seconds is the pillar's lifetime, not a knockback's duration."""
    cc = extract_cc_from_text(
        [
            "Trundle erects a pillar of ice at the target location for 6 seconds, "
            "which knocks back enemy champions hit up to 225 units from its center."
        ]
    )
    assert cc["cc_hard_seconds"] == 0.0
