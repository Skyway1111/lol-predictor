"""Tests for ability-level feature extraction."""

# file-length-ok: one module's contract mirror, split adds cross-file hops

import pytest

from league_predictor.draft_grading.champion_attributes.ability_schema import (
    ALL_FEATURE_KEYS,
)
from league_predictor.draft_grading.champion_attributes.wiki_export import (
    DEFAULT_WIKI_EXPORT_PATH,
    load_wiki_champion_export,
)
from scripts.champion_attributes.ability_features import (
    extract_ability_features,
    extract_all_features,
    extract_champion_stats,
    extract_heal_amount,
    extract_leveling_entries,
    extract_slot_features,
    extract_slow_duration,
    max_rank_value,
    merge_multi_form,
    normalize_spell_effects,
    parse_numeric_string,
)


@pytest.fixture(scope="module")
def wiki_data():
    # DEFAULT_WIKI_EXPORT_PATH honours LOL_WIKI_CHAMPIONS_JSON, else the in-repo export.
    return load_wiki_champion_export(DEFAULT_WIKI_EXPORT_PATH)


# --- parse_numeric_string ---


# --- max_rank_value ---


def test_max_rank_value_basic():
    mods = [{"values": [20, 45, 70, 95, 120], "units": ["", "", "", "", ""]}]
    assert max_rank_value(mods) == 120.0


def test_max_rank_value_ult():
    mods = [{"values": [150, 250, 350], "units": ["", "", ""]}]
    assert max_rank_value(mods) == 350.0


def test_max_rank_value_empty():
    assert max_rank_value([]) == 0.0


# --- normalize_spell_effects ---


def test_normalize_aoe_variants():
    assert normalize_spell_effects("AoE") == "aoe"
    assert normalize_spell_effects("AOE") == "aoe"
    assert normalize_spell_effects("Area of Effect") == "aoe"
    assert normalize_spell_effects("spellaoe") == "aoe"


def test_normalize_dot():
    assert normalize_spell_effects("AoEDoT") == "dot"
    assert normalize_spell_effects("periodic") == "dot"


def test_normalize_none():
    assert normalize_spell_effects(None) == "none"


# --- Ornn Q integration (real wiki data) ---


def test_ornn_q_structured_fields(wiki_data):
    ornn = wiki_data["Ornn"]
    q_ability = ornn["abilities"]["Q"][0]
    features = extract_ability_features(q_ability)

    assert features["damage_type_physical"] == 1.0
    assert features["is_skillshot"] == 1.0
    assert features["target_range"] == 750.0
    assert features["width"] == 130.0
    assert features["base_damage"] == 120.0
    assert features["scaling_ad"] == 110.0


def test_ornn_w_percent_max_hp(wiki_data):
    ornn = wiki_data["Ornn"]
    w_ability = ornn["abilities"]["W"][0]
    features = extract_ability_features(w_ability)

    assert features["damage_type_magic"] == 1.0
    assert features["scaling_max_hp"] > 0, "Ornn W should have %maxHP scaling"
    assert features["has_unstoppable"] == 1.0, "Ornn W grants displacement immunity"


# --- Multi-form merge (Nidalee Q) ---


def test_merge_nidalee_q(wiki_data):
    nidalee = wiki_data["Nidalee"]
    q_forms = nidalee["abilities"]["Q"]
    assert len(q_forms) >= 2, "Nidalee should have human + cougar Q"

    form_features = [extract_ability_features(form) for form in q_forms]
    merged = merge_multi_form(form_features)

    # Max damage from either form
    assert merged["base_damage"] == max(f["base_damage"] for f in form_features)
    # Max range from either form (human Q has long range)
    assert merged["target_range"] == max(f["target_range"] for f in form_features)
    # Min cooldown from either form
    nonzero_cds = [f["cooldown"] for f in form_features if f["cooldown"] > 0]
    if nonzero_cds:
        assert merged["cooldown"] == min(nonzero_cds)


# --- Champion stats ---


def test_ornn_champion_stats(wiki_data):
    ornn = wiki_data["Ornn"]
    stats = extract_champion_stats(ornn)

    assert stats["hp_base"] == 660.0
    assert stats["hp_per_level"] == 109.0
    assert stats["is_melee"] == 1.0
    assert stats["is_transform"] == 0.0
    assert stats["resource_mana"] == 1.0


def test_nidalee_is_transform(wiki_data):
    nidalee = wiki_data["Nidalee"]
    stats = extract_champion_stats(nidalee)
    assert stats["is_transform"] == 1.0
    assert stats["is_melee"] == 0.0  # base form is ranged


# --- Full extraction ---


def test_all_features_correct_keys(wiki_data):
    ornn = wiki_data["Ornn"]
    features = extract_all_features("Ornn", ornn)

    assert set(features.keys()) == set(ALL_FEATURE_KEYS)
    assert len(features) == len(ALL_FEATURE_KEYS)


def test_maokai_vs_ornn_r_distinguishable(wiki_data):
    """The motivating test: Maokai R ≠ Ornn R despite both being Vanguards."""
    maokai_f = extract_all_features("Maokai", wiki_data["Maokai"])
    ornn_f = extract_all_features("Ornn", wiki_data["Ornn"])

    # Ornn R has knockup, Maokai R has root
    assert ornn_f["R_cc_type_knockup"] == 1.0
    assert maokai_f["R_cc_type_knockup"] == 0.0
    assert maokai_f["R_cc_type_root"] == 1.0
    assert ornn_f["R_cc_type_root"] == 0.0

    # Different range profiles
    assert ornn_f["R_effect_radius"] > maokai_f["R_effect_radius"]
    assert maokai_f["R_target_range"] > ornn_f["R_target_range"]


def test_slow_duration_extraction():
    texts = ["Deals damage to enemy champions hit and slows them by 40% for 2 seconds."]
    assert extract_slow_duration(texts) == 2.0


# --- Schema v4: tank dims (six-tank reference fixtures, real wiki data) ---


def _slot(wiki_data, champ: str, slot: str) -> dict[str, float]:
    return extract_slot_features(wiki_data[champ]["abilities"][slot])


def test_malphite_w_armor_steroid_and_ratio(wiki_data):
    w = _slot(wiki_data, "Malphite", "W")
    assert w["armor_gain_pct"] == 90.0  # 'Increased Bonus Armor' = 90 '% armor'
    assert w["armor_gain_flat"] == 0.0  # his steroid is %-of-own-armor, not flat
    assert w["scaling_armor"] == 15.0  # damage entries carry 15 '% armor'


def test_malphite_e_armor_ratio(wiki_data):
    assert _slot(wiki_data, "Malphite", "E")["scaling_armor"] == 40.0


def test_malphite_p_percent_hp_shield(wiki_data):
    # "a shield equal to 10% of his maximum health" — text-only on the wiki
    assert _slot(wiki_data, "Malphite", "P")["shield_hp_pct"] == 10.0


def test_chogath_r_health_stacking(wiki_data):
    r = _slot(wiki_data, "Chogath", "R")
    assert r["hp_gain_flat"] == 160.0  # 'Bonus Health Per Stack'
    assert r["scaling_bonus_hp"] == 10.0  # true damage scales with own bonus health
    assert r["heal_amount"] == 0.0  # a health GRANT is not a heal


def test_nasus_r_bonus_health_is_not_a_heal(wiki_data):
    r = _slot(wiki_data, "Nasus", "R")
    assert r["hp_gain_flat"] == 600.0  # 'Bonus Health' grant
    assert r["heal_amount"] == 0.0  # 'Bonus Health' contains 'heal' as a substring


def test_heal_label_word_bounded():
    # 'heal' must match as a word ('Bonus Health' never books); restore stems count.
    assert extract_heal_amount([("Heal", 190.0, "")]) == 190.0
    assert extract_heal_amount([("Total Healing", 240.0, "")]) == 240.0
    assert extract_heal_amount([("Health Restored", 600.0, "")]) == 600.0
    assert extract_heal_amount([("Bonus Health", 600.0, "")]) == 0.0
    assert extract_heal_amount([("Healing Reduction", 40.0, "")]) == 0.0


def test_sion_w_shield_polarity(wiki_data):
    w = _slot(wiki_data, "Sion", "W")
    assert w["shield_hp_pct"] == 16.0  # 'Shield Strength' = 16 '% maximum health'
    # Polarity fix: the own-HP shield must NOT book as anti-tank scaling; the
    # damage entry (14 "% of target's maximum health") is the real scaling_max_hp.
    assert w["scaling_max_hp"] == 14.0


def test_sion_e_armor_shred(wiki_data):
    # source text: inflicts them with 25 percent armor reduction
    assert _slot(wiki_data, "Sion", "E")["armor_shred_pct"] == 25.0


def test_ksante_q_resist_ratios(wiki_data):
    q = _slot(wiki_data, "KSante", "Q")
    assert q["scaling_armor"] == 40.0  # '% bonus armor'
    assert q["scaling_mr"] == 40.0  # '% bonus magic resistance'


def test_ksante_e_bonus_hp_shield(wiki_data):
    assert _slot(wiki_data, "KSante", "E")["shield_hp_pct"] == 13.5


def test_ksante_w_damage_reduction(wiki_data):
    # source text: gains displacement immunity and 30 percent damage reduction
    assert _slot(wiki_data, "KSante", "W")["damage_reduction_pct"] == 30.0


def test_alistar_r_damage_reduction(wiki_data):
    # 'Damage Reduction' = 75 '%' — structured leveling entry
    assert _slot(wiki_data, "Alistar", "R")["damage_reduction_pct"] == 75.0


def test_tryndamere_w_ad_debuff_is_not_damage_reduction(wiki_data):
    # 'Attack Damage Reduction' = 80 is an enemy AD debuff, not incoming-DR
    assert _slot(wiki_data, "Tryndamere", "W")["damage_reduction_pct"] == 0.0


def test_leona_w_resist_grants(wiki_data):
    w = _slot(wiki_data, "Leona", "W")
    assert w["armor_gain_flat"] == 50.0
    assert w["mr_gain_flat"] == 50.0
    assert w["armor_gain_pct"] == 20.0
    assert w["mr_gain_pct"] == 20.0


def test_non_tank_new_dims_stay_zero(wiki_data):
    """Ashe has no tank kit: every v4 dim must stay 0 across all slots."""
    v4_keys = (
        "armor_gain_flat",
        "armor_gain_pct",
        "mr_gain_flat",
        "mr_gain_pct",
        "hp_gain_flat",
        "damage_reduction_pct",
        "shield_hp_pct",
        "scaling_armor",
        "scaling_mr",
        "scaling_bonus_hp",
        "armor_shred_pct",
    )
    features = extract_all_features("Ashe", wiki_data["Ashe"])
    nonzero = {
        f"{slot}_{k}": features[f"{slot}_{k}"]
        for slot in ("P", "Q", "W", "E", "R")
        for k in v4_keys
        if features[f"{slot}_{k}"] != 0.0
    }
    assert not nonzero, f"unexpected tank dims on Ashe: {nonzero}"


def test_slow_duration_no_match():
    texts = ["deals 100 damage"]
    assert extract_slow_duration(texts) == 0.0


def test_extract_leveling_entries_structure(wiki_data):
    ornn = wiki_data["Ornn"]
    effects = ornn["abilities"]["Q"][0].get("effects", [])
    entries = extract_leveling_entries(effects)
    assert len(entries) > 0
    attr, val, unit = entries[0]
    assert isinstance(attr, str)
    assert isinstance(val, float)
    assert isinstance(unit, str)


# --- The shipped artifact resolves aliases (moved from the retired attribute store) ---


def test_shipped_artifact_resolves_alias_and_punctuation():
    """ability_features.json resolves punctuated names and Riot-internal aliases."""
    from league_predictor.draft_grading.champion_attributes.ability_store import (
        AbilityFeatureStore,
    )

    store = AbilityFeatureStore()
    assert store.resolve_name("dr. mundo") == "Dr. Mundo"  # punctuation + spacing
    assert store.resolve_name("kaisa") == "Kai'Sa"  # dropped apostrophe
    assert store.resolve_name("MonkeyKing") == "Wukong"  # Riot-internal alias
    assert store.resolve_name("NotAChampion") is None


# --- Schema v6: key layout ---


def test_damage_shares_transcribed(wiki_data):
    features = extract_all_features("Gwen", wiki_data["Gwen"], damage_shares=(0.13, 0.71, 0.16))
    assert features["physical_damage_share"] == pytest.approx(0.13)
    assert features["magic_damage_share"] == pytest.approx(0.71)
    assert features["true_damage_share"] == pytest.approx(0.16)


def test_damage_share_resolution_cache_then_fallback(wiki_data):
    """Offline resolution: cache hit for Gwen; a miss uses the stat heuristic, never network."""
    from scripts.champion_attributes.damage_shares import (
        fallback_damage_shares,
        load_damage_profile_cache,
    )
    from scripts.champion_attributes.generate_abilities import (
        resolve_damage_shares,
    )

    cache = load_damage_profile_cache()
    shares = resolve_damage_shares(cache, "Gwen", wiki_data["Gwen"])
    assert shares == pytest.approx((0.1291, 0.7077, 0.1632))

    fallback = resolve_damage_shares({"profiles": {}}, "Gwen", wiki_data["Gwen"])
    assert fallback == pytest.approx(fallback_damage_shares(wiki_data["Gwen"]))
    assert sum(fallback) == pytest.approx(1.0)


# --- The golden mechanic set (schema v3 extraction, over the real export) ---

#: (champion export key, slot-prefixed feature, truthiness) — the plan's pinned table.
GOLDEN_MUST_FLAG = [
    ("Gwen", "W_has_untargetable"),
    ("Kindred", "R_has_invulnerable"),
    ("Bard", "R_has_stasis"),
    ("Sivir", "E_has_spellshield"),
    ("Nocturne", "W_has_spellshield"),
    # cc-immunity shields count, per the pinned has_spellshield definition.
    ("Morgana", "E_has_spellshield"),
    ("Yasuo", "W_has_projectile_block"),
    ("Braum", "E_has_projectile_block"),
    ("Samira", "W_has_projectile_block"),
    ("Zilean", "R_has_revive"),
    # "enters resurrection" with a fatal-damage grant IS an owned revive.
    ("Zac", "P_has_revive"),
    # "will be resurrected" (completed form) IS an owned revive.
    ("Akshan", "W_has_revive"),
    ("Alistar", "R_has_damage_reduction"),
    ("Ahri", "E_cc_type_charm"),
    ("Shen", "E_cc_type_taunt"),
    ("Fiddlesticks", "Q_cc_type_fear"),
    ("Zoe", "E_cc_type_sleep"),
    ("Lulu", "W_cc_type_polymorph"),
    ("Alistar", "W_cc_type_knockback"),
    ("Blitzcrank", "Q_cc_type_pull"),
    ("Cassiopeia", "W_cc_type_ground"),
    ("Rammus", "Q_has_ms_boost"),
    ("Kalista", "R_has_untargetable"),
    ("Kalista", "R_has_invulnerable"),
    ("Kalista", "R_affects_allies"),
]

GOLDEN_MUST_NOT_FLAG = [
    ("Gwen", "W_has_invulnerable"),
    # charm/taunt/fear/sleep left the silence bucket (silence = 'silence' only).
    ("Ahri", "E_cc_type_silence"),
    ("Shen", "E_cc_type_silence"),
    ("Fiddlesticks", "Q_cc_type_silence"),
    ("Zoe", "E_cc_type_silence"),
    ("Alistar", "W_cc_type_knockup"),
    # Ally reposition is not enemy CC.
    ("Kalista", "R_cc_type_pull"),
    # The SOLDIER is untargetable, not Azir.
    ("Azir", "W_has_untargetable"),
    # "enters resurrection" as a trigger/condition (reacting to an EXTERNAL
    # revive) is not an owned revive.
    ("Aurora", "R_has_revive"),
    ("Yone", "E_has_revive"),
]


@pytest.fixture(scope="module")
def golden_features(wiki_data):
    keys = sorted({key for key, _feat in GOLDEN_MUST_FLAG + GOLDEN_MUST_NOT_FLAG} | {"Karthus"})
    return {key: extract_all_features(key, wiki_data[key]) for key in keys}


@pytest.mark.parametrize(("champion", "feature"), GOLDEN_MUST_FLAG)
def test_golden_must_flag(golden_features, champion, feature):
    assert golden_features[champion][feature] == 1.0, f"{champion}.{feature} must flag"


@pytest.mark.parametrize(("champion", "feature"), GOLDEN_MUST_NOT_FLAG)
def test_golden_must_not_flag(golden_features, champion, feature):
    assert golden_features[champion][feature] == 0.0, f"{champion}.{feature} must NOT flag"


def test_golden_lulu_polymorph_counts_hard_cc(golden_features):
    """Polymorph durations count (the Disable Duration leveling entry backs the verb)."""
    assert golden_features["Lulu"]["W_cc_hard_seconds"] > 0


def test_golden_karthus_global_sentinel(golden_features):
    """parse_numeric_string maps Karthus R's 'Global' effectRadius to 25000."""
    assert golden_features["Karthus"]["R_effect_radius"] == 25000.0


#: Every NEW schema-v3 key: ≥1 champion (a 0-roster key is a dead rule) and ≤60
#: (a 100+ roster is a false-positive epidemic).
NEW_SLOT_KEYS = (
    "cc_type_charm",
    "cc_type_taunt",
    "cc_type_fear",
    "cc_type_sleep",
    "cc_type_knockback",
    "cc_type_pull",
    "cc_type_ground",
    "cc_type_disarm",
    "cc_type_polymorph",
    "has_untargetable",
    "has_invulnerable",
    "has_stasis",
    "has_spellshield",
    "has_projectile_block",
    "has_damage_reduction",
    "has_revive",
    "affects_allies",
    "has_ms_boost",
)


def test_new_key_rosters_are_sane(wiki_data):
    """Roster floor/ceiling over the whole export for every new per-slot key."""
    all_features = {key: extract_all_features(key, champ) for key, champ in wiki_data.items()}
    slots = ("P", "Q", "W", "E", "R")
    for base in NEW_SLOT_KEYS:
        roster = sorted(
            name
            for name, feats in all_features.items()
            if any(feats[f"{slot}_{base}"] == 1.0 for slot in slots)
        )
        assert len(roster) >= 1, f"{base}: dead rule (no champion flags it)"
        assert len(roster) <= 60, (
            f"{base}: false-positive epidemic ({len(roster)} champions: {roster[:12]}...)"
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("800", 800.0),
        ("0.25", 0.25),
        ("800 / Up to 1340", 800.0),
        ("Global", 25000.0),
        (None, 0.0),
        ("Varied", 0.0),
        (550.0, 550.0),
    ],
)
def test_parse_numeric_string(value, expected):
    assert parse_numeric_string(value) == expected
