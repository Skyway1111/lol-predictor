"""The frozen champion knowledge column order and artifact schema."""

ABILITY_SLOTS = ("P", "Q", "W", "E", "R")

ABILITY_FEATURE_KEYS: tuple[str, ...] = (
    "base_damage",
    "damage_type_physical",
    "damage_type_magic",
    "damage_type_true",
    "scaling_ad",
    "scaling_ap",
    "scaling_max_hp",
    "scaling_missing_hp",
    "heal_amount",
    "shield_amount",
    "cc_hard_seconds",
    "cc_slow_duration",
    "cc_type_knockup",
    "cc_type_stun",
    "cc_type_root",
    "cc_type_suppress",
    "cc_type_silence",
    "cooldown",
    "cost",
    "cast_time",
    "target_range",
    "effect_radius",
    "width",
    "speed",
    "is_aoe",
    "is_dot",
    "is_projectile",
    "is_skillshot",
    "is_targeted",
    "is_passive",
    "has_dash",
    "has_unstoppable",
    "has_stealth",
    "has_terrain",
    "has_execute",
    "has_stacking",
    "has_persistent_aoe",
    # --- schema v3: CC types (the text names the recipient; charm/taunt/fear/sleep
    # durations already count via text_mining._HARD_CC_VERBS) ---
    "cc_type_charm",
    "cc_type_taunt",
    "cc_type_fear",
    "cc_type_sleep",
    "cc_type_knockback",
    "cc_type_pull",
    "cc_type_ground",
    "cc_type_disarm",
    "cc_type_polymorph",
    # --- schema v3: the denial family (champion/ally recipient required) ---
    "has_untargetable",
    "has_invulnerable",
    "has_stasis",
    "has_spellshield",
    "has_projectile_block",
    "has_damage_reduction",
    "has_revive",
    # --- schema v3: utility ---
    "affects_allies",
    "has_ms_boost",
    # --- schema v4: tank identity (offensive scaling_* stay damage-labeled and
    # target-polarity; this family reads the defensive grant labels) ---
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
    # --- schema v5: the slow's sourced strength, read by exact leveling label
    # (cc_labels.LABEL_RULES); cc_slow_duration stays the prose-and-label seconds ---
    "cc_slow_pct",
)

CHAMPION_FEATURE_KEYS: tuple[str, ...] = (
    "hp_base",
    "hp_per_level",
    "armor_base",
    "armor_per_level",
    "mr_base",
    "mr_per_level",
    "ad_base",
    "ad_per_level",
    "as_base",
    "as_per_level",
    "ms_base",
    "attack_range",
    "is_melee",
    "is_transform",
    "resource_mana",
    "resource_energy",
    "resource_none",
    # LeagueOfGraphs observed damage split (cache → stat fallback; damage_shares.py).
    "physical_damage_share",
    "magic_damage_share",
    "true_damage_share",
    # --- schema v6: the reviewed contract's penetration, stated per slot and carried
    # once per champion (reviewed_kit.CHAMPION_LEVEL_DIMS); prose sources none of it ---
    "armor_pen_flat",
    "armor_pen_pct",
    "magic_pen_flat",
    "magic_pen_pct",
    "lethality",
)

SCHEMA_VERSION = 6

ALL_FEATURE_KEYS: tuple[str, ...] = CHAMPION_FEATURE_KEYS + tuple(
    f"{slot}_{key}" for slot in ABILITY_SLOTS for key in ABILITY_FEATURE_KEYS
)
