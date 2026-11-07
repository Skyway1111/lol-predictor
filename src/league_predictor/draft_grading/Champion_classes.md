# Machineloling Champion Class Trees — transcribed 2026-05-15

Source: [machineloling.com/2026/05/05/we-used-ai-to-discover-the-hidden-topology-behind-champion-identities/](https://machineloling.com/2026/05/05/we-used-ai-to-discover-the-hidden-topology-behind-champion-identities/) (Diamond+ Season 16, NN embeddings + Ward clustering on per-role dendrograms).

## Naming convention

The article's NN clusters did **not** have human-readable names — they're cluster IDs from a similarity ranking. Class labels here are descriptive names coined for this transcription from each cluster's champion membership. To rename one, do it in one place (this file); consumers parse the labels from here.

## Visual cuts vs article's class counts

Per the article's [Definitive Champion Class Encyclopedia](https://machineloling.com/2026/04/13/the-definitive-season-16-champion-class-encyclopedia/), per-role class counts are: Top=4, Jungle=8, Mid=7, ADC=4, Support=5 (29 total). This transcription enumerates **38 visual clusters** total (Jng=8, Top=7, Mid=9, Bot=7, Sup=7) — slightly finer than the article's aggregation in four of the five roles. This is consistent with the dendrograms displaying sub-cluster colorings within the larger article-cut classes. Two ways to consume them:

- **Use the visual-cut classes (38 total)** for maximum granularity.
- **Merge to article-cut counts** by collapsing the finest sub-groups into their parent cluster (e.g., merge Mid's "Smolder/Swain" white outlier with its nearest neighbor; merge Top's "Lane_Bullies" + "Zone_Controllers" if they share the article's "Anti-Ranged Tank" parent).

Default: visual-cut granularity. Drop a class to its parent only if feature importance shows the fine split is noise.

---

## Jungle — 8 clusters

| ID | Color | Coined label | Champions |
|---|---|---|---|
| J1 | green | **Heavy_Fighters** | Hecarim, Diana, Shyvana, Udyr, Lillia, Sylas, Nunu, Gwen, Volibear |
| J2 | teal | **Ult_Tanks** | Zac, Malphite, Amumu, Rammus, Nocturne, DrMundo, Sejuani |
| J3 | pink | **Bruisers** | Briar, Ambessa, Aatrox, Kayn_Red, Zaahen |
| J4 | blue | **Divers** | Warwick, Jax, MonkeyKing, LeeSin, RekSai, JarvanIV, Vi, XinZhao |
| J5 | purple | **AP_Skirmishers** | Elise, Evelynn, Ekko, Nidalee, Taliyah, Karthus, Zyra, Shaco_Box, FiddleSticks, Ivern |
| J6 | cyan | **Hypercarries** | Graves, Kindred, MasterYi, Viego |
| J7 | white | **AD_Assassins** | Jayce, Qiyana, Talon, Zed, Khazix, Rengar |
| J8 | yellow | **Late_Scaling_Assassins** | Belveth, Shaco_Stab, Kayn_Blue, Naafiri |

53 entries (Kayn and Shaco split across builds).

---

## Top — 7 clusters

| ID | Color | Coined label | Champions |
|---|---|---|---|
| T1 | green | **Stat_Stick_Tanks** | Malphite, Nasus, Ornn, Chogath, TahmKench, DrMundo, Sion |
| T2 | cyan | **AP_Topside** | Sylas, Gragas, Ryze, Akali, Vladimir, Kennen, Teemo |
| T3 | white | **Flex_Hybrids** | Rumble, Swain, Varus, Gwen, Mordekaiser |
| T4 | purple | **Brawlers** | Warwick, Jax, Volibear, Sett, Zaahen, Darius, Renekton |
| T5 | red | **Duelists** | Trundle, Tryndamere, Fiora, Riven, Aatrox, Ambessa, Yone, Yorick, Irelia, Yasuo |
| T6 | yellow | **Zone_Controllers** | Camille, Kled, Shen, Singed, Garen, Illaoi, KSante, Poppy |
| T7 | blue | **Lane_Bullies** | Gangplank, Olaf, Kayle, Quinn, Vayne, Gnar, Jayce, Pantheon, Urgot |

53 entries. Note "Lane_Bullies" mixes melee (Gangplank, Olaf, Pantheon, Urgot) with ranged (Kayle, Quinn, Vayne, Gnar, Jayce) — if this hurts feature signal, split on `range_type` from the recovered champion attributes.

---

## Mid — 9 clusters

| ID | Color | Coined label | Champions |
|---|---|---|---|
| M1 | purple | **Artillery_Mages** | AurelionSol, Velkoz, Xerath, Mel, Syndra, Lux, Veigar |
| M2 | white | **Scaling_Generalists** | Smolder, Swain |
| M3 | green | **Control_Mages** | Anivia, Hwei, Viktor, Annie, Malzahar, Galio, Lissandra, TwistedFate, Vex |
| M4 | blue | **Battle_Mages** | Orianna, Azir, Taliyah, Ryze, Cassiopeia, Vladimir |
| M5 | white (second) | **Wind_Skirmishers** | Yone, Irelia, Yasuo |
| M6 | cyan | **Mobile_Assassins** | Akshan, Naafiri, Qiyana, Talon, Zed |
| M7 | dark blue | **All_In_Assassins** | Katarina, Ekko, Fizz, Diana, Kassadin |
| M8 | light blue | **Pick_Assassins** | Sylas, Akali, Leblanc |
| M9 | pink | **Pick_Mages** | Aurora, Ahri, Zoe |

36 entries. M2 (Smolder, Swain) is suspicious as its own cluster — both are scalers that don't sit neatly in a mage archetype. Possibly merge with M1 (Artillery_Mages) or M3 (Control_Mages) per the article's 7-class cut.

---

## Bot — 7 clusters

| ID | Color | Coined label | Champions |
|---|---|---|---|
| B1 | blue | **Lethality_ADCs** | Ezreal, Varus_Lethality, KogMaw, Varus |
| B2 | green | **Mage_Bots** | Mel, Ziggs, Brand, Swain, Veigar |
| B3 | white | **Stack_Scalers** | Senna_Damage, Smolder |
| B4 | green (second) | **Burst_ADCs** | Corki, Sivir, Xayah |
| B5 | blue (second) | **Long_Range_ADCs** | Ashe, MissFortune, Caitlyn, Jhin |
| B6 | pink | **Hypercarry_ADCs** | Twitch, Draven, Aphelios, Jinx |
| B7 | blue (third) | **Mobile_ADCs** | Kaisa, Vayne, Lucian, Tristana, Yunara, Zeri, Nilah, Yasuo, Kalista, Samira |

32 entries. B7 is large (10 champions); the article's 4-class cut likely groups B1+B5 (range-based immobile/poke), B2+B3 (mage damage), B4 (burst), and B6+B7 (carry/mobile). The 7-way visual cut is kept; LightGBM can learn across the finer split.

---

## Support — 7 clusters

| ID | Color | Coined label | Champions |
|---|---|---|---|
| S1 | purple | **Pick_Supports** | Pyke, Pantheon, Poppy, Elise, TahmKench |
| S2 | yellow | **Engage_Tanks** | Blitzcrank, Thresh, Maokai, Leona, Nautilus |
| S3 | pink | **Wardens** | Braum, Taric, Alistar, Rakan, Rell |
| S4 | green | **Mage_Supports** | Morgana, Swain, Neeko, Brand, Zyra, Lux, Mel, Velkoz, Xerath |
| S5 | blue | **Utility_Damage** | Karma_Damage, Senna_Damage, Renata, Seraphine, Bard, Zilean |
| S6 | white | **Pure_Enchanters** | Sona, Senna_Enchanter, Yuumi |
| S7 | light blue | **Classic_Enchanters** | Karma_Enchanter, Lulu, Janna, Nami, Milio, Soraka |

40 entries. The article's 5-class Support cut (Warden, Playmaker, Mage, Cat, Enchanter) maps roughly: S3 → Warden, S1+S2 → Playmaker (?), S4+S5 → Mage, S6 (Yuumi cluster) → Cat, S7 → Enchanter. The visual 7-way split distinguishes between hard-engage and pick-style playmakers (S1 vs S2) and between pure-stat and utility-damage enchanters (S6 vs S5).

---

## Champion form / build variants

These champions are split across forms or builds (treat each as a distinct (champion, build) entry; the model needs to know which form was played):

| Champion | Variants | Roles affected |
|---|---|---|
| Kayn | `Kayn_Red` (Darkin), `Kayn_Blue` (Shadow Assassin) | Jungle |
| Shaco | `Shaco_Box` (AP/box), `Shaco_Stab` (AD assassin) | Jungle |
| Karma | `Karma_Damage`, `Karma_Enchanter` | Support |
| Senna | `Senna_Damage`, `Senna_Enchanter` | Bot, Support |
| Varus | `Varus`, `Varus_Lethality` | Bot |

The model uses dominant-form only: a base display name like `"Kayn"` resolves to its dominant pro form (`"Kayn_Red"`) via `normalize.FORM_DEFAULTS` before the class-table lookup.

## Champion role flexibility

Several champions appear in multiple role dendrograms with **different class assignments per role** — the per-role class field is genuinely needed:

| Champion | Roles | Notes |
|---|---|---|
| Swain | Top, Mid, Bot, Sup | All four roles. Different class label in each. |
| Sylas | Top, Jng, Mid | T2 (AP_Topside) / J1 (Heavy_Fighters) / M8 (Pick_Assassins) |
| Yasuo | Top, Mid, Bot | T5 (Duelists) / M5 (Wind_Skirmishers) / B7 (Mobile_ADCs) |
| Yone | Top, Mid | Same Wind_Skirmishers cluster in both |
| Mel | Mid, Bot, Sup | Different class each |
| Veigar | Mid, Bot, Sup | Different class each |
| Brand | Bot, Sup | Same class concept ("Mage_*") but in different role dendrograms |
| Varus | Top, Bot | T3 (Flex_Hybrids) / B1 (Lethality_ADCs) |
| Velkoz | Mid, Sup | M1 (Artillery_Mages) / S4 (Mage_Supports) |
| Xerath | Mid, Sup | M1 (Artillery_Mages) / S4 (Mage_Supports) |
| Lux | Mid, Sup | M1 (Artillery_Mages) / S4 (Mage_Supports) |
| Smolder | Mid, Bot | M2 (Scaling_Generalists) / B3 (Stack_Scalers) |
| Pantheon | Top, Sup | T7 (Lane_Bullies) / S1 (Pick_Supports) |
| Poppy | Top, Sup | T6 (Zone_Controllers) / S1 (Pick_Supports) |
| Elise | Jng, Sup | J5 (AP_Skirmishers) / S1 (Pick_Supports) |
| TahmKench | Top, Sup | T1 (Stat_Stick_Tanks) / S1 (Pick_Supports) |
| DrMundo | Top, Jng | T1 / J2 (Ult_Tanks) |
| Volibear | Top, Jng | T4 (Brawlers) / J1 (Heavy_Fighters) |
| Jax | Top, Jng | T4 (Brawlers) / J4 (Divers) |
| Warwick | Top, Jng | T4 (Brawlers) / J4 (Divers) |
| Diana | Mid, Jng | M7 (All_In_Assassins) / J1 (Heavy_Fighters) |
| Ekko | Mid, Jng | M7 (All_In_Assassins) / J5 (AP_Skirmishers) |
| Ryze | Top, Mid | T2 (AP_Topside) / M4 (Battle_Mages) |
| Vladimir | Top, Mid | T2 (AP_Topside) / M4 (Battle_Mages) |
| Akali | Top, Mid | T2 (AP_Topside) / M8 (Pick_Assassins) |
| Taliyah | Mid, Jng | M4 (Battle_Mages) / J5 (AP_Skirmishers) |
| Talon | Mid, Jng | M6 (Mobile_Assassins) / J7 (AD_Assassins) |
| Zed | Mid, Jng | M6 (Mobile_Assassins) / J7 (AD_Assassins) |
| Qiyana | Mid, Jng | M6 (Mobile_Assassins) / J7 (AD_Assassins) |
| Naafiri | Mid, Jng | M6 (Mobile_Assassins) / J8 (Late_Scaling_Assassins) |
| Karthus | Mid (presumably), Jng | J5 only in image set; Mid not shown |
| Zyra | Sup, Jng | S4 (Mage_Supports) / J5 (AP_Skirmishers) |
| Gwen | Top, Jng | T3 (Flex_Hybrids) / J1 (Heavy_Fighters) |
| Aatrox | Top, Jng | T5 (Duelists) / J3 (Bruisers) |
| Ambessa | Top, Jng | T5 (Duelists) / J3 (Bruisers) |
| Zaahen | Top, Jng | T4 (Brawlers) / J3 (Bruisers) |

## How this file is consumed

`draft_grading/champion_attributes/classes.py` parses this markdown into a
`(canonical_champion_name, role) -> class_label` dict. The parser walks
section headers (`## Jungle`, `## Top`, etc.) and table rows, reading the
bold label in the third pipe-delimited column and the comma-separated
champion list in the fourth. The result feeds:

- `CLASS_LABEL_TO_INDEX` — the 38 cluster labels sorted alphabetically,
  plus `UNKNOWN_CLASS` at the end, mapped to integer indices 0–38.
- `effective_class(display_name, role)` — the main lookup. Extends the
  direct table with two fills: `MANUAL_CLASS_OVERRIDES` (the two
  `CLASS_FILE_GAPS` champions, Heimerdinger and Skarner, hand-assigned
  per role into existing clusters) and cross-role borrowing (a champion
  clustered in some role but not the queried one inherits its primary
  role's class, so that champions known to flex still resolve cleanly).
- `champion_roles(display_name)` — the set of roles machineloling
  clustered this champion in; `CLASS_FILE_GAPS` champions return the
  empty set.

Parsing is lazy (first access via `ensure_loaded()`), so importing the
module is safe in minimal test environments where this file is absent.
