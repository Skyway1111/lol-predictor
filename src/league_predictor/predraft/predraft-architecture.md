# Predraft architecture

Predraft owns the served predictor, the skill filter behind the `rating` model, and
analytical Impact. The lifecycle package owns the artifact swap and transactions. [Ratings architecture](ratings/ratings-architecture.md) maps
the skill filter, region rating, and Impact modules.

## Module ownership

| Module | Responsibility |
| --- | --- |
| `cleaning.py` | Completed-map eligibility, remake detection, and enriched facts reads. Series identity comes from ingestion. |
| `league_tier.py` | League spelling, tier, domestic pool, and region normalization. |
| `features/manifest.py` | Original-family columns, pruning rules, and feature-schema identity. |
| `features/snapshots/` | Team timelines and player prefixes, queried at a map's `AsOf` position. |
| `features/families/original.py` | Persisted Original-family indexes over those snapshots. |
| `features/families/recency.py` | Walk-forward form and momentum features. |
| `features/latent_strength.py` | Features from the live `rating` bag. |
| `features/lineup.py` | Day-strict lineup composition and inactivity. |
| `pipeline/chosen_model.py` | Served recipe, fitted artifact, artifact-bound recipe reads, and fitting. |
| `pipeline/lightgbm_bag.py` | Typed feature frames and the concurrent LightGBM seed bag. |
| `pipeline/walk_forward.py` | Shared family indexes and the chronological feature corpus. |
| `pipeline/serve.py` | Map probabilities through the artifact's complete formula. |
| `pipeline/served_oos.py` | Monthly walk-forward served scores. |
| `pipeline/operations.py` | Fit, evaluation, and coverage preparation for lifecycle. |
| `evaluation/` | The shared eligible series population, scoring, and reporting. |

## Prediction and fitting

`ServedModel` holds its fitted bag, feature names, `rating_dependency`, region table,
`rating_blend`, and temperature. `serve.predict` and `serve.predict_batch` build features in
both orientations, score the bag symmetrically, and blend toward the `rating` logit.
`adjust_map_prob` then applies the region residual, the artifact temperature, and the
inactivity dampener, in that order. The dampener reaches only a match whose lineups the
rating cannot see, a scheduled one, unless the artifact's recipe also damps facts games.
Cached-corpus backfill uses the same blend and post-base functions.

`chosen_model()` returns the source recipe. `artifact_recipe()` rebuilds the same recipe
type from a persisted family for lifecycle evaluation. Lifecycle can refit an incumbent
family after a feature-schema change if its named columns still exist. Before serving
learned-rating features, the artifact loader checks the exact `rating` bag dependency.

`build_fit_context` always provides every family index and one `FeaturizedCorpus`. The
Original-family build can overlap other index builds on its own database connection. An
exact corpus-cache hit skips featurization. `fit_window` slices the corpus and applies
window-specific recency weights, then fits the seed bag. Each seed uses one native LightGBM
thread. Members combine in logit space over one shared typed frame.

The market-free evaluator groups games by canonical `series_id`, fits each monthly window on
earlier rows, and reports map and series metrics. Series aggregation takes the maps required
to win from `vocab/series.py` and walks the remaining maps in `pipeline/series_math.py`, where
each map of lead shifts the leader's pre-series log-odds by a fitted momentum.

## Temporal and cache contracts

An `AsOf` query sees series that started earlier and its own earlier maps, never a sibling
series that started at the same instant. The lineup index uses the query's calendar day.
The `rating` index reads states at the start of that day, and a facts game reads its own
five and context. The feature corpus stores both orientations at each map's own position.

`db/facts_prefix.py` owns content identities for historical facts. Original indexes and
served corpora use `SERVED_CORPUS_INPUTS`. Skill-filter corpora, region-fit caches, and
enriched map reads use `RATING_REPLAY_INPUTS`. A committed WAL correction changes these
identities even when row counts and the main SQLite file's timestamp stay the same.

Disk caches resolve under `config.DATA_DIR / "cache"`. In-memory databases bypass
persistent corpus caches. `atomic_write.write_pickle` publishes cache files atomically. `pipeline/fit_corpus_store.py`
keeps its most recently used groups per kind, so a new recipe, rating artifact, or schema
does not strand the old group's blobs. The
connection-path reader keeps no connection objects. Region caches keep one facts vintage and
evict all its fitted tables when that vintage changes.

## Artifacts and identity axes

| Identity | Owner | Changes when |
| --- | --- | --- |
| Served recipe `version` | `pipeline/chosen_model.py` | Prediction meaning changes. |
| `CONFIG_SCHEMA_VERSION` | `pipeline/chosen_model.py` | Persisted recipe structure changes. |
| `_SERVED_FORMAT` | `pipeline/chosen_model.py` | Served family file layout changes. |
| `SCHEMA_VERSION` and feature fingerprint | `features/manifest.py` | Column semantics or pruning change. Documentation notes do not change the fingerprint. |
| `_FIT_CORPUS_CACHE_VERSION` | `pipeline/fit_corpus_store.py` | Stored feature-row semantics change. |
| Original `_CACHE_VERSION` | `features/families/original.py` | Stored Original-family index semantics change. |
| `CORPUS_CACHE_VERSION` | `ratings/skill_filter/corpus.py` | Serialized skill-filter corpus fields or dtypes change. |

`models/served/served.json` records the recipe, feature schema, `rating`
dependency, and `.member{i}.lgb` files. New recipes serialize seeds as a list. The reader
also accepts the incumbent's nested seed metadata when its per-seed overrides are empty.

`artifacts.py` owns artifact-family identities. Source and fast tests freeze no production
SHA.
