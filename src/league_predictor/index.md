# Package index

Pick a starting file by task. Source and each domain's architecture document own
implementation detail. Package paths are relative to `src/league_predictor/`.

## Start by task

| Task | Start here | Then, only if needed |
| --- | --- | --- |
| Understand the whole system | `architecture.md` | none |
| Find a package owner | The Layout block of the root `CLAUDE.md` | The package's architecture document |
| Change retrain or backfill | `lifecycle/base.py` | The model's `lifecycle/<model>.py` |
| Change the served recipe | `predraft/pipeline/chosen_model.py` | `predraft/predraft-architecture.md` |
| Change what a served query sees: the walk-forward cutoff, the series score | `asof.py` (`AsOf`, `MatchContext`) | `predraft/predraft-architecture.md` |
| Trace or change a served feature ("where is `diff_cspm` computed?") | `predraft/features/manifest.py`, `predraft/features/families/` | `predraft/pipeline/chosen_features.py` (`SERVED_FEATURES`, the locked column list) |
| Follow a live prediction | `predraft/pipeline/serve.py` | `predraft/pipeline/chosen_model.py` |
| Change skill filter math | `predraft/ratings/skill_filter/recipe.py` | `predraft/ratings/ratings-architecture.md` |
| Change the skill filter's training corpus | `predraft/ratings/skill_filter/corpus.py` | `predraft/ratings/ratings-architecture.md` |
| Change the Impact recipe or prior | `predraft/ratings/player_impact/recipe.py` (`IMPACT_RECIPE`) | `predraft/ratings/stat_context/` |
| Change draft model meaning | `draft_grading/DRAFT-MODEL.md` | The grader source it names |
| Change champion knowledge, the 360-dim store | `scripts/champion_attributes/ability_features.py` (extraction), `scripts/champion_attributes/cc_labels.py` (sourced control labels) | `draft_grading/grader/knowledge_tables.py` (the frozen serving member) |
| Change the dense named kit vector, or claim how a champion should read | `draft_grading/champion_attributes/kit_vector.py` (`KIT_FAMILIES`) | `tests/draft_grading/test_kit_knowledge.py` (the claim table) |
| Change the computed per-role power curves | `draft_grading/champion_attributes/power_store.py` (the cache reader and its channels) | `draft_grading/grader/power_curve_features.py` (the frozen per-slot power table) |
| Change the post-draft stack | `postdraft/stack.py` (the fit), `postdraft/coverage.py` (persistence) | `lifecycle/served.py`, `lifecycle/draft.py` (the rebuilds) |
| Change what a draft fit builds | `draft_grading/recipe.py` (`DRAFT_RECIPE`) | `draft_grading/training/set_fit.py` (`fit_candidate`) |
| Change draft package structure | `draft_grading/draft-architecture.md` | The source it names |
| Change live pick suggestions | `draft_grading/suggest.py` | `dashboard/model_serving.py` (the live handle) |
| Change how picks land in roles, the play-count law | `draft_grading/roles.py` | Its reader: `draft_grading/suggest.py` |
| Change ingestion | `ingestion/ingestion-architecture.md` | `ingestion/oracles_elixir.py` |
| Add a tournament region or league | `predraft/league_tier.py` (the league table) | `ingestion/ingestion-architecture.md` |
| Change region or league strength ratings | `predraft/ratings/region_rating/` (the derivation) | `predraft/ratings/ratings-architecture.md` |
| Change upcoming matches or PandaScore | `upcoming/upcoming-architecture.md` | `upcoming/service.py` |
| Trace `is_playoffs` for a game | `predraft/features/snapshots/__init__.py` (the stage keyword rule) | `predraft/features/families/original.py` (on-demand stage resolution) |
| Change dashboard data | `analytics/analytics-architecture.md` | The builder it names |
| Change the dashboard API or UI | `dashboard/dashboard-architecture.md` | The route or static asset |
| A dashboard number looks wrong | The model's row in Model homes, then `db/db-architecture.md` (the read chain) | The route in `dashboard/routes/` |
| Change CLI behavior | `orchestrator/orchestrator-architecture.md` | `orchestrator/jobs.py` |
| Find current model quality | `benchmarks.md` | none |
| Find a module's tests | The same path under `tests/` | none |

## Model homes

Recipe is the file holding the model's frozen definition, `lifecycle.get(model).recipe`.

| Model | Recipe | Math | Lifecycle |
| --- | --- | --- | --- |
| `served` | `predraft/pipeline/chosen_model.py` | `predraft/pipeline/serve.py` (score), `predraft/pipeline/operations.py` (fit and coverage), `predraft/pipeline/live.py` (live loading) | `lifecycle/served.py` |
| `rating` | `predraft/ratings/skill_filter/recipe.py` | `predraft/ratings/skill_filter/operations.py` (fit and coverage) | `lifecycle/rating.py` |
| `draft` | `draft_grading/recipe.py` | `draft_grading/set_variant.py` (the format-4 set encoder), `draft_grading/training/`, `draft_grading/coverage.py`, `draft_grading/oos_walk.py`, `draft_grading/live.py` | `lifecycle/draft.py` |
| `impact` | `predraft/ratings/player_impact/recipe.py` | `predraft/ratings/impact_operations.py` | `lifecycle/impact.py` |

## Shared root modules

Every root module is a dependency leaf. Only `artifacts.py` imports another package
module, `config.py`.

| File | Owns |
| --- | --- |
| `config.py` | Environment loading and patchable runtime values, read through the module at call time |
| `artifacts.py` | Model files: paths, per-model family enumeration, and content-addressed ids (`mint`, `family_sha`) |
| `probability.py` | Probability, log-odds, and Elo-scale transforms |
| `metrics.py` | Log-loss at logits and accuracy |
| `clock.py` | The naive-UTC wall-clock bridge (`utc_now`, `utc_today`), where "now" enters the facts date domain |
| `asof.py` | `AsOf` query positions and `MatchContext` format, stage, and series score |
| `decay.py` | Exponential decay weights, in half-life pow and rate forms |
| `resources.py` | Memory-aware worker limits |
| `torch_graphs.py` | CUDA training capture, owned sample storage, and warmup stream reuse |
