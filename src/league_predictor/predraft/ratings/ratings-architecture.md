# Ratings architecture

The skill filter, behind the `rating` token, predicts matchups from latent strength. Impact
estimates player contributions from results, regional strength, and statistical priors.
Region rating is the one league-strength derivation. The Naming boundary section of
`architecture.md` lists its consumers.

## Skill filter

| Module | Responsibility |
| --- | --- |
| `skill_filter/recipe.py` | The fixed skill-filter architecture and training schedule. |
| `skill_filter/corpus_build.py` | Eligible chronological games, masks, rosters, and normalization. |
| `skill_filter/corpus_schema.py` | Corpus tensors, vocabulary, and normalization dataclasses. |
| `skill_filter/corpus.py` | The persistent corpus cache, bound to the facts digest. |
| `skill_filter/model.py` | Parameters and the shared scoring and update formula. |
| `skill_filter/replay.py` | Sequential serving replay. DEER's differentiable oracle lives in `tests/predraft/ratings/_replay_reference.py`. |
| `skill_filter/replay_bag.py` | Batched replay across compatible bag members. |
| `skill_filter/deer.py` | Parallel fixed-point trajectories and implicit gradients. |
| `skill_filter/train.py` | Fixed CUDA and DEER bag training, with resource and convergence refusals. |
| `skill_filter/artifact.py` | Bag serialization, deployed-config translation, and dependency identity. |
| `skill_filter/serving.py` | Artifact loading, corpus replay, and one verified snapshot lookup. |
| `skill_filter/serving_index.py` | Batched queries over one member's frozen timelines. |
| `skill_filter/serving_frozen.py` | The flat timelines a serving index retains, and `GameInputs`, each corpus game's own inputs. |
| `skill_filter/serving_bag.py` | Probability-average predictions and representative-member latent features. |
| `skill_filter/state_export.py` | `team_state_rows(index, model_id)` and materialized-state reconciliation. |
| `skill_filter/operations.py` | Candidate fitting, evaluation, and coverage preparation. |

`SkillFilter` owns the trainable PyTorch parameters, and `BatchedSkillFilter` holds stacked
parameter tensors. Both score and update through `_FilterMath`. Solo serving keeps its
linear kernels and batched replay keeps `bmm`, so the shared formula neither changes
numerical layout nor allocates a K axis on each serving query.

The filter predicts each day before applying that day's updates. Player and organization
states decay toward their priors across inactivity. The filter reads roster continuity,
debut status, box-score features, context, and latent dispersion. Unknown league
embeddings add nothing beyond the global prior.

A query names a facts game through its `AsOf`. The index then scores that game from its own
fielded five, context, and lineup continuity (`GameInputs`), the inputs the replay trained
on, so serving and training read one model. Any other query, a scheduled match or a bare
day, reads each team's last record with neutral context. Artifacts from recipe version 2
serve facts games this way; an unstamped or version 1 artifact reads every query from the
last record.

`BaggedSkillFilterBackend` loads the sole artifact format, including one-member test bags.
The backend checks frozen timeline structure before compatible member groups share one
batched replay. The bag averages member probabilities. Member latent coordinates are not
aligned, so the bag exposes latent features from one representative member.

A `rating.index.pkl` snapshot binds the artifact SHA, facts digest, source signature,
and replay cutoff. An absent or stale snapshot triggers replay. State export writes one
end-of-day strength row per team appearance, stamped with the appearance day.

## Region rating

`region_rating/corpus.py` reads completed games and player appearances into sparse
matrices. The process cache holds one facts vintage. Each fit derives home pools and
international context from its strict cutoff prefix. `region_rating/fit.py` fits player,
pool, and tier levels. `region_rating/table.py` resolves domestic teams and applies the
served residual.

Pool aliases come from `league_tier.py`. An unknown team gets the tier fallback, not the
population average. Same-pool matchups, and same-tier matchups below tier one, get an
identity residual. Fitted tables carry the recipe's residual coefficients.

## Player profiles and statistical context

`player_performance_prefix.py` builds chronological career prefixes. The profile query
returns sufficient statistics to `player_performance.assemble_performance_profile`. The
independent row-wise reference lives only in the profile parity tests.

`player_scoring.py` owns role pillars, baseline queries, and score composition.
`stat_context/appearances.py` builds the game-level statistical input.
`stat_context/grid.py` fits covariate-adjusted expectations from trailing windows and forms
residual z scores. The supported Impact build adds the opponent-adjusted lane-BT prior.
`stat_context/prior.py` aggregates the context and loads the fitted team-sum weights from
`config.DATA_DIR`. Callers choose a shrinkage count, where zero means the caller owns
shrinkage.

`stat_context/vocab.py` owns the stat and pillar vocabulary, grid constants, and the at-10
laning scale. Missing measurements stay absent, and thin cells never become zero
observations. The weight-span guard keeps scaled-prefix arithmetic from overflowing.

## Impact

| Module | Responsibility |
| --- | --- |
| `player_impact/recipe.py` | The supported Impact recipe, statistical build kwargs, and prior blocks. |
| `impact_corpus.py` | Chronological factorized games, lineups, and appearance roles. |
| `player_impact/design.py` | Sparse result-model design and the ridge solve. |
| `player_impact/fit.py` | Player qualification, priors, regional levels, and fitted values. |
| `player_impact/predictor.py` | The linear matchup predictor and published contribution values. |
| `impact_artifact.py` | Artifact recipe validation and materialized player-score fields. |
| `impact_history.py` | The fixed online filter for the historical Impact grid. |
| `impact_operations.py` | The fit and history preparation. |

The Impact corpus projects only the columns it consumes, so added facts columns do not
invalidate its schema. It still validates lineup width, role, and timestamp.
Production statistical context comes from the validated artifact recipe.

The weights reader accepts the incumbent's covariate, no-champion-offset build metadata
and normalizes those fixed fields before comparison. It refuses any other recorded build.
A change of model meaning needs a new recipe version and a lifecycle evaluation.

`db.latest_played_date` anchors current fits and bounded history. Fit cutoffs are
exclusive, so materialized score rows carry the last included day.
