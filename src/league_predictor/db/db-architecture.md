# DB architecture

`db/` is the bottom layer every domain reads through: SQLite schema, identities, and shared
read rules. It knows no model's math. It owns the contracts that keep persisted data readable
the same way everywhere.

## Module map

| Module | Owns |
| --- | --- |
| `schema.py` | The one DDL home: facts and output tables, `PREDICTION_CONTEXTS`, `apply_current_schema` |
| `facts.py` | `facts_generation`, `cache_revision`, and `latest_played_date`, optionally bounded by `through` |
| `registry.py` | `models` rows through `ensure_registered`, `get_model`, and `prune_orphaned_models`, plus `code_fingerprint`. Ids mint in the `artifacts` leaf. |
| `asof.py` | The two temporal read rules: `team_state` strictly before, impact scores on or before |
| `predictions_read.py` | The one `predictions` read home: fragments, the walk and live rule, oriented oos priors |
| `draft_scores_read.py` | The one `draft_scores` read rule: one row per game and model, `oos` winning over `backfill` |
| `names.py` | Canonical team and player display names, newest name wins |
| `rows.py` | Typed row shapes shared by readers |
| `writes.py` | Write helpers for tables with more than one writer |
| `facts_prefix.py` | Facts-prefix digests and population identities |
| `series_identity.py` | Derived series-identity inputs |
| `rebuild_guard.py` | `require_rebuild_schema`: names the tables a rebuild needs before touching any |
| `runner.py` | Supported baseline validation before schema upgrade |
| `__init__.py` | `get_connection`, plus `create_schema` and `ensure_database_ready`, both aliases of `schema.apply_current_schema` |

## Schema bootstrap and facts generation

`ensure_database_ready` accepts fresh databases, the complete migration-0020 baseline, and
schema version 21. It validates every retained baseline table before DDL, then upgrades
atomically. Older or unversioned populated databases raise before any write. To open one
here, first upgrade a copy through migration 0020 with the prior release.

`facts_state.generation` advances inside every facts, identity, roster, draft-score, and
Impact-history write transaction. Readers see committed generations, including updates held
in WAL. Serving handles key on this scalar alongside artifact stamps.
`player_id_map.evidence` keeps adjudication provenance after a fold removes its evidence rows.

## The two temporal rules (`asof.py`)

- A `team_state` row stamped `asof_date = D` includes games through the end of day D, so it
  prices only events on D+1 or later: **strictly before** (`<`).
- A `player_impact_scores` row stamped D is readable **on** D (`<=`). The fit's `as_of`
  cutoff is exclusive, so the writer stamps `as_of − 1 day`.

Every dashboard as-of read goes through these two composers. Never hand-write the comparison.

## The post-draft stack (`postdraft_stack`)

One row per team, plus the global row `'*'`. The global row carries the shared coefficients
`a` and `beta`, the penalty the time-ordered holdout chose, `holdout_gain`, and `u = 0`.
`holdout_gain` is global holdout log loss minus team-aware, so positive means the team terms
helped. Every other row carries that team's utilization `u` and the `n_oos` games behind it.
When the team terms lose the holdout, every `u` is 0. Both stamp columns carry the parent model
pair, so `postdraft.coverage.assert_fresh` catches a stale stack as it catches stale scores.
`postdraft.coverage.rebuild` refits and rewrites the table with `postdraft_scores` in one
transaction.

## Tracing a dashboard number

Every number HERALD shows follows one read chain:

1. The model home fits, and its lifecycle verb writes the output table: `predictions`,
   `draft_scores`, `player_impact_scores`, or `postdraft_stack` with `postdraft_scores`.
2. An analytics mart copies it daily (`analytics/analytics-architecture.md`), or the route
   reads it live through `asof.py` or `predictions_read.py`.
3. The route in `dashboard/routes/` shapes the payload.

Impact and the served prediction are LIVE reads, never mart copies. Grades and marts are
daily. Model homes: the Model homes table in `src/league_predictor/index.md`.
