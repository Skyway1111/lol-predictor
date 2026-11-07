# Analytics architecture

Implementation map for `src/league_predictor/analytics/`. System view:
`../../../architecture.md`.

This package writes the three analytics marts. The orchestrator invokes it, and the read-only
dashboard consumes its tables and display-scale contracts. Player marts consume the public
scoring and snapshot APIs. `tach.toml` owns package boundaries.

## Package layout

| Module | Role |
| --- | --- |
| `builders.py` | `refresh_analytics_marts`, the one generation transaction |
| `team_snapshot.py` | `team_daily_snapshot` builder. It resolves each team's league exactly with `predraft.league_tier.resolve_domestic_league`, anchors it through the learned region table `predraft.ratings.region_rating.fit_region_ratings(...).elo`, a cached walk-forward fit, and gets the region-anchored team power value from `team_power_display.compute_team_power_display`. |
| `player_snapshot.py` | Two-pass `player_daily_snapshot` builder. Pass 1 aggregates per-game z from `predraft/ratings/stat_context/`, the one context-adjustment home, shared with the Impact prior, under the Impact artifact's recipe config from `impact_artifact.production_stat_context_kwargs`. Pass 2 role-norms, shrinks by games, and assembles rows. |
| `player_context.py` | Frozen bulk-query lookups for the draft-axes pass and row assembly: tier baselines, player tiers, and season champions |
| `player_draft_inputs.py` | DB-side draft-radar input assembly |
| `player_draft.py` | Pure five-axis draft-radar aggregators |
| `team_power_display.py` | `compute_team_power_display`, the formula that adds the region-rating anchor to the 1500-centered display, and the power ranking |
| `team_strength_display.py` | Maps skill-filter latent strength to the unanchored team-strength display |
| `row_types.py` | Frozen row contracts and INSERT generation |
| `_common.py` | Transaction and run-log helpers |

## Refresh contract

`orchestrator.jobs` calls `builders.refresh_analytics_marts(conn, as_of_date=...)` as the
last step of `update` and `retrain`. The dashboard never calls it. The function first
asserts the rebuilt schema and runs predraft's `state_export.ensure_team_state_current`, the
`rating` model's `team_state` freshness check, which predraft sequences. One transaction then
wraps all three builders, so the marts publish together or the generation rolls back.

Each builder runs in a SAVEPOINT through `tracked_builder`, replaces its complete date slice,
and records its result in memory. `analytics_runs` provenance commits afterward in a separate
transaction, so an error record survives a generation rollback.

| Builder | Output |
| --- | --- |
| `refresh_team_daily_snapshot` | `team_daily_snapshot` |
| `refresh_player_daily_snapshot` | `player_daily_snapshot` |
| `refresh_team_power_display` | `power_rating_snapshots` |

Impact stays a live as-of read from `player_impact_scores`, never copied into these marts.
It is also the ONLY headline player number. The snapshot holds role-specific pillar
ratings and the draft axes, and no player rating or grade of its own.

The radar is draft-derived only. A player enters it on at least one pick-scores axis, and
Overall Quality is a weighted blend renormalized over the non-NULL axes. `player_draft.RADAR_AXES`
is the one home for the axis keys, the labels the dashboard draws, and the weights. The
players endpoint serves all three, so no other surface restates them.

Marts persist fields the dashboard displays. Each player has one generic `pillars_json`
list. `RowBase` generates plain INSERT statements from frozen dataclass fields. Slice
replacement, not `OR REPLACE`, provides idempotency, so a leftover key collision fails loudly.

`team_daily_snapshot.power_rating` and `power_rating_snapshots` keep their persisted names.
In code and UI they hold the region-anchored **team power display**. Historical game rows use
the separate, unanchored **team strength display**, derived directly from latent strength.
