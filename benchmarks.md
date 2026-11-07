# Benchmarks

Canonical model quality and timings. Each measurement names its date and population.

## Reference machine

- Windows, Python 3.14
- AMD Ryzen 7 7800X3D
- NVIDIA RTX 5080
- 32 GB RAM

## Model quality

Held-out evaluation of each model: a model fit only on earlier data scores every game.
`through` is the last played game date in the evaluation window.

| model | through | n | log-loss | accuracy |
| --- | --- | ---: | ---: | ---: |
| Served | 2026-09-11 | 2,492 maps | 0.6139 | 0.6517 |
| Rating | 2026-08-15 | 1,545 maps | 0.6103 | 0.6595 |
| Draft | 2026-09-11 | 2,322 games | 0.6899 | 0.5444 |
| Impact | 2026-08-14 | 2,719 games | 0.6219 | 0.6513 |

## Pipeline

| run | measured | wall |
| --- | --- | ---: |
| `update`, all four backfills and the dashboard tables, cold feature index | 2026-11-07 | 8 min 36 s |
| `retrain`, all four models, one draft refit at a time | 2026-11-07 | 40 min |
| one real OE year into an empty DB, `tests/ingestion/test_oe_real_csvs.py` | 2026-10-13 | 16 to 29 s |

Within a draft retrain, the fit takes about 2 minutes and its walk-forward history twelve
monthly refits of 76 to 120 s each (2026-11-07). `draft_grading/oos_walk.py` sizes how many run at
once from free RAM (8 GiB each, plus an 8 GiB reserve) and free VRAM (1.5 GB each).

## Dashboard latency

Warm medians through `dashboard.model_serving.DraftScorer`, three runs per board,
measured 2026-09-12.

| board | median |
| --- | ---: |
| suggest, opening | 0.099 s |
| suggest, mid-draft, 5 picks and 6 bans | 0.070 s |
| suggest, closing, 9 picks | 0.008 s |
| score, full 10-pick draft | 0.009 s |

## Feature builds

Unit costs of the snapshot stack, `predraft/features/snapshots/stream.py`, on the full DB
in one process, measured 2026-09-10. A cold `original` index over 92,601 maps in both
orientations takes 111 s; `data/cache/` holds it afterwards.

| unit | cost | count |
| --- | ---: | ---: |
| bulk store load: every team's timeline | 3.9 s | 1,900 teams |
| bulk store load: every player's career prefix | 8.0 s | 10,139 players |
| one team snapshot at a map position, every group | 0.51 ms | 185,216 |
| `learned_rating` / `lineup` / `recency` index build | 34.2 s / 3.9 s / 0.5 s | |
