# Traps

Findings an agent would otherwise re-derive, each with the test or guard that pins it.

## Data and caches

- **A SQLite WAL commit moves no file mtime, row count, or maximum game id.** Caches key on
  content: `facts_state` triggers, `db.facts_prefix`, and `db.facts.cache_revision`. Pins:
  `tests/db/test_facts_state.py`, `tests/predraft/ratings/test_corpus_cache.py`.
- **An ingestion marker certifies CSV bytes, not identity adjudication.** Every ingestion
  run reads the collision report. Pin: `tests/ingestion/test_service.py`.
- **A blank OE `playerid` or `teamid` is an entity OE never linked.** Stringified, blanks
  pool into one `'nan'` entity, so ingestion skips those rows and every game with an
  unlinked team. Pin: `tests/ingestion/test_oe_facts_ingest.py`.
- **Name reads go through `db.names`.** An unordered `LIMIT 1` returns any alias row, and
  the greatest `rowid` is the current handle. Pin: `tests/db/test_names.py`.
- **Stored model output is stale after any data refresh, under an unchanged model id.** OE
  re-exports revise rows, and one revised game shifts every later walk-forward feature, so
  `update` rescores every game.
- **A feature row semantic change bumps every cache layer.** Team snapshots, Original-family
  indexes, and fit corpora version independently. Pin: `guard_batched_backfill`.

## Models

- **A serving surface reads its recipe from the artifact, never from source.**
  `artifacts.persisted_recipe` is the read.
- **An artifact's field names and filenames are its identity.** Readers assert fields and
  raise `ValueError`, never `.get()`. `artifacts.mint` hashes names and bytes, and
  artifacts stamp a source fingerprint, so to judge determinism compare fitted members,
  never ids.
- **Served pins the exact rating artifact it was fit on.** A new rating artifact makes the
  served artifact refuse to load, so `retrain rating` also retrains served.
- **A score row carries `as_of - 1 day`.** Readers apply `asof_date <= window end`, so a
  row stamped with the exclusive fit cutoff serves nothing.
- **A family freezes its `.knowledge.npz`.** Rebuilding the champion stores changes the next
  draft fit, never a fitted family.
- **A bitwise mirror computes the favoured side and takes `1 -` it.** `1 - q` is exact only
  for `q` in [0.5, 1], and an F-order row reduces in another order. Homes:
  `probability.sigmoid_mirrored`, `EdgeCurve.win_rate`.
- **A structural probe cannot draw an index column from `randn`.** `set_variant._probe`
  draws ids in `slot_inputs.IDENTITY_COLUMN`. Pin: `tests/draft_grading/test_set_fit.py`.
- **A served query is a game, not a day.** `asof.AsOf` sees series that started strictly
  before its own plus its own earlier maps, never a same-instant sibling, and the rating
  index scores the game's own five and context. Careers sort by
  `(scheduled_ts, series_id, game_no)`, and a team's games are its series' games. Pins:
  `tests/shared/test_asof.py`, `tests/predraft/features/test_snapshot_stream.py`,
  `tests/predraft/ratings/test_skill_filter_derived.py`.
- **The draft walk sizes its refits once, at launch** (`oos_walk.walk_worker_count`), from
  free RAM and VRAM. Run nothing memory-hungry beside a draft retrain.

## GPU

- **A GPU result depends on device and batch shape.** CPU and CUDA read 1e-8 apart, and
  cuBLAS picks kernels by row count, so serving pads to `FORWARD_ROWS` and a pin names one
  device. `pin_fit_randomness` enables deterministic algorithms.
- **Graphed training owns its storage.** Clone sample inputs, accumulate into owned
  gradients, and run a batch with a fully masked head eagerly. Pin:
  `test_graphed_training_preserves_every_weight_and_loss`.

## Draft knowledge

- **A reviewed set that sources an artifact cannot also gate it.** Compare prose CC kinds
  against `reviewed_cc.json` (`audit_prose`), never the finished store. Pin:
  `tests/draft_grading/test_cc_knowledge.py`.
- **A wiki leveling array of 18 or more values is the level axis, not rank.**
  `ability_fields._axis_index` owns the rule. Pin: `tests/draft_grading/test_store_precedence.py`.

## Tooling

- **Read config through the module, at call time.** A `from` import and Click defaults
  freeze it. Pin: `tests/shared/test_config_access.py`.
- **`RUF100` and `PT018` are report-only,** because their `--fix` deletes comments. A
  wrapped import's `# noqa` goes on the `from (` line, and `S608` anchors to the first line.
- **Git Bash's `/tmp` is not Python's `C:\tmp`.** Put scratch files under `.tmp/`.
- **Parallel pytest sessions exhaust memory.** `tests/conftest.py` serializes them on a
  named mutex. Pass only `-n auto`, and read free memory before a heavy script.
