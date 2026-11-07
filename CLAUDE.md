# LoL Professional Match Predictor

Pro-match prediction and a read-only dashboard on one pipeline:
`CSVs → SQLite facts → four models → dashboard`. The four models (`served`, `rating`,
`draft`, `impact`) share one lifecycle, `retrain` and `backfill`, and every fit and every
historical prediction is walk-forward.

## Read by task
- Where a file lives: `src/league_predictor/index.md`. The narrative and the four-model
  table: `architecture.md`.
- Quality and performance numbers: `benchmarks.md`, canonical, never restated.
- A structural change inside a domain: its `<domain>/<domain>-architecture.md`, updated in
  the same change.
- What the draft model means: `src/league_predictor/draft_grading/DRAFT-MODEL.md`.
- A model recipe change: its recipe file (index, Model homes), bump the recipe version, then
  `lol-predict retrain <model>`.
- Refresh data: `lol-predict update`. Refit on new data: `lol-predict retrain`.
- Known traps: `TRAPS.md`. Read it before re-deriving anything.

## Layout
```
src/league_predictor/
  ingestion/      Oracle's Elixir CSVs to facts, identities, rosters
  db/             schema bootstrap, registry, temporal and name reads
  predraft/       served predictor, skill filter (the rating model), features, Impact
  draft_grading/  the draft grader
  postdraft/      served x draft stack, a derived output, not a model
  lifecycle/      retrain and backfill for every model
  analytics/      dashboard marts
  dashboard/      read-only FastAPI routes and the HERALD UI
  upcoming/       PandaScore schedule and pre-match predictions
  orchestrator/   the lol-predict CLI and its jobs
  vocab/          dependency-leaf roles and draft order
  <root>.py       config, clock, artifacts, probability, metrics: dependency leaves
scripts/champion_attributes/   rebuilds the draft model's champion knowledge stores
models/<model>/   the current artifact family of each model
data/             the SQLite facts DB and caches; only power_curves.db is in git
```

## Invariants
Safe, intuitive, performant, in that order.
- No silent cost cliffs. A fallback auto-engages only at ≤2x wall, the same memory class
  and the same model meaning. Anything bigger raises, naming the cause and the opt-in.
- Warnings are not safety. A condition that changes what the operator should do is an
  error. A warning only records.
- Time ordering everywhere: latent strength, player profiles, roster features, training
  windows and splits are walk-forward. A served query is an `asof.AsOf`, a game's place in
  the facts order: it sees the series that started before its own plus its own earlier
  maps, never a sibling series that started at the same instant.
- The `predictions` table holds one canonical `p_a` per game and model (`team_a < team_b`,
  `context IN ('backfill','oos')`). Draft and Impact own their own output tables.
- Only `update` and `retrain` write the database and `models/`; the dashboard only reads.
- A change of model meaning bumps its recipe version.
- One fact, one home. Docs state the present in plain English; git holds the past.

## Gates
```
.venv/Scripts/python -m pytest tests          # fast lane, about a minute
.venv/Scripts/python -m pytest tests -m ""    # adds real-DB smokes and real-CSV ingests
.venv/Scripts/ruff check
.venv/Scripts/tach check                      # package boundaries, tach.toml
```

## Environment
- Python 3.14 locally (3.11 is the floor); always `.venv/Scripts/python` from the repo
  root. PowerShell, not Bash.
- `requirements.lock` pins the dev venv, including CUDA torch:
  `uv pip sync requirements.lock`.
- Subagents run Opus.
- `.claude/worktrees/` holds ignored temporary copies of this repo; never read it as source.
