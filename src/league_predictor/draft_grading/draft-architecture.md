# Draft architecture

The draft model fits and serves one format-4 family of set encoders. `recipe.py` defines the
model, and `DRAFT-MODEL.md` explains its inputs, targets, and grades. `lifecycle/draft.py`
owns the artifact swap and the database transaction.

## Family and serving

| Home | Responsibility |
| --- | --- |
| `state.py` | Typed drafts, role-ordered slots, bans, and bulk facts loading |
| `grader/embeddings.py` | Champion indexes and aliases, with an UNKNOWN row |
| `grader/slot_inputs.py` | Frozen input specification and per-slot tensors |
| `grader/set_encoder.py` | Shared encoder, antisymmetric score, and auxiliary heads |
| `grader/set_training.py` | Chronological early stopping, seed fits, and weight payload |
| `grader/knowledge_tables.py` | Frozen knowledge member and integrity checks |
| `grader/set_artifact.py` | Three-member family serialization and loading |
| `set_variant.py` | Seed scoring, calibration, grades, and marginalized swaps |
| `live.py` | Production loader and hypothetical draft construction |

A family holds top JSON metadata, `.encoder.pt` weights, and `.knowledge.npz` knowledge.
Loading checks format, required metadata, checksum, shapes, and exact encoder antisymmetry.
It rebuilds inputs from the artifact, never from a changing store. `artifacts.py` owns
family identity.

The predictor averages seed scores, applies its fitted temperature, and computes the
favoured side's probability once. A mirrored draft scores exactly `1 - p`. Every encoder
forward runs at `FORWARD_ROWS` rows, so a row's score is independent of its batch: the D6
pin. Batch calls require a shared `BulkStatsCache`, and all swap arities share one
board-expansion implementation. CUDA serving captures the seed bag once and transfers its
outputs together. The predictor serializes graph replay because captured inputs and outputs
share storage.

## Fitting

`training/set_fit.py` assembles chronological drafts and fits the frozen production recipe
and the out-of-fold calibration pool. Final seeds use the epoch count chosen outside that
pool. `training/provenance.py` stamps the exclusive cutoff and code fingerprint. Training prepares compact label indices before each epoch and captures CUDA
encoder forwards and gradients. Calibration scoring uses the fit's device selection.

`grader/bulk_stats.py` computes strict-before-date champion-role statistics from decayed
prefixes. `grader/pro_excess.py` owns their team-form and impact debias maps. Baselines and
enemy responses use `grader/corpus_memo.py`, keyed by the connection and by local and
external writes. Team form sees earlier series and earlier maps of its own series, never a
simultaneous sibling series. `grader/dense_labels.py` supplies masked auxiliary targets,
used only in training.

## Applications

`draft_scoring.py` writes whole-draft scores. A malformed batch length raises before any row
enters the write buffer. A predictor exception falls back to isolated per-game reads.
`player_pick_scoring.py` uses action order and enemy responses to write pick evaluations.
Both writers enforce the predictor-error budget before commit.

`coverage.py` reconciles current-family scores and runs the D6 reproducibility spot check.
`oos_walk.py` refits at trailing month boundaries and refuses partial coverage. The
post-draft stack consumes only games both parents score out of sample. Historical draft
tendencies become prior-only served features.

`roles.py` assigns roles by play counts. `suggest.py` ranks legal actions by current role
support, counterfactual scores, and rollout values. Each candidate's displayed value matches
its post-action draft.

Champion schema constants live in `champion_attributes/ability_schema.py`, and extraction
commands live in `scripts/champion_attributes/`. Tests own independent knowledge audits.
