# Draft model

The draft model estimates the advantage carried by the champions in each role. It reads no
team or player identity. `recipe.py` owns the frozen production definition,
`draft-architecture.md` maps the implementation, and `benchmarks.md` holds quality numbers.

## Inputs and score

Each of ten slots carries its role, champion kit vector, simulated power curve, and
strict-before-date professional form. `grader/slot_inputs.py` persists the exact blocks and
column order. UNKNOWN champions get the frozen neutral row. Recipe-controlled input denials
and identity residuals use the same builder.

The shared encoder's score is exactly antisymmetric under a team swap. Named kit
interactions compare offence and defence across teams. Auxiliary heads supervise lane and
objective observations during training. Serving averages seed scores, applies the
out-of-fold temperature, computes the favoured side once, and mirrors the other, so
`p(mirror) == 1 - p`. Fixed-size forwards keep each row independent of its batch.

## Training and grades

Training reads games strictly before its cutoff in facts order. Champion form removes the
preceding team-strength baseline. The outcome loss uses the recipe's training-only offset,
and a missing served prior never becomes an artificial coin-flip observation. Role-aware
auxiliary targets are masked where observations are absent.

Chronological folds choose the stopping epoch and collect out-of-fold scores. Final seeds
train for that epoch count. Calibration fits a temperature and a monotone, symmetric
score-to-win-rate curve on the out-of-fold pool. Grade thresholds are preimages of declared
edge cuts. `edge_pp_blue` is calibrated advantage in win-rate percentage points, and red's
edge is its negation. An empty grade tail means no score reached that edge.

## Frozen knowledge and applications

Wiki declarations supply ability and base-stat features. Commands in
`scripts/champion_attributes/` apply reviewed calculator contracts where one exists and keep
prose for unpriced mechanics. Declared exceptions stay explicit. Tests compare stored
knowledge against independent control declarations and champion-specific kit claims.

The dense kit vector groups mechanics into named families. Power-grid inputs carry per-role
simulated curves and borrowed-cell flags. Cached damage profiles and a deterministic stat
fallback supply damage shares during building. Every family freezes these inputs into
`.knowledge.npz`, checked against its metadata checksum and shapes at load. A store refresh
changes the next fit, never an already fitted family.

`draft_scores` holds whole-draft outputs, and `pick_scores` holds per-pick outputs. Backfill applies
the current family over history. The monthly out-of-sample walk refits before each scored
month, and the source and artifact recipes must agree before it runs. Served tendency features read only
prior games' draft scores. Suggestions assign roles by play counts, enforce legality and
current support, and show each action's realized value through the frozen edge curve.
