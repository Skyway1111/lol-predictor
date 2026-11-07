# Orchestrator architecture

The orchestrator maps Click commands to domain operations. Commands parse, dispatch, and
print. `architecture.md` owns the command map and cross-domain flow.

| Module | Responsibility |
| --- | --- |
| `cli.py` | Click arguments, options, and result printing. |
| `jobs.py` | `update`, `retrain`, and the PandaScore schedule sync. |
| `runtime.py` | Host and port checks, database readiness, then the Uvicorn server. |

`config.py` loads the environment files at import. Consumers resolve its mutable paths when
each operation runs. `runtime.py` reads `LOL_PREDICT_PUBLIC` to allow a public bind, and
refuses to start on a database with no games.

`update` and `retrain` share `jobs._ingested`: hold the pipeline lock
(`models/.pipeline.lock`), ingest new games, run the model step, then rebuild the dashboard
marts once. A database with no games ingests every configured year. An unresolved player
collision or an ingestion error stops the run unless the operator passes
`--allow-dirty-ingest`. Each model commits its own outputs, so a failure leaves the models
before it current and the failing one unchanged.

Schedule sync commits fetched matches and played-series links before prediction. A changed
team pairing clears both its prediction and its played-series link.
`upcoming/upcoming-architecture.md` holds the PandaScore details and temporal rules.
