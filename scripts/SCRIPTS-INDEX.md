# Scripts index

Run a script from the repository root with `.venv/Scripts/python -m scripts.<module>`.

## Champion knowledge

Run these in order to refresh the draft model's champion stores, then `lol-predict retrain
draft`. A refresh changes the next draft fit, never a fitted family. The two
`build_reviewed` modules need a checkout of the damage calculator beside this repo.

| Module | What it does |
| --- | --- |
| `champion_attributes/fetch_wiki_export.py` | Scrapes a fresh champion export from the LoL wiki, minutes. Needs `pip install -e ".[wiki]"`. |
| `champion_attributes/fetch_leagueofgraphs.py` | Fetches missing League of Graphs damage shares into the cache. `--force` refetches. |
| `champion_attributes/build_reviewed_kit.py` | Copies the calculator's reviewed kit contract into the repo. |
| `champion_attributes/build_reviewed_cc.py` | Copies the calculator's reviewed crowd-control declarations, the oracle for CC drift. |
| `champion_attributes/generate_abilities.py` | Builds the 360-dim ability-feature artifact from the wiki export under the reviewed contract. |

The other modules in `scripts/champion_attributes/` are shared readers for these builders and their tests.
The serving schema lives in `src/league_predictor/draft_grading/champion_attributes/ability_schema.py`.
