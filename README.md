# LoL Professional Match Predictor

Predictions and a dashboard for professional League of Legends, built on
[Oracle's Elixir](https://oracleselixir.com/) match data.

```
Oracle's Elixir CSVs → SQLite → four models → dashboard
```

The headline number is P(team A wins), predicted before the draft. Every fit and every
historical prediction is walk-forward: a model only sees games played before the one it
scores.

| Model | What it answers |
| --- | --- |
| `served` | Who wins, before the draft. Bo1, Bo3, and Bo5, from any series score. |
| `rating` | How strong each player and team is right now, as latent strength. |
| `draft` | How good a finished draft is from champions alone, as an S to D grade and edge. |
| `impact` | How many map-win percentage points a player adds over an average starter in their role. |

[`architecture.md`](architecture.md) covers how they fit together.
[`benchmarks.md`](benchmarks.md) has their measured quality.

## Quickstart (Windows)

Install Python 3.11 or newer, clone the repo, then double-click:

| File | What it does | Time |
| --- | --- | --- |
| `dashboard.bat` | Opens the dashboard at http://localhost:5000 | seconds |
| `update.bat` | Downloads new games and scores them with the current models | about 8 min |
| `retrain.bat` | Downloads new games and refits all four models on them | about 40 min |

The first `dashboard.bat` sets up Python and builds the database, which takes about
40 minutes once. Trained models ship in `models/`, so you never need to retrain to get
predictions. Retrain when you want the models to learn from new games; an NVIDIA GPU
makes the rating and draft fits much faster.

## Other platforms, or the command line

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e .
lol-predict update                  # pull new games, score them
lol-predict dashboard --open        # http://localhost:5000
lol-predict retrain                 # refit all four models; or e.g. `retrain impact`
```

`lol-predict upcoming` fetches the PandaScore schedule and predicts unplayed matches. It
needs a free [PandaScore](https://pandascore.co/) key: put `PANDA="<key>"` in
`panda.env` at the repo root.

## Development

```bash
pip install -e ".[dev,lint]"
python -m pytest tests     # about a minute
ruff check && ruff format --check .
```

## License and data

The code is MIT licensed ([`LICENSE`](LICENSE)). The data it runs on is not ours and is
not covered by that license:

- **Match data** comes from [Oracle's Elixir](https://oracleselixir.com/), downloaded from
  Oracle's Elixir's public Google Drive (`OE_GDRIVE_IDS` in `src/league_predictor/config.py`).
  Every model in `models/` is trained on it. Credit Oracle's Elixir and follow its terms
  when you use or share anything built from it.
- **Champion data** in `src/league_predictor/draft_grading/champion_attributes/data/` comes
  from the [League of Legends Wiki](https://wiki.leagueoflegends.com/) (CC BY-SA) and
  [League of Graphs](https://www.leagueofgraphs.com/), scraped with a vendored subset of
  [lolstaticdata](https://github.com/meraki-analytics/lolstaticdata) (MIT, its own
  `LICENSE` beside it).
- **Upcoming schedules** come from the [PandaScore](https://pandascore.co/) API under your
  own key.

*League of Legends is a trademark of Riot Games, Inc. This project is not affiliated with
or endorsed by Riot Games.*
