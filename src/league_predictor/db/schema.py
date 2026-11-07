"""Canonical current schema: facts, then derived."""  # sightline-ok: 27 - one DDL contract

# file-length-ok: one DDL contract — a single CURRENT_SCHEMA_SQL string plus the verb that
# applies it; splitting it by table group would let two halves of one schema drift apart.

# Facts never carry model identity; canonical orientation (team_a < team_b, every probability
# P(team_a)) is a facts-layer CHECK. Derived rows are stamped with their models row.

from __future__ import annotations

from typing import TYPE_CHECKING

from league_predictor.db.runner import SCHEMA_VERSION, schema_upgrade_sql

if TYPE_CHECKING:
    import sqlite3

PREDICTION_CONTEXTS: tuple[str, ...] = ("backfill", "oos")
_PREDICTION_CONTEXT_SQL = ", ".join(repr(context) for context in PREDICTION_CONTEXTS)

CURRENT_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS facts_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    generation INTEGER NOT NULL
);
INSERT OR IGNORE INTO facts_state VALUES (1, 0);
-- ============================================================================
-- FACTS — the append-only world. No model identity ever lives here.
-- ============================================================================

CREATE TABLE IF NOT EXISTS leagues (
    league_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    region TEXT,
    tier INTEGER
);

CREATE TABLE IF NOT EXISTS teams (
    team_id TEXT PRIMARY KEY,
    name TEXT,
    region TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id TEXT PRIMARY KEY,
    name TEXT
);

-- One row per series, scheduled OR completed. winner/status stay NULL until
-- the series is played. Canonical orientation is enforced here: team_a sorts
-- strictly before team_b by team id.
CREATE TABLE IF NOT EXISTS series (
    series_id TEXT PRIMARY KEY,
    league_id TEXT REFERENCES leagues(league_id),
    team_a TEXT NOT NULL REFERENCES teams(team_id),
    team_b TEXT NOT NULL REFERENCES teams(team_id),
    scheduled_ts TEXT,
    best_of INTEGER,
    winner TEXT REFERENCES teams(team_id),
    status TEXT,
    CHECK (team_a < team_b)
);

-- Per-map facts under a series. Blue/red side is a per-game fact. event_type
-- carries the playoffs fact ('regular'/'playoffs', from OE's playoffs flag at
-- ingest) -- the is_playoffs feature reads it.
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    series_id TEXT NOT NULL REFERENCES series(series_id),
    game_no INTEGER NOT NULL,
    blue_team TEXT REFERENCES teams(team_id),
    red_team TEXT REFERENCES teams(team_id),
    winner TEXT REFERENCES teams(team_id),
    patch TEXT,
    duration_s INTEGER,
    event_type TEXT,
    UNIQUE(series_id, game_no)
);

-- Per-(game, player) box score.
CREATE TABLE IF NOT EXISTS game_players (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    player_id TEXT NOT NULL REFERENCES players(player_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    role TEXT CHECK(role IN ('top', 'jng', 'mid', 'bot', 'sup')),
    champion_id TEXT,
    result INTEGER,
    kills INTEGER,
    deaths INTEGER,
    assists INTEGER,
    teamkills INTEGER,
    teamdeaths INTEGER,
    damageshare REAL,
    killparticipation REAL,
    deathshare REAL,
    vspm REAL,
    controlwardsbought INTEGER,
    golddiffat15 REAL,
    csdiffat15 REAL,
    xpdiffat15 REAL,
    golddiffat10 REAL,
    csdiffat10 REAL,
    xpdiffat10 REAL,
    goldat15 REAL,
    csat15 REAL,
    xpat15 REAL,
    dpm REAL,
    cspm REAL,
    earnedgoldshare REAL,
    firstblood INTEGER,
    firsttower INTEGER,
    firstdragon INTEGER,
    firstbaron INTEGER,
    wpm REAL,
    wcpm REAL,
    wardsplaced INTEGER,
    wardskilled INTEGER,
    damagetakenperminute REAL,
    damagemitigatedperminute REAL,
    firstbloodkill INTEGER,
    firstbloodassist INTEGER,
    killsat15 INTEGER,
    assistsat15 INTEGER,
    deathsat15 INTEGER,
    PRIMARY KEY (game_id, player_id)
);

-- Per-(game, team) box score: the team-event aggregates (objective counts,
-- gold totals, team K/D, length) that have no per-player decomposition. A facts
-- table, not a feature cache -- ~15 served feature columns read it.
CREATE TABLE IF NOT EXISTS game_team_stats (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    team_id TEXT NOT NULL REFERENCES teams(team_id),
    dragons INTEGER,
    barons INTEGER,
    towers INTEGER,
    heralds INTEGER,
    firstdragon INTEGER,
    firstbaron INTEGER,
    firsttower INTEGER,
    firstherald INTEGER,
    totalgold INTEGER,
    totalcs INTEGER,
    teamkills INTEGER,
    teamdeaths INTEGER,
    gamelength INTEGER,
    void_grubs INTEGER,
    PRIMARY KEY (game_id, team_id)
);

-- Reconstructed pick/ban sequence, one row per draft action.
CREATE TABLE IF NOT EXISTS draft_actions (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    seq INTEGER NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('blue', 'red')),
    kind TEXT NOT NULL CHECK(kind IN ('pick', 'ban')),
    champion_id TEXT,
    role TEXT,
    PRIMARY KEY (game_id, seq)
);

-- ============================================================================
-- CARRIED REFERENCE / AUX TABLES — outside the two contracts: the
-- name-resolution vocabularies (the OE team-name vocabulary AND the player
-- vocabulary — the player_id_map fold evidence is a durability contract) and
-- their durable fold evidence.
-- ============================================================================

CREATE TABLE IF NOT EXISTS oe_team_names (
    team_id TEXT NOT NULL, team_name TEXT NOT NULL,
    UNIQUE(team_id, team_name)
);

-- The OE player-name vocabulary + the adjudicated player-id fold map. The fold
-- map is a DURABILITY contract: folding deletes the duplicate-row evidence it
-- was detected from, so the map is the only record of the adjudication and must
-- survive every rebuild (ingestion.player_reconciler owns both).
CREATE TABLE IF NOT EXISTS player_aliases (
    id INTEGER PRIMARY KEY,
    player_id TEXT NOT NULL,
    source TEXT NOT NULL,
    source_name TEXT NOT NULL,
    UNIQUE(source, source_name)
);

CREATE TABLE IF NOT EXISTS player_id_map (
    alias_id TEXT PRIMARY KEY,
    canonical_id TEXT NOT NULL,
    evidence TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Team roster validity windows, derived from played games.
CREATE TABLE IF NOT EXISTS rosters (
    id INTEGER PRIMARY KEY,
    team_id TEXT NOT NULL,
    player_id TEXT NOT NULL,
    role TEXT,
    start_date TEXT NOT NULL,
    end_date TEXT,
    UNIQUE(team_id, player_id, role, start_date)
);

-- Walk-forward Impact grid: one row per (player, biweekly fit date), the player's
-- posterior win-log-odds from a fit at that date (which sees only earlier games).
-- Written whole by lifecycle/impact.py on `backfill impact`; read by the
-- draft grader's impact_bt pro-excess baseline as the latest row STRICTLY before a
-- game's date. The PK serves that latest-before scan (SQLite walks it backwards).
CREATE TABLE IF NOT EXISTS impact_history (
    player_id TEXT NOT NULL,
    asof_date TEXT NOT NULL
        CHECK (asof_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    impact_logodds REAL NOT NULL,
    PRIMARY KEY (player_id, asof_date)
);

-- ============================================================================
-- DERIVED — model outputs. Every row is stamped with the model that made it.
-- ============================================================================

-- The one model registry. model_id strings are minted only by db/registry.py.
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    artifact_sha TEXT
);

-- One row per (team, appearance-day) under a model. Appearance-days only;
-- read exclusively through db/asof.py (latest row strictly before the event).
CREATE TABLE IF NOT EXISTS team_state (
    team_id TEXT NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    asof_date TEXT NOT NULL
        CHECK (asof_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    strength REAL,
    PRIMARY KEY (team_id, model_id, asof_date)
);

-- The unified append-only prediction log. subject_id points at a series or a
-- game per subject_kind; context tags how the row was produced.
--
-- subject_kind semantics (subject_id in parentheses):
--   'series' (series_id) — p_a is the series-win probability for team_a.
--   'game'   (game_id)   — the map probability attached to a concrete played
--                          game; the eval fans these over finished games.
--   'map'    (series_id) — the map-grain matchup probability before any game
--                          exists: the model's native per-map output.
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY,
    subject_kind TEXT NOT NULL CHECK(subject_kind IN ('series', 'game', 'map')),
    subject_id TEXT NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    p_a REAL NOT NULL,
    asof_date TEXT NOT NULL
        CHECK (asof_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    context TEXT NOT NULL CHECK(context IN (__PREDICTION_CONTEXTS__)),
    created_at TEXT NOT NULL,
    UNIQUE(subject_kind, subject_id, model_id, context, created_at)
);

-- Whole-draft grades, one row per (game, model, context). edge_pp_blue is blue's calibrated
-- expected edge in percentage points (red's is exactly its negation — one stored
-- orientation); NULL under a pre-edge (quantile-semantics) grader. context carries the same
-- coverage taxonomy the predictions log does: a whole-history refit, or the walk-forward
-- out-of-sample walk. Readers take ONE row per (game, model) via db/draft_scores_read.py.
CREATE TABLE IF NOT EXISTS draft_scores (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    model_id TEXT NOT NULL REFERENCES models(model_id),
    p_blue REAL,
    grade_blue TEXT,
    grade_red TEXT,
    edge_pp_blue REAL,
    context TEXT NOT NULL DEFAULT 'backfill' CHECK(context IN (__PREDICTION_CONTEXTS__)),
    PRIMARY KEY (game_id, model_id, context)
);

-- Post-draft win probability, one row per game: the fitted stack of the served pre-draft
-- prior and the draft model's calibrated p_blue. A derived composition of two parents
-- (NOT a lifecycle token), so every row is stamped with BOTH parent model ids; a
-- served/draft retrain or backfill rebuilds it. A game missing either parent gets no row
-- (dash in the UI), never a half-composition.
CREATE TABLE IF NOT EXISTS postdraft_scores (
    game_id TEXT PRIMARY KEY REFERENCES games(game_id),
    p_blue REAL NOT NULL,
    served_model_id TEXT NOT NULL REFERENCES models(model_id),
    draft_model_id TEXT NOT NULL REFERENCES models(model_id)
);

-- The fitted post-draft stack: logit_final = a*l_served + (beta + u_blue + u_red)*l_draft,
-- blue-oriented. The global row is team_id '*' and carries a, beta, the chosen penalty and
-- holdout_gain (global holdout log loss minus team-aware, positive means the team terms
-- helped) with u = 0; every other row is a team's utilization u and its oos game count.
-- Rebuilt with postdraft_scores from the same oos parents, in the same transaction.
CREATE TABLE IF NOT EXISTS postdraft_stack (
    team_id TEXT PRIMARY KEY,
    a REAL,
    beta REAL,
    u REAL NOT NULL,
    n_oos INTEGER NOT NULL,
    penalty REAL,
    holdout_gain REAL,
    served_model_id TEXT NOT NULL,
    draft_model_id TEXT NOT NULL
);

-- Per-pick draft radar scores, one row per (game, side, role, model).
CREATE TABLE IF NOT EXISTS pick_scores (
    game_id TEXT NOT NULL REFERENCES games(game_id),
    side TEXT NOT NULL CHECK(side IN ('blue', 'red')),
    role TEXT NOT NULL,
    model_id TEXT NOT NULL REFERENCES models(model_id),
    player_id TEXT REFERENCES players(player_id),
    actual_score REAL,
    alt_count INTEGER,
    best_score REAL,
    worst_score REAL,
    percentile REAL,
    PRIMARY KEY (game_id, side, role, model_id)
);

-- Per-player Impact under a model, one row per (player_id, model_id, asof_date).
-- The impact model's backfill product (player-grain, not the predictions log):
-- feeds the dashboard Impact view.
CREATE TABLE IF NOT EXISTS player_impact_scores (
    player_id TEXT NOT NULL REFERENCES players(player_id),
    model_id TEXT NOT NULL REFERENCES models(model_id),
    asof_date TEXT NOT NULL
        CHECK (asof_date GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    role TEXT,
    impact REAL,
    stats_pp REAL,
    results_pp REAL,
    percentile REAL,
    grade TEXT,
    PRIMARY KEY (player_id, model_id, asof_date)
);

-- Upcoming matches: the PandaScore schedule plus the served model's pre-match
-- prediction (written by upcoming/service.py, shown on the dashboard's
-- Upcoming page). Deliberately OUTSIDE the predictions table: a model write
-- (wholesale replace per context, pruning earlier ids) would erase the ex-ante
-- record these rows exist to keep. One row per PandaScore match; team_a/team_b
-- are canonical team ids in team_a < team_b orientation once matched (NULL
-- until then, with the PandaScore display names kept alongside); p_a is the
-- series-win probability for team_a from the series' current state, p_a_map
-- its next map; score_a/score_b are the maps each has won so far (the feed's
-- finished games, NULL until matched); series_id links the realized series once
-- Oracle's Elixir ingests it.
CREATE TABLE IF NOT EXISTS upcoming_matches (
    ps_match_id INTEGER PRIMARY KEY,
    league_name TEXT NOT NULL,
    league_id TEXT REFERENCES leagues(league_id),
    best_of INTEGER,
    stage TEXT,
    scheduled_ts TEXT NOT NULL,
    ps_status TEXT NOT NULL,
    team_a TEXT REFERENCES teams(team_id),
    team_b TEXT REFERENCES teams(team_id),
    team_a_name TEXT,
    team_b_name TEXT,
    score_a INTEGER,
    score_b INTEGER,
    p_a REAL,
    p_a_map REAL,
    model_id TEXT REFERENCES models(model_id),
    predicted_at TEXT,
    series_id TEXT REFERENCES series(series_id),
    fetched_at TEXT NOT NULL,
    CHECK (team_a IS NULL OR team_b IS NULL OR team_a < team_b)
);

-- The durable PandaScore→canonical team identity map. PandaScore team ids are
-- stable, so a team is name-matched once ever; a row with NULL team_id records
-- a known-unmatched team awaiting a hand mapping (matched_via mirrors that).
CREATE TABLE IF NOT EXISTS pandascore_team_map (
    ps_team_id INTEGER PRIMARY KEY,
    ps_name TEXT NOT NULL,
    team_id TEXT REFERENCES teams(team_id),
    matched_via TEXT NOT NULL CHECK(matched_via IN ('name', 'manual', 'unmatched')),
    created_at TEXT NOT NULL,
    CHECK ((team_id IS NULL) = (matched_via = 'unmatched'))
);

-- Analytics marts: derived and fully rebuildable from the facts by
-- refresh_analytics_marts (analytics/builders.py). DDL only -- the ETL
-- never copies them; the refresh refills them.
CREATE TABLE IF NOT EXISTS analytics_runs (
    id INTEGER PRIMARY KEY,
    builder TEXT NOT NULL,
    started_at TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',
    details_json TEXT
);

CREATE TABLE IF NOT EXISTS team_daily_snapshot (
    team_id TEXT NOT NULL,
    date TEXT NOT NULL,
    display_name TEXT,
    region TEXT,
    roster_json TEXT NOT NULL DEFAULT '{}',
    win_rate_10 REAL NOT NULL DEFAULT 0.0,
    win_rate_5 REAL NOT NULL DEFAULT 0.0,
    win_streak INTEGER NOT NULL DEFAULT 0,
    loss_streak INTEGER NOT NULL DEFAULT 0,
    dragon_rate REAL NOT NULL DEFAULT 0.0,
    baron_rate REAL NOT NULL DEFAULT 0.0,
    tower_rate REAL NOT NULL DEFAULT 0.0,
    first_dragon_rate REAL NOT NULL DEFAULT 0.0,
    first_baron_rate REAL NOT NULL DEFAULT 0.0,
    first_tower_rate REAL NOT NULL DEFAULT 0.0,
    avg_gold_diff_15 REAL NOT NULL DEFAULT 0.0,
    avg_cs_diff_15 REAL NOT NULL DEFAULT 0.0,
    avg_xp_diff_15 REAL NOT NULL DEFAULT 0.0,
    power_rating REAL NOT NULL DEFAULT 1500.0,
    PRIMARY KEY (team_id, date)
);

CREATE TABLE IF NOT EXISTS player_daily_snapshot (
    player_id TEXT NOT NULL,
    date TEXT NOT NULL,
    team_id TEXT,
    role TEXT,
    display_name TEXT,
    champion_pool_json TEXT NOT NULL DEFAULT '{}',
    games_played INTEGER NOT NULL DEFAULT 0,
    pillars_json TEXT,
    -- Draft radar axes: five percentile-rank values (REAL, 0-100) + their sample
    -- counts (INTEGER). Columns-only storage: exactly one writer and one reader per field.
    pool_depth REAL,
    blind_quality REAL,
    counter_quality REAL,
    counter_rate REAL,
    consistency REAL,
    overall_quality REAL,
    pool_depth_n INTEGER,
    blind_quality_n INTEGER,
    counter_quality_n INTEGER,
    counter_rate_n INTEGER,
    consistency_n INTEGER,
    PRIMARY KEY (player_id, date)
);

CREATE TABLE IF NOT EXISTS power_rating_snapshots (
    date TEXT NOT NULL,
    team_id TEXT NOT NULL,
    display_name TEXT,
    region TEXT,
    rating REAL NOT NULL DEFAULT 0.0,
    global_rank INTEGER,
    PRIMARY KEY (date, team_id)
);

-- ============================================================================
-- INDEXES. PKs already cover team_state and draft_actions(game_id, seq).
-- The predictions subject index serves the per-subject prediction reads and the
-- retention deletes.
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_predictions_subject
    ON predictions(subject_kind, subject_id, context, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_games_series ON games(series_id);
CREATE INDEX IF NOT EXISTS idx_game_players_game ON game_players(game_id);
CREATE INDEX IF NOT EXISTS idx_game_players_player ON game_players(player_id);
CREATE INDEX IF NOT EXISTS idx_game_team_stats_team ON game_team_stats(team_id, game_id);
-- Hot-path tuning for the snapshot/feature stack, which issues thousands of
-- per-(team, as-of) lookups per walk; without these every one is a full
-- games/game_players scan.
CREATE INDEX IF NOT EXISTS idx_series_scheduled ON series(scheduled_ts);
CREATE INDEX IF NOT EXISTS idx_series_league ON series(league_id);
CREATE INDEX IF NOT EXISTS idx_series_team_a ON series(team_a);
CREATE INDEX IF NOT EXISTS idx_series_team_b ON series(team_b);
CREATE INDEX IF NOT EXISTS idx_games_blue_team ON games(blue_team);
CREATE INDEX IF NOT EXISTS idx_games_red_team ON games(red_team);
CREATE INDEX IF NOT EXISTS idx_game_players_team ON game_players(team_id, game_id);
CREATE INDEX IF NOT EXISTS idx_game_players_champ_role ON game_players(champion_id, role);
CREATE INDEX IF NOT EXISTS idx_game_players_player_champ ON game_players(player_id, champion_id);
CREATE INDEX IF NOT EXISTS idx_game_players_role ON game_players(role);
CREATE INDEX IF NOT EXISTS idx_oe_team_names_name ON oe_team_names(team_name);
CREATE INDEX IF NOT EXISTS idx_oe_team_names_team_id ON oe_team_names(team_id);
CREATE INDEX IF NOT EXISTS idx_player_aliases_source_name ON player_aliases(source_name);
CREATE INDEX IF NOT EXISTS idx_team_daily_snapshot_team_date
    ON team_daily_snapshot(team_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_team_daily_snapshot_date ON team_daily_snapshot(date DESC);
CREATE INDEX IF NOT EXISTS idx_player_daily_snapshot_player_date
    ON player_daily_snapshot(player_id, date DESC);
CREATE INDEX IF NOT EXISTS idx_player_daily_snapshot_date
    ON player_daily_snapshot(date DESC);
CREATE INDEX IF NOT EXISTS idx_power_rating_snapshots_date_rating
    ON power_rating_snapshots(date DESC, rating DESC);
CREATE INDEX IF NOT EXISTS idx_rosters_team ON rosters(team_id);
CREATE INDEX IF NOT EXISTS idx_rosters_player ON rosters(player_id);
CREATE INDEX IF NOT EXISTS idx_upcoming_matches_scheduled ON upcoming_matches(scheduled_ts);
""".replace("__PREDICTION_CONTEXTS__", _PREDICTION_CONTEXT_SQL)


def apply_current_schema(conn: sqlite3.Connection) -> None:
    """Apply the canonical current schema to a connection."""
    upgrade = schema_upgrade_sql(conn, CURRENT_SCHEMA_SQL)
    if upgrade is None:
        return
    try:
        conn.executescript("BEGIN IMMEDIATE;\n" + upgrade + CURRENT_SCHEMA_SQL)
        for table in (
            "leagues",
            "teams",
            "players",
            "series",
            "games",
            "game_players",
            "game_team_stats",
            "draft_actions",
            "rosters",
            "player_id_map",
            "player_aliases",
            "oe_team_names",
            "draft_scores",
            "impact_history",
        ):
            for operation in ("INSERT", "UPDATE", "DELETE"):
                conn.execute(
                    f"CREATE TRIGGER IF NOT EXISTS facts_{table}_{operation} "  # noqa: S608 - identifiers come from the fixed table and operation tuples
                    f"AFTER {operation} ON {table} BEGIN "
                    "UPDATE facts_state SET generation = generation + 1 WHERE id = 1; END"
                )
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
