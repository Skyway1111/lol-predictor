"""Provide fixtures for Draft score writing and version reconciliation."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

ROLES = ("top", "jng", "mid", "bot", "sup")
BLUE_PICKS = {"top": "Gnar", "jng": "LeeSin", "mid": "Ahri", "bot": "Jinx", "sup": "Thresh"}
RED_PICKS = {"top": "Ksante", "jng": "Viego", "mid": "Syndra", "bot": "Caitlyn", "sup": "Lulu"}
BENCH = {
    "top": ["Aatrox", "Jax"],
    "jng": ["Elise", "Sejuani"],
    "mid": ["Azir", "Orianna"],
    "bot": ["Ezreal", "Kaisa"],
    "sup": ["Nautilus", "Rakan"],
}
# Bans written at seq 1, 2, 4 → seq 3, 5, 6 (and the whole phase-2 ban block
# 13-16) are HOLES. Detection counts only picks, so holes must not exclude the game.
_BANS = [("blue", "Kalista", 1), ("red", "Yone", 2), ("red", "Karma", 4)]

# Pick actions (seq, side, role): a blind/counter mix. seq is monotonic in pick
# order, so blue top (7) is blind vs red top (8); red top is the counter, etc.
_PICK_SEQ = [
    (7, "blue", "top"),
    (8, "red", "top"),
    (9, "red", "jng"),
    (10, "blue", "jng"),
    (11, "blue", "mid"),
    (12, "red", "mid"),
    (17, "red", "bot"),
    (18, "blue", "bot"),
    (19, "blue", "sup"),
    (20, "red", "sup"),
]

ALL_CHAMPS = sorted(
    {*BLUE_PICKS.values(), *RED_PICKS.values()}
    | {c for cs in BENCH.values() for c in cs}
    | {"Kalista", "Yone", "Karma"}
)


def role_pools() -> dict[str, set[str]]:
    """Per-role alternative universe: the bench for that role plus both picks."""
    return {r: set(BENCH[r]) | {BLUE_PICKS[r], RED_PICKS[r]} for r in ROLES}


def ability_store_json(tmp_path, names, **overrides):
    """Write a tiny ability-store JSON over exactly ``names`` and return the store.

    Args:
        overrides: champion name → feature dict; empty (the default) leaves every raw
            vector zero, so z-scoring yields a neutral corpus.
    """
    import json as _json

    from league_predictor.draft_grading.champion_attributes.ability_schema import SCHEMA_VERSION
    from league_predictor.draft_grading.champion_attributes.ability_store import (
        AbilityFeatureStore,
    )

    path = tmp_path / "ability_features.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "source_default": "prose",
                "champions": {name: dict(overrides.get(name, {})) for name in names},
                "sources": {},
            }
        ),
        encoding="utf-8",
    )
    return AbilityFeatureStore(path=path)


class _MockIndex:
    def __init__(self, champs):
        self.known_champions = list(champs)


class MockPredictor:
    """Stand-in for DraftPredictor: records swap dispatches, returns a fixed score ramp.

    Attributes:
        swap_calls: one ``(match_id, side, role, enemy_responses)`` per swap dispatch;
            ``enemy_responses`` is the distribution the pick marginalizes over.
    """

    def __init__(self, champs=ALL_CHAMPS, p_blue=0.55, edge_pp_blue=None):
        self.index = _MockIndex(champs)
        self.artifact_stem = "grader_draft_test"
        # DraftPredictor contract: the excess debias baseline the features are
        # built under; consumers build matching stats caches from it.
        self.pro_excess_baseline = "team_form"
        self._p_blue = p_blue
        # DraftScore contract: None on an analytical quantile predictor,
        # a float pp value on an edge-semantics one.
        self._edge_pp_blue = edge_pp_blue
        self.swap_calls: list[tuple] = []

    def _ramp(self, n: int) -> np.ndarray:
        return np.linspace(0.3, 0.7, n)

    def predict_swaps(self, draft, side, role, alternatives, stats_cache=None):  # noqa: ARG002 - mirrors DraftPredictor.predict_swaps; callers bind stats_cache by keyword
        enemy = draft.red if side == "blue" else draft.blue
        idx = ROLES.index(role)
        self.swap_calls.append((draft.match_id, side, role, [(enemy.slots[idx].champion_id, 1.0)]))
        return self._ramp(len(alternatives))

    def predict_swaps_marginalized(
        self,
        draft,
        side,
        role,
        alternatives,
        enemy_role,  # noqa: ARG002 - DraftPredictor.predict_swaps_marginalized binds it by keyword
        enemy_responses,
        stats_cache=None,  # noqa: ARG002 - same contract, also bound by keyword
    ):
        self.swap_calls.append((draft.match_id, side, role, list(enemy_responses)))
        return self._ramp(len(alternatives))

    def predict_swaps_marginalized_batch(self, draft, requests, stats_cache=None):  # noqa: ARG002 - mirrors DraftPredictor.predict_swaps_marginalized_batch; callers bind stats_cache by keyword
        out = []
        for req in requests:
            self.swap_calls.append((draft.match_id, req.side, req.role, list(req.enemy_responses)))
            out.append(self._ramp(len(req.alternatives)))
        return out

    def predict(self, draft, stats_cache=None):  # noqa: ARG002 - mirrors DraftPredictor.predict; callers bind stats_cache by keyword
        return SimpleNamespace(
            prob_blue_wins=self._p_blue,
            grade_blue="B",
            grade_red="C",
            edge_pp_blue=self._edge_pp_blue,
        )

    def predict_batch(self, drafts, stats_cache=None):
        return [self.predict(d, stats_cache=stats_cache) for d in drafts]


@pytest.fixture(scope="session")
def prose_reading():
    """What `ability_features` reads before precedence, keyed the way the store keys it."""
    from league_predictor.draft_grading.champion_attributes.wiki_export import (
        DEFAULT_WIKI_EXPORT_PATH,
        load_wiki_champion_export,
    )
    from scripts.champion_attributes.ability_features import extract_all_features

    wiki = load_wiki_champion_export(DEFAULT_WIKI_EXPORT_PATH)
    return {
        str(data.get("name") or key): extract_all_features(key, data) for key, data in wiki.items()
    }


@pytest.fixture
def predictor() -> MockPredictor:
    return MockPredictor()


@pytest.fixture
def draft_db(rebuild_db):
    """The canonical schema with the draft scorer's teams and players."""
    rebuild_db.executemany("INSERT INTO teams (team_id) VALUES (?)", [("T_A",), ("T_B",)])
    players = [f"bp_{r}" for r in ROLES] + [f"rp_{r}" for r in ROLES]
    rebuild_db.executemany("INSERT INTO players (player_id) VALUES (?)", [(p,) for p in players])
    rebuild_db.commit()
    return rebuild_db


@pytest.fixture(autouse=True)
def _stub_enemy_prior(monkeypatch):
    """Stub the enemy-response prior empty; a test wanting the marginalized branch re-patches."""
    import league_predictor.draft_grading.player_pick_scoring as pps

    monkeypatch.setattr(
        pps,
        "cached_enemy_response_distribution",
        lambda db, role, as_of, top_k=10: [],  # noqa: ARG005 - mirrors cached_enemy_response_distribution(db, role, as_of, top_k=...)
    )


def seed_game(
    conn: sqlite3.Connection,
    game_id: str,
    *,
    scheduled_ts: str = "2026-03-15T12:00:00+00:00",
    winner: str | None = "T_A",
    complete_picks: bool = True,
) -> str:
    """Seed one game: series + game + 10 game_players + draft_actions (bans with holes).

    Args:
        complete_picks: False drops the last pick action, leaving an unscorable 9-pick draft.
    """
    series_id = f"ser_{game_id}"
    conn.execute(
        "INSERT OR IGNORE INTO series (series_id, team_a, team_b, scheduled_ts, status) "
        "VALUES (?, ?, ?, ?, ?)",
        (series_id, "T_A", "T_B", scheduled_ts, "completed"),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (game_id, series_id, 1, "T_A", "T_B", winner),
    )
    gp = []
    for r in ROLES:
        gp.append((game_id, f"bp_{r}", "T_A", r, BLUE_PICKS[r]))
        gp.append((game_id, f"rp_{r}", "T_B", r, RED_PICKS[r]))
    conn.executemany(
        "INSERT INTO game_players (game_id, player_id, team_id, role, champion_id) "
        "VALUES (?, ?, ?, ?, ?)",
        gp,
    )
    actions = [(game_id, seq, side, "ban", champ, None) for side, champ, seq in _BANS]
    picks = _PICK_SEQ if complete_picks else _PICK_SEQ[:-1]
    for seq, side, role in picks:
        champ = (BLUE_PICKS if side == "blue" else RED_PICKS)[role]
        actions.append((game_id, seq, side, "pick", champ, role))
    conn.executemany(
        "INSERT INTO draft_actions (game_id, seq, side, kind, champion_id, role) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        actions,
    )
    conn.commit()
    return game_id


def insert_pick_row(
    conn: sqlite3.Connection,
    game_id: str,
    side: str,
    role: str,
    model_id: str,
    *,
    actual: float = 0.5,
) -> None:
    """Insert one pick_scores row directly (fixture for coexistence / reconcile)."""
    player = f"{'bp' if side == 'blue' else 'rp'}_{role}"
    conn.execute(
        "INSERT OR REPLACE INTO pick_scores "
        "(game_id, side, role, model_id, player_id, actual_score, alt_count, "
        " best_score, worst_score, percentile) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (game_id, side, role, model_id, player, actual, 3, 0.7, 0.3, 0.5),
    )
    conn.commit()


def insert_draft_row(
    conn: sqlite3.Connection,
    game_id: str,
    model_id: str,
    *,
    p_blue: float = 0.5,
) -> None:
    """Insert one draft_scores row directly (fixture for coexistence / reconcile)."""
    conn.execute(
        "INSERT OR REPLACE INTO draft_scores (game_id, model_id, p_blue, grade_blue, grade_red) "
        "VALUES (?, ?, ?, ?, ?)",
        (game_id, model_id, p_blue, "B", "C"),
    )
    conn.commit()


@pytest.fixture(scope="module")
def store():
    from league_predictor.draft_grading.champion_attributes.ability_store import AbilityFeatureStore

    return AbilityFeatureStore()


@pytest.fixture(scope="module")
def index(store):
    from league_predictor.draft_grading.grader.embeddings import ChampionIndex

    return ChampionIndex(store)
