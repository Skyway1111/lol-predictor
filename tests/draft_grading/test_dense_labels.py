"""The dense heads' labels: read from the facts, masked where a fact is absent, walk-forward."""

from __future__ import annotations

import sqlite3

import numpy as np
import pytest

from league_predictor.draft_grading.champion_attributes.classes import (
    CLASS_LABEL_TO_INDEX,
    UNKNOWN_CLASS,
    effective_class,
    ensure_loaded,
)
from league_predictor.draft_grading.grader.dense_labels import (
    _SQL,
    LONG_GAME_SECONDS,
    dense_targets,
)
from league_predictor.draft_grading.grader.gd15_baseline import PlayerGd15Baseline
from tests.draft_grading.grader.conftest import GRADER_TEST_SCHEMA

BLUE = ("Gnar", "LeeSin", "Ahri", "Jinx", "Thresh")
RED = ("Renekton", "Viego", "Syndra", "Caitlyn", "Lulu")
ROLES = ("top", "jng", "mid", "bot", "sup")


def _db(games: list[dict]) -> sqlite3.Connection:
    """A facts DB holding exactly ``games``; each entry names its own columns."""
    db = sqlite3.connect(":memory:")
    db.executescript(GRADER_TEST_SCHEMA)
    for i, game in enumerate(games):
        gid = game["id"]
        db.execute(
            "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, winner, "
            "status) VALUES (?, 'LCK', 'blue', 'red', ?, ?, 'completed')",
            (f"s{gid}", f"{game['day']}T00:00:00+00:00", game.get("winner")),
        )
        db.execute(
            "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner, "
            "duration_s) VALUES (?, ?, ?, 'blue', 'red', ?, ?)",
            (gid, f"s{gid}", i + 1, game.get("winner"), game.get("duration_s")),
        )
        for j, role in enumerate(ROLES):
            for team, champs in (("blue", BLUE), ("red", RED)):
                db.execute(
                    "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, "
                    "result, golddiffat15, dpm, damageshare) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?)",
                    (
                        gid,
                        f"{team}_p{j}",
                        team,
                        champs[j],
                        role,
                        game.get("gd15", {}).get((team, role), 100.0 if team == "blue" else -100.0),
                        game.get("dpm", {}).get((team, role), 400.0),
                        game.get("share", {}).get((team, role), 0.2),
                    ),
                )
    db.commit()
    return db


def _game(gid: str, day: str, **kwargs) -> dict:
    return {"id": gid, "day": day, "winner": "blue", "duration_s": 1800, **kwargs}


def test_the_slot_labels_read_the_facts_row_by_row():
    db = _db([_game("g0", "2025-06-01", dpm={("blue", "mid"): 950.0}, share={("red", "bot"): 0.4})])
    targets = dense_targets(db, ["g0"])
    assert targets.labels["dpm"][0, 2] == pytest.approx(950.0)
    assert targets.labels["damage_share"][0, 8] == pytest.approx(0.4)
    assert targets.masks["dpm"].all()
    assert targets.masks["damage_share"].all()


def test_a_missing_fact_masks_its_slot_rather_than_reading_zero():
    db = _db([_game("g0", "2025-06-01")])
    db.execute("UPDATE game_players SET dpm = NULL WHERE team_id = 'red' AND role = 'sup'")
    targets = dense_targets(db, ["g0"])
    assert targets.masks["dpm"].tolist()[0] == [True] * 9 + [False]
    assert targets.labels["dpm"][0, 9] == 0.0


def test_the_long_game_label_is_masked_to_games_past_thirty_five_minutes():
    db = _db(
        [
            _game("g0", "2025-06-01", duration_s=LONG_GAME_SECONDS),
            _game("g1", "2025-06-02", duration_s=LONG_GAME_SECONDS + 1),
            _game("g2", "2025-06-03", duration_s=LONG_GAME_SECONDS + 600, winner="red"),
        ]
    )
    targets = dense_targets(db, ["g0", "g1", "g2"])
    assert targets.masks["long_game"].tolist() == [False, True, True]
    assert targets.labels["long_game"].tolist() == [0.0, 1.0, 0.0]


def test_a_game_with_no_duration_carries_no_long_game_label():
    db = _db([_game("g0", "2025-06-01", duration_s=None)])
    assert not dense_targets(db, ["g0"]).masks["long_game"].any()


def test_the_class_label_is_the_slot_s_class_in_its_own_role():
    ensure_loaded()
    targets = dense_targets(_db([_game("g0", "2025-06-01")]), ["g0"])
    for position, (champion, role) in enumerate(
        [*zip(BLUE, ROLES, strict=True), *zip(RED, ROLES, strict=True)]
    ):
        assert (
            targets.labels["class"][0, position]
            == CLASS_LABEL_TO_INDEX[effective_class(champion, role)]
        )
    assert targets.masks["class"][0].all()


def test_an_unclustered_slot_is_masked_out_of_the_class_head():
    ensure_loaded()
    db = _db([_game("g0", "2025-06-01")])
    db.execute("UPDATE game_players SET champion_id = 'NotAChampion' WHERE role = 'top'")
    targets = dense_targets(db, ["g0"])
    unknown = CLASS_LABEL_TO_INDEX[UNKNOWN_CLASS]
    assert targets.labels["class"][0, 0] == unknown
    assert targets.masks["class"].tolist()[0] == [False, *[True] * 4, False, *[True] * 4]


# --- The gold-diff residual -----------------------------------------------------------


def test_the_first_game_s_gd15_residual_is_its_raw_team_diff():
    """Every player's baseline is 0 before their first game, so the residual is the diff."""
    db = _db([_game("g0", "2025-06-01")])
    assert dense_targets(db, ["g0"]).labels["gd15"][0] == pytest.approx(500.0)
    assert dense_targets(db, ["g0"]).masks["gd15"].all()


def test_a_later_game_reads_the_baseline_the_earlier_games_built():
    days = [f"2025-06-{d:02d}" for d in (1, 2, 3)]
    db = _db([_game(f"g{i}", day) for i, day in enumerate(days)])
    targets = dense_targets(db, ["g0", "g1", "g2"])

    baselines = PlayerGd15Baseline()
    expected = []
    for day in days:
        ordinal = np.datetime64(day).astype("datetime64[D]").astype(int) + 719163
        expected.append(
            5 * 100.0
            - sum(
                baselines.read(f"blue_p{j}", ordinal) - baselines.read(f"red_p{j}", ordinal)
                for j in range(5)
            )
        )
        for j in range(5):
            baselines.advance(f"blue_p{j}", ordinal, 100.0)
            baselines.advance(f"red_p{j}", ordinal, -100.0)
    assert targets.labels["gd15"].tolist() == pytest.approx(expected, abs=1e-4)


def test_a_games_own_result_never_enters_its_own_baseline():
    """Two corpora agreeing up to game two read the same residual for it, tail or not."""
    early = [_game("g0", "2025-06-01"), _game("g1", "2025-06-02")]
    with_tail = [*early, _game("g2", "2025-06-03")]
    assert dense_targets(_db(early), ["g1"]).labels["gd15"][0] == pytest.approx(
        dense_targets(_db(with_tail), ["g1"]).labels["gd15"][0]
    )


def test_a_lane_with_no_gold_diff_masks_the_whole_team_residual():
    db = _db([_game("g0", "2025-06-01")])
    db.execute("UPDATE game_players SET golddiffat15 = NULL WHERE team_id = 'red' AND role = 'mid'")
    assert not dense_targets(db, ["g0"]).masks["gd15"].any()


# --- Provenance -----------------------------------------------------------------------


def test_every_label_is_read_from_the_facts_and_from_no_model_table():
    named = {token.strip() for token in _SQL.replace("\n", " ").split() if token.islower()}
    assert {"game_players", "games", "series"} <= named
    assert not named & {"draft_scores", "predictions", "impact_history", "player_pick_scores"}


def test_a_requested_game_the_facts_do_not_carry_is_refused():
    db = _db([_game("g0", "2025-06-01")])
    with pytest.raises(RuntimeError, match="no facts row"):
        dense_targets(db, ["g0", "ghost"])


def test_the_row_order_follows_the_game_ids_handed_in():
    db = _db([_game("g0", "2025-06-01", dpm={("blue", "top"): 111.0}), _game("g1", "2025-06-02")])
    targets = dense_targets(db, ["g1", "g0"])
    assert targets.labels["dpm"][1, 0] == pytest.approx(111.0)
    assert len(targets) == 2
