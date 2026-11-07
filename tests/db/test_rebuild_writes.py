"""Tests for prediction appends, team-state replacement, and facts upserts."""

from __future__ import annotations

from league_predictor.db.rows import (
    DraftActionRow,
    GamePlayerRow,
    GameRow,
    PredictionRow,
    SeriesRow,
    TeamStateRow,
)
from league_predictor.db.writes import (
    append_predictions,
    replace_team_state,
    upsert_draft_actions,
    upsert_game,
    upsert_game_players,
    upsert_series,
)
from tests.conftest import register_model


def _pred(
    context="backfill",
    created_at="2026-07-01T10:00:00+00:00",
    p_a=0.6,
    subject_id="s1",
    model_id="rating:abc",
) -> PredictionRow:
    return PredictionRow(
        subject_kind="series",
        subject_id=subject_id,
        model_id=model_id,
        p_a=p_a,
        asof_date="2026-07-01",
        context=context,
        created_at=created_at,
    )


# --- prediction log ---------------------------------------------------------


def test_append_predictions_batch_inserts_all(rebuild_db):
    register_model(rebuild_db, "rating:abc")
    rows = [_pred(subject_id=f"s{i}", created_at=f"2026-07-01T10:0{i}:00+00:00") for i in range(5)]
    append_predictions(rebuild_db, rows)
    count = rebuild_db.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    assert count == 5


# --- team_state replace -----------------------------------------------------


def test_replace_team_state_swaps_only_the_target_model(rebuild_db):
    register_model(rebuild_db, "rating:m1")
    register_model(rebuild_db, "rating:m2")
    replace_team_state(
        rebuild_db,
        "rating:m1",
        [
            TeamStateRow("A", "rating:m1", "2026-01-01", strength=1.0),
            TeamStateRow("B", "rating:m1", "2026-01-01", strength=2.0),
        ],
    )
    replace_team_state(
        rebuild_db,
        "rating:m2",
        [
            TeamStateRow("A", "rating:m2", "2026-01-01", strength=9.0),
        ],
    )

    # Re-export m1 with a single fresh row; m2 must be untouched.
    replace_team_state(
        rebuild_db,
        "rating:m1",
        [
            TeamStateRow("A", "rating:m1", "2026-02-01", strength=5.0),
        ],
    )

    m1 = rebuild_db.execute(
        "SELECT asof_date, strength FROM team_state WHERE model_id = 'rating:m1'"
    ).fetchall()
    assert [(r["asof_date"], r["strength"]) for r in m1] == [("2026-02-01", 5.0)]

    m2_count = rebuild_db.execute(
        "SELECT COUNT(*) FROM team_state WHERE model_id = 'rating:m2'"
    ).fetchone()[0]
    assert m2_count == 1


# --- facts upserts ----------------------------------------------------------


def _seed_facts(rebuild_db) -> None:
    rebuild_db.executemany("INSERT INTO teams (team_id) VALUES (?)", [("A",), ("B",)])
    rebuild_db.execute("INSERT INTO players (player_id) VALUES ('p1')")


def test_upsert_series_is_idempotent_by_series_id(rebuild_db):
    _seed_facts(rebuild_db)
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status=None, best_of=3))
    upsert_series(
        rebuild_db,
        SeriesRow("s1", team_a="A", team_b="B", status="completed", best_of=3, winner="A"),
    )
    rows = rebuild_db.execute("SELECT status, winner FROM series WHERE series_id = 's1'").fetchall()
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["winner"] == "A"


def test_upsert_series_none_never_wipes_known_values(rebuild_db):
    """winner / status / scheduled_ts COALESCE-preserve: None means "unknown", never "erase"."""
    _seed_facts(rebuild_db)
    upsert_series(
        rebuild_db,
        SeriesRow(
            "s1",
            team_a="A",
            team_b="B",
            scheduled_ts="2026-07-05T17:00:00+00:00",
            status="completed",
            winner="A",
            best_of=3,
        ),
    )

    # A schedule-source re-export that knows nothing about outcome or timing
    # must not wipe what we know.
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", best_of=3))

    row = rebuild_db.execute(
        "SELECT scheduled_ts, status, winner FROM series WHERE series_id = 's1'"
    ).fetchone()
    assert row["scheduled_ts"] == "2026-07-05T17:00:00+00:00"
    assert row["status"] == "completed"
    assert row["winner"] == "A"

    # A non-None value always applies — a reschedule moves the timestamp.
    upsert_series(
        rebuild_db,
        SeriesRow(
            "s1",
            team_a="A",
            team_b="B",
            scheduled_ts="2026-07-06T17:00:00+00:00",
            best_of=3,
        ),
    )
    moved = rebuild_db.execute("SELECT scheduled_ts FROM series WHERE series_id = 's1'").fetchone()
    assert moved["scheduled_ts"] == "2026-07-06T17:00:00+00:00"


def _series_status(conn) -> str:
    return conn.execute("SELECT status FROM series WHERE series_id = 's1'").fetchone()["status"]


def test_upsert_series_status_is_monotonic_over_terminal(rebuild_db):
    """A non-terminal status never replaces a terminal one; terminal -> terminal applies."""
    _seed_facts(rebuild_db)
    upsert_series(
        rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status="completed", winner="A")
    )

    # a stale schedule re-export must not walk the series back to scheduled
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status="scheduled"))
    assert _series_status(rebuild_db) == "completed"

    # a partial OE re-read (the facts transform's 'incomplete') must not either
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status="incomplete"))
    assert _series_status(rebuild_db) == "completed"

    # terminal -> terminal still applies (an event can be voided after completion)
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status="canceled"))
    assert _series_status(rebuild_db) == "canceled"


def test_upsert_series_non_terminal_statuses_stay_mutable(rebuild_db):
    """Before a terminal status lands, non-terminal statuses replace each other freely."""
    _seed_facts(rebuild_db)
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status="scheduled"))
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status="incomplete"))
    assert _series_status(rebuild_db) == "incomplete"
    upsert_series(
        rebuild_db, SeriesRow("s1", team_a="A", team_b="B", status="completed", winner="B")
    )
    assert _series_status(rebuild_db) == "completed"


def test_parent_upsert_preserves_children(rebuild_db):
    """Re-upserting a series or game must not cascade away its children."""
    _seed_facts(rebuild_db)
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B", best_of=3))
    upsert_game(rebuild_db, GameRow("g1", series_id="s1", game_no=1))
    upsert_game_players(
        rebuild_db,
        [
            GamePlayerRow("g1", "p1", team_id="A", role="mid", champion_id="Ahri"),
        ],
    )
    upsert_draft_actions(
        rebuild_db,
        [
            DraftActionRow("g1", seq=1, side="blue", kind="ban", champion_id="Kalista"),
        ],
    )

    upsert_series(
        rebuild_db,
        SeriesRow("s1", team_a="A", team_b="B", best_of=3, status="completed", winner="A"),
    )
    upsert_game(rebuild_db, GameRow("g1", series_id="s1", game_no=1, winner="A"))

    assert rebuild_db.execute("SELECT COUNT(*) FROM games").fetchone()[0] == 1
    assert rebuild_db.execute("SELECT COUNT(*) FROM game_players").fetchone()[0] == 1
    assert rebuild_db.execute("SELECT COUNT(*) FROM draft_actions").fetchone()[0] == 1


def test_upsert_game_is_idempotent_by_game_id(rebuild_db):
    _seed_facts(rebuild_db)
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B"))
    upsert_game(rebuild_db, GameRow("g1", series_id="s1", game_no=1, blue_team="A", red_team="B"))
    upsert_game(
        rebuild_db,
        GameRow(
            "g1", series_id="s1", game_no=1, blue_team="A", red_team="B", winner="B", patch="14.13"
        ),
    )
    rows = rebuild_db.execute("SELECT winner, patch FROM games WHERE game_id = 'g1'").fetchall()
    assert len(rows) == 1
    assert rows[0]["winner"] == "B"
    assert rows[0]["patch"] == "14.13"


def test_upsert_game_players_is_idempotent_by_grain(rebuild_db):
    _seed_facts(rebuild_db)
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B"))
    upsert_game(rebuild_db, GameRow("g1", series_id="s1", game_no=1))
    upsert_game_players(
        rebuild_db,
        [
            GamePlayerRow("g1", "p1", team_id="A", role="mid", champion_id="Ahri", kills=3),
        ],
    )
    upsert_game_players(
        rebuild_db,
        [
            GamePlayerRow("g1", "p1", team_id="A", role="mid", champion_id="Azir", kills=7),
        ],
    )
    rows = rebuild_db.execute(
        "SELECT champion_id, kills FROM game_players WHERE game_id = 'g1'"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["champion_id"] == "Azir"
    assert rows[0]["kills"] == 7


def test_upsert_draft_actions_is_idempotent_by_seq(rebuild_db):
    _seed_facts(rebuild_db)
    upsert_series(rebuild_db, SeriesRow("s1", team_a="A", team_b="B"))
    upsert_game(rebuild_db, GameRow("g1", series_id="s1", game_no=1))
    upsert_draft_actions(
        rebuild_db,
        [
            DraftActionRow("g1", seq=1, side="blue", kind="ban", champion_id="Kalista"),
            DraftActionRow("g1", seq=2, side="red", kind="ban", champion_id="Yone"),
        ],
    )
    upsert_draft_actions(
        rebuild_db,
        [
            DraftActionRow("g1", seq=1, side="blue", kind="ban", champion_id="Varus"),
        ],
    )
    rows = rebuild_db.execute(
        "SELECT seq, champion_id FROM draft_actions WHERE game_id = 'g1' ORDER BY seq"
    ).fetchall()
    assert [(r["seq"], r["champion_id"]) for r in rows] == [(1, "Varus"), (2, "Yone")]
