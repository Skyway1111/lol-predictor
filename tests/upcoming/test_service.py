"""The upcoming-pipeline verbs: schedule sync, series pairing, pre-match prediction."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from league_predictor.upcoming.pandascore import PandaMatch, PandaTeam
from league_predictor.upcoming.service import pair_realized, predict_unfinished, sync_schedule

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def _seed_team(conn, team_id: str, name: str) -> None:
    conn.execute("INSERT INTO teams (team_id, name) VALUES (?, ?)", (team_id, name))
    conn.execute("INSERT INTO oe_team_names (team_id, team_name) VALUES (?, ?)", (team_id, name))


def _match(ps_match_id=500, scheduled="2026-07-16T09:00:00+00:00", status="not_started", **kw):
    defaults = {
        "league_name": "LCK",
        "best_of": 3,
        "teams": (
            PandaTeam(ps_team_id=1, name="Beta Team"),
            PandaTeam(ps_team_id=2, name="Alpha Team"),
        ),
    }
    defaults.update(kw)
    return PandaMatch(ps_match_id=ps_match_id, scheduled_ts=scheduled, status=status, **defaults)


@pytest.fixture
def db(rebuild_db):
    from tests.conftest import register_model

    conn = rebuild_db
    register_model(conn, "served:test_model", kind="served")
    conn.execute("INSERT INTO leagues (league_id, name) VALUES ('LCK', 'LCK')")
    # ids chosen so the ALPHABETICALLY smaller id belongs to the SECOND feed slot,
    # proving orientation follows team ids, not feed order
    _seed_team(conn, "oe:team:x-beta", "Beta Team")
    _seed_team(conn, "oe:team:m-alpha", "Alpha Team")
    return conn


def _row(conn, ps_match_id=500):
    return conn.execute(
        "SELECT * FROM upcoming_matches WHERE ps_match_id = ?", (ps_match_id,)
    ).fetchone()


# ------------------------------------------------------------------ sync


def test_sync_inserts_matched_match_in_canonical_orientation(db):
    report = sync_schedule(db, [_match()], now=NOW)
    row = _row(db)
    assert (row["team_a"], row["team_b"]) == ("oe:team:m-alpha", "oe:team:x-beta")
    assert (row["team_a_name"], row["team_b_name"]) == ("Alpha Team", "Beta Team")
    assert row["league_id"] == "LCK"
    assert row["best_of"] == 3
    assert report.upserted == 1
    assert report.unmatched == []


def test_sync_persists_the_stage_for_the_feature_path(db):
    """The tournament stage lands on the row; the on-demand feature fallback reads it."""
    sync_schedule(db, [_match(tournament_name="Playoffs")], now=NOW)
    assert _row(db)["stage"] == "Playoffs"

    sync_schedule(db, [_match(tournament_name="Grand Final")], now=NOW)
    assert _row(db)["stage"] == "Grand Final"


def test_sync_keeps_unmatched_teams_visible_without_ids(db):
    match = _match(teams=(PandaTeam(9, "Mystery Org"), PandaTeam(2, "Alpha Team")))
    report = sync_schedule(db, [match], now=NOW)
    row = _row(db)
    assert row["team_a"] is None
    assert row["team_b"] is None
    assert (row["team_a_name"], row["team_b_name"]) == ("Mystery Org", "Alpha Team")
    assert report.unmatched == ["Mystery Org (ps_team_id 9)"]


def test_sync_records_tbd_slots_without_prediction_targets(db):
    match = _match(teams=(PandaTeam(2, "Alpha Team"),))
    sync_schedule(db, [match], now=NOW)
    row = _row(db)
    assert row["team_a"] is None
    assert row["team_a_name"] == "Alpha Team"
    assert row["team_b_name"] is None


def test_sync_rerun_updates_schedule_and_preserves_prediction(db):
    sync_schedule(db, [_match()], now=NOW)
    db.execute(
        "UPDATE upcoming_matches SET p_a = 0.61, p_a_map = 0.58, predicted_at = '2026-07-15'"
        " WHERE ps_match_id = 500"
    )
    sync_schedule(db, [_match(scheduled="2026-07-16T11:00:00+00:00")], now=NOW)
    row = _row(db)
    assert row["scheduled_ts"] == "2026-07-16T11:00:00+00:00"
    assert row["p_a"] == 0.61


def test_sync_raises_a_named_error_when_both_teams_map_to_one_id(db):
    """A duplicate mapping is corrupt identity data — fail loud, naming the remedy."""
    db.execute(
        "INSERT INTO pandascore_team_map (ps_team_id, ps_name, team_id, matched_via, created_at)"
        " VALUES (1, 'Beta Team', 'oe:team:m-alpha', 'manual', '2026-07-15')"
    )
    with pytest.raises(ValueError, match="upcoming-map-team"):
        sync_schedule(db, [_match()], now=NOW)


def test_sync_voids_prediction_when_the_matchup_changes(db):
    _seed_team(db, "oe:team:z-new", "New Team")
    sync_schedule(db, [_match()], now=NOW)
    _seed_series(db, "prior_series", "2026-07-16T09:00:00+00:00")
    db.execute(
        "UPDATE upcoming_matches SET p_a = 0.61, series_id = 'prior_series' WHERE ps_match_id = 500"
    )
    changed = _match(teams=(PandaTeam(3, "New Team"), PandaTeam(2, "Alpha Team")))
    sync_schedule(db, [changed], now=NOW)
    row = _row(db)
    assert row["team_b"] == "oe:team:z-new"
    assert row["p_a"] is None
    assert row["series_id"] is None


def test_sync_cancels_unstarted_rows_that_left_the_feed(db):
    sync_schedule(
        db, [_match(), _match(ps_match_id=501, scheduled="2026-07-17T09:00:00+00:00")], now=NOW
    )
    report = sync_schedule(db, [_match()], now=NOW)
    assert _row(db, 501)["ps_status"] == "canceled"
    assert _row(db, 500)["ps_status"] == "not_started"
    assert report.canceled == 1


def test_sync_never_cancels_rows_beyond_the_fetch_horizon(db):
    far_future = "2026-08-20T09:00:00+00:00"  # beyond the 14-day horizon
    sync_schedule(db, [_match(ps_match_id=502, scheduled=far_future)], now=NOW)
    sync_schedule(db, [_match()], now=NOW)
    assert _row(db, 502)["ps_status"] == "not_started"


def test_sync_writes_a_running_matchs_score_in_canonical_orientation(db):
    """Maps won follow the team ids: feed slot 1 (Beta) leads 2-0, so score_b is 2."""
    running = _match(status="running", scheduled="2026-07-15T09:00:00+00:00", wins=((1, 2),))
    sync_schedule(db, [running], now=NOW)
    row = _row(db)
    assert (row["score_a"], row["score_b"]) == (0, 2)
    assert row["ps_status"] == "finished"  # 2-0 decides a Bo3 whatever the feed still says

    bo5 = _match(status="running", scheduled="2026-07-15T09:00:00+00:00", best_of=5, wins=((1, 2),))
    sync_schedule(db, [bo5], now=NOW)
    row = _row(db)
    assert (row["score_a"], row["score_b"], row["ps_status"]) == (0, 2, "running")


# ------------------------------------------------------------------ pairing


def _seed_series(conn, series_id: str, scheduled: str) -> None:
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, best_of, status)"
        " VALUES (?, 'LCK', 'oe:team:m-alpha', 'oe:team:x-beta', ?, 3, 'completed')",
        (series_id, scheduled),
    )


def test_pair_links_the_nearest_series_for_the_same_matchup(db):
    sync_schedule(db, [_match(scheduled="2026-07-14T09:00:00+00:00")], now=NOW)
    _seed_series(db, "ser_far", "2026-07-13T05:00:00+00:00")
    _seed_series(db, "ser_near", "2026-07-14T09:30:00+00:00")
    assert pair_realized(db, now=NOW) == 1
    assert _row(db)["series_id"] == "ser_near"


def test_pair_skips_when_no_series_within_the_window(db):
    sync_schedule(db, [_match(scheduled="2026-07-14T09:00:00+00:00")], now=NOW)
    _seed_series(db, "ser_old", "2026-07-01T09:00:00+00:00")
    assert pair_realized(db, now=NOW) == 0
    assert _row(db)["series_id"] is None


def test_pair_never_reuses_an_already_paired_series(db):
    sync_schedule(
        db,
        [
            _match(ps_match_id=500, scheduled="2026-07-14T09:00:00+00:00"),
            _match(ps_match_id=501, scheduled="2026-07-14T09:10:00+00:00"),
        ],
        now=NOW,
    )
    _seed_series(db, "ser_only", "2026-07-14T09:00:00+00:00")
    assert pair_realized(db, now=NOW) == 1
    paired = [
        r["series_id"]
        for r in db.execute("SELECT series_id FROM upcoming_matches ORDER BY ps_match_id")
    ]
    assert paired.count("ser_only") == 1


def test_pair_never_gives_a_played_series_to_a_future_match(db):
    """A same-pair rematch scheduled tomorrow never steals yesterday's series."""
    sync_schedule(
        db,
        [
            _match(ps_match_id=500, scheduled="2026-07-16T09:00:00+00:00"),  # future
            _match(ps_match_id=501, scheduled="2026-07-15T09:00:00+00:00"),  # begun
        ],
        now=NOW,
    )
    _seed_series(db, "ser_played", "2026-07-15T09:05:00+00:00")
    assert pair_realized(db, now=NOW) == 1
    assert _row(db, 500)["series_id"] is None
    assert _row(db, 501)["series_id"] == "ser_played"


def test_pair_ignores_canceled_rows(db):
    sync_schedule(db, [_match(scheduled="2026-07-14T09:00:00+00:00")], now=NOW)
    db.execute("UPDATE upcoming_matches SET ps_status = 'canceled' WHERE ps_match_id = 500")
    _seed_series(db, "ser_x", "2026-07-14T09:00:00+00:00")
    assert pair_realized(db, now=NOW) == 0


# ------------------------------------------------------------------ prediction


def _fake_loader(p_map=0.6, calls=None):
    def load():
        if calls is not None:
            calls.append("loaded")
        return (lambda *_: p_map), "served:test_model"

    return load


def test_predict_scores_matched_unstarted_matches(db):
    from league_predictor.predraft.pipeline.serve import series_prob

    sync_schedule(db, [_match()], now=NOW)
    assert predict_unfinished(db, now=NOW, load_predictor=_fake_loader()) == 1
    row = _row(db)
    assert row["p_a_map"] == 0.6
    assert row["p_a"] == pytest.approx(series_prob(0.6, "Bo3"))
    assert row["model_id"] == "served:test_model"
    assert row["predicted_at"] is not None


def test_predict_skips_oracle_load_when_nothing_to_score(db):
    calls: list[str] = []
    match = _match(teams=(PandaTeam(9, "Mystery Org"), PandaTeam(2, "Alpha Team")))
    sync_schedule(db, [match], now=NOW)  # unmatched -> no candidates
    assert predict_unfinished(db, now=NOW, load_predictor=_fake_loader(calls=calls)) == 0
    assert calls == []


def test_predict_refreshes_until_start_but_never_after(db):
    sync_schedule(db, [_match()], now=NOW)
    predict_unfinished(db, now=NOW, load_predictor=_fake_loader(p_map=0.6))
    predict_unfinished(db, now=NOW, load_predictor=_fake_loader(p_map=0.7))
    assert _row(db)["p_a_map"] == 0.7  # re-predicted while still unstarted

    started = datetime(2026, 7, 16, 9, 30, tzinfo=UTC)  # past scheduled_ts
    assert predict_unfinished(db, now=started, load_predictor=_fake_loader(p_map=0.9)) == 0
    assert _row(db)["p_a_map"] == 0.7  # frozen at the last pre-start prediction


def test_predict_prices_a_running_series_from_its_score(db):
    """A running Bo5 at 0-2 scores at that state; a decided series is never rescored."""
    from league_predictor.asof import MatchContext
    from league_predictor.predraft.pipeline.serve import series_prob

    seen = {}

    def load():
        def predict(team_a, team_b, as_of):
            seen["query"] = (team_a, team_b, as_of)
            return 0.6

        return predict, "served:test_model"

    running = _match(
        status="running",
        scheduled="2026-07-15T09:00:00+00:00",
        best_of=5,
        tournament_name="Playoffs",
        wins=((1, 2),),  # Beta (team_b) leads 2-0
    )
    sync_schedule(db, [running], now=NOW)
    assert predict_unfinished(db, now=NOW, load_predictor=load) == 1
    team_a, team_b, query = seen["query"]
    assert (team_a, team_b) == ("oe:team:m-alpha", "oe:team:x-beta")
    assert query.ts == NOW.replace(tzinfo=None).isoformat() + "+00:00"  # everything before now
    assert query.context == MatchContext.of("Bo5", "Playoffs", {team_a: 0, team_b: 2})
    row = _row(db)
    assert row["p_a_map"] == 0.6
    assert row["p_a"] == pytest.approx(series_prob(0.6, "Bo5", state=(0, 2)))

    decided = _match(
        status="running", scheduled="2026-07-15T09:00:00+00:00", best_of=5, wins=((1, 3),)
    )
    sync_schedule(db, [decided], now=NOW)
    assert predict_unfinished(db, now=NOW, load_predictor=_fake_loader(p_map=0.9)) == 0
    assert _row(db)["p_a_map"] == 0.6  # frozen once the series is decided
