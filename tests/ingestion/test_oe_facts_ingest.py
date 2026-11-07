"""End-to-end tests for the live OE daily-results facts ingest."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from league_predictor.db.rows import SeriesRow
from league_predictor.db.schema import apply_current_schema
from league_predictor.db.series_identity import mint_series_id
from league_predictor.db.writes import upsert_series
from league_predictor.ingestion.oracles_elixir import ingest_oe_results

A, B = "oe:A", "oe:B"  # A < B, so the canonical pair is (A, B)
_ROLES = ("top", "jng", "mid", "bot", "sup")


# Local (not the hoisted ``rebuild_db``): the ingest suite hands out a persistent
# schema conn without teardown, and names it for the facts it seeds.
@pytest.fixture
def facts_conn() -> sqlite3.Connection:
    """Current-schema DB, including the player vocabulary the ingest maintains."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    apply_current_schema(conn)
    return conn


def _team_row(game_id, day, teamid, side, winner_side, gn, picks, first_pick_side) -> dict:
    won = 1 if side.lower() == winner_side else 0
    row = {
        "gameid": game_id,
        "date": f"{day} 10:00:00",
        "league": "LCK",
        "playoffs": 0,
        "game": gn,
        "gamelength": 1980,
        "patch": "14.1",
        "position": "team",
        "side": side,
        "teamid": teamid,
        "teamname": f"{teamid}-name",
        "result": won,
        "firstPick": 1 if side.lower() == first_pick_side else 0,
        "dragons": 3 if won else 1,
        "barons": 1 if won else 0,
        "towers": 9 if won else 3,
        "teamkills": 15 if won else 7,
        "teamdeaths": 7 if won else 15,
        "total cs": 900,
        "totalgold": 60000 if won else 52000,
    }
    for i, champ in enumerate(picks, start=1):
        row[f"pick{i}"] = champ
        row[f"ban{i}"] = f"{side[0]}ban{i}"
    return row


def _player_rows(game_id, day, teamid, side, winner_side, gn, picks) -> list[dict]:
    won = 1 if side.lower() == winner_side else 0
    return [
        {
            "gameid": game_id,
            "date": f"{day} 10:00:00",
            "league": "LCK",
            "game": gn,
            "position": role,
            "side": side,
            "teamid": teamid,
            "playerid": f"{teamid}_{role}",
            "playername": f"{teamid}-{role}",
            "champion": champ,
            "result": won,
            "kills": 4,
            "deaths": 3,
            "assists": 6,
            "teamkills": 15 if won else 7,
            "teamdeaths": 7 if won else 15,
        }
        for role, champ in zip(_ROLES, picks, strict=True)
    ]


def _game_rows(game_id, day, blue, red, winner_side, gn) -> list[dict]:
    blue_picks = [f"BC{i}" for i in range(1, 6)]
    red_picks = [f"RC{i}" for i in range(1, 6)]
    rows = [
        _team_row(game_id, day, blue, "Blue", winner_side, gn, blue_picks, "blue"),
        _team_row(game_id, day, red, "Red", winner_side, gn, red_picks, "blue"),
    ]
    rows += _player_rows(game_id, day, blue, "Blue", winner_side, gn, blue_picks)
    rows += _player_rows(game_id, day, red, "Red", winner_side, gn, red_picks)
    return rows


def _bo3_sweep_frame(day="2026-06-01") -> pd.DataFrame:
    """A completed Bo3, A wins 2-0. Game 1 A on blue; game 2 A on red (side swap)."""
    rows = _game_rows("g1", day, blue=A, red=B, winner_side="blue", gn=1)  # A (blue) wins
    rows += _game_rows("g2", day, blue=B, red=A, winner_side="red", gn=2)  # A (red) wins
    return pd.DataFrame(rows)


def _ambiguous_first_pick_frame(day="2026-06-02") -> pd.DataFrame:
    """One completed game where neither side claims first pick (both firstPick=0)."""
    blue_picks = [f"BC{i}" for i in range(1, 6)]
    red_picks = [f"RC{i}" for i in range(1, 6)]
    rows = [
        _team_row("h1", day, A, "Blue", "blue", 1, blue_picks, first_pick_side="none"),
        _team_row("h1", day, B, "Red", "blue", 1, red_picks, first_pick_side="none"),
    ]
    rows += _player_rows("h1", day, A, "Blue", "blue", 1, blue_picks)
    rows += _player_rows("h1", day, B, "Red", "blue", 1, red_picks)
    return pd.DataFrame(rows)


def _counts(conn) -> dict:
    return {
        t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ("series", "games", "game_players", "game_team_stats", "draft_actions")
    }


def test_ingest_writes_all_facts_tables(facts_conn):
    written = ingest_oe_results(facts_conn, _bo3_sweep_frame())
    assert written == 2  # two games

    series = facts_conn.execute("SELECT * FROM series").fetchall()
    assert len(series) == 1
    s = series[0]
    assert (s["team_a"], s["team_b"]) == (A, B)  # canonical
    assert s["winner"] == A
    assert s["status"] == "completed"
    assert s["best_of"] == 3
    assert str(s["scheduled_ts"]).startswith("2026-06-01")

    games = facts_conn.execute(
        "SELECT game_id, game_no, blue_team, red_team, winner, event_type FROM games "
        "ORDER BY game_no"
    ).fetchall()
    assert [(g["game_no"], g["blue_team"], g["red_team"], g["winner"]) for g in games] == [
        (1, A, B, A),
        (2, B, A, A),
    ]
    # the playoffs fact: OE playoffs=0 -> event_type 'regular'
    assert all(g["event_type"] == "regular" for g in games)

    assert _counts(facts_conn)["game_players"] == 20  # 10 per game
    assert _counts(facts_conn)["game_team_stats"] == 4  # 2 per game
    # a full draft per game: 10 bans + 10 picks, all slots filled
    assert _counts(facts_conn)["draft_actions"] == 40
    roles = {
        r[0]
        for r in facts_conn.execute(
            "SELECT role FROM draft_actions WHERE kind='pick' AND game_id='g1'"
        )
    }
    assert roles == set(_ROLES)


def test_unknown_first_pick_skips_ordered_draft_but_keeps_facts(facts_conn):
    # Every draft_actions seq must be a true draft slot, so an ambiguous firstPick skips the
    # ordered rows; the unordered facts (players, team stats, champion per player) still land.
    written = ingest_oe_results(facts_conn, _ambiguous_first_pick_frame())
    assert written == 1
    counts = _counts(facts_conn)
    assert counts["games"] == 1
    assert counts["game_players"] == 10
    assert counts["game_team_stats"] == 2
    assert counts["draft_actions"] == 0  # no fabricated ordering
    # The unordered pick fact survives: every champion is recorded on its player row.
    champs = {
        r[0] for r in facts_conn.execute("SELECT champion_id FROM game_players WHERE game_id='h1'")
    }
    assert champs == {f"BC{i}" for i in range(1, 6)} | {f"RC{i}" for i in range(1, 6)}


def test_unidentified_players_are_skipped_not_pooled(facts_conn):
    # OE leaves playerid blank for players it never linked. A blank id is no identity: two
    # in one game must not collapse into one row, and must not become a shared 'nan' player.
    frame = _bo3_sweep_frame()
    blank = (
        (frame["gameid"] == "g1") & (frame["teamid"] == A) & frame["position"].isin(["top", "jng"])
    )
    frame.loc[blank, "playerid"] = None

    ingest_oe_results(facts_conn, frame)

    assert _counts(facts_conn)["game_players"] == 18
    stored = {r[0] for r in facts_conn.execute("SELECT player_id FROM game_players")}
    stored |= {r[0] for r in facts_conn.execute("SELECT player_id FROM players")}
    assert None not in stored
    assert "nan" not in stored


def test_games_with_an_unidentified_team_are_skipped(facts_conn):
    # A blank OE teamid is a team OE never linked; a game against it rates no one.
    frame = _bo3_sweep_frame()
    frame.loc[(frame["gameid"] == "g2") & (frame["teamid"] == B), "teamid"] = None

    assert ingest_oe_results(facts_conn, frame) == 1

    assert [r[0] for r in facts_conn.execute("SELECT game_id FROM games")] == ["g1"]
    teams = {r[0] for r in facts_conn.execute("SELECT team_id FROM teams")}
    teams |= {r[0] for r in facts_conn.execute("SELECT team_id FROM game_players")}
    assert teams == {A, B}


def test_every_missing_required_value_is_reported_in_one_error(facts_conn):
    # A linked game whose player rows lose their teamid breaks game_players.team_id. Every
    # such row is named in one error before any write, not the first one per run.
    frame = _bo3_sweep_frame()
    players = frame["position"] != "team"
    frame.loc[players & (frame["gameid"] == "g1") & (frame["teamid"] == A), "teamid"] = None
    frame.loc[players & (frame["gameid"] == "g2") & (frame["teamid"] == B), "teamid"] = None

    with pytest.raises(ValueError, match=r"game_players\.team_id: 10 rows .*'g1', 'g2'"):
        ingest_oe_results(facts_conn, frame)

    assert _counts(facts_conn)["series"] == 0
    assert facts_conn.execute("SELECT COUNT(*) FROM oe_team_names").fetchone()[0] == 0


def test_failed_ingest_rolls_back_the_team_name_vocabulary(facts_conn, monkeypatch):
    # ingest_oe_results owns the transaction: a failure anywhere leaves the DB untouched,
    # including the oe_team_names vocabulary, which must not self-commit ahead of the facts.
    from league_predictor.ingestion import oracles_elixir

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated mid-ingest failure")

    monkeypatch.setattr(oracles_elixir, "upsert_game", _boom)
    with pytest.raises(RuntimeError):
        ingest_oe_results(facts_conn, _bo3_sweep_frame())

    assert facts_conn.execute("SELECT COUNT(*) FROM oe_team_names").fetchone()[0] == 0
    assert _counts(facts_conn)["series"] == 0  # nothing else leaked either


def test_reingest_is_idempotent(facts_conn):
    frame = _bo3_sweep_frame()
    ingest_oe_results(facts_conn, frame)
    before = _counts(facts_conn)
    ingest_oe_results(facts_conn, frame)  # OE re-export of the same day
    assert _counts(facts_conn) == before  # nothing duplicated


def test_result_merges_onto_scheduled_series(facts_conn):
    # A scheduled series filed under the deterministic (day, pair, seq=0) id...
    sid = mint_series_id("2026-06-01", A, B, 0)
    facts_conn.executemany("INSERT INTO teams (team_id) VALUES (?)", [(A,), (B,)])
    facts_conn.execute("INSERT INTO leagues (league_id, name) VALUES ('LCK', 'LCK')")
    upsert_series(
        facts_conn,
        SeriesRow(
            series_id=sid,
            team_a=A,
            team_b=B,
            league_id="LCK",
            scheduled_ts="2026-06-01T09:00:00+00:00",
            best_of=3,
            status="scheduled",
        ),
    )
    facts_conn.commit()

    # ...then the OE result lands on the SAME id: winner/status filled, games attached.
    ingest_oe_results(facts_conn, _bo3_sweep_frame())
    rows = facts_conn.execute("SELECT * FROM series").fetchall()
    assert len(rows) == 1
    assert rows[0]["series_id"] == sid
    assert rows[0]["status"] == "completed"
    assert rows[0]["winner"] == A
    assert (
        facts_conn.execute("SELECT COUNT(*) FROM games WHERE series_id=?", (sid,)).fetchone()[0]
        == 2
    )


def test_incomplete_reread_never_wipes_a_recorded_winner(facts_conn):
    # Against a re-read with winner None / status incomplete: None never erases a recorded
    # winner, and status is monotonic -- 'incomplete' never walks back 'completed'.
    ingest_oe_results(facts_conn, _bo3_sweep_frame())
    sid = mint_series_id("2026-06-01", A, B, 0)
    upsert_series(
        facts_conn, SeriesRow(series_id=sid, team_a=A, team_b=B, winner=None, status="incomplete")
    )
    facts_conn.commit()
    row = facts_conn.execute(
        "SELECT winner, status FROM series WHERE series_id=?", (sid,)
    ).fetchone()
    assert row["winner"] == A
    assert row["status"] == "completed"
