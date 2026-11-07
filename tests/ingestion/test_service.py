"""Tests for Oracle's Elixir ingestion workflows."""

import shutil
import sqlite3


def test_run_current_year_downloads_only_that_year(tmp_path, db_template_path, monkeypatch):
    """Only the requested year is downloaded."""
    from league_predictor.ingestion.service import run_current_year

    db_path = tmp_path / "test.db"
    shutil.copy(db_template_path, db_path)
    tmp_db = sqlite3.connect(str(db_path))
    tmp_db.row_factory = sqlite3.Row

    downloaded_years = []

    def mock_download(year):
        downloaded_years.append(year)
        raise OSError("offline")

    monkeypatch.setattr("league_predictor.ingestion.service.download_csv", mock_download)

    result = run_current_year(tmp_db, 2026)
    assert downloaded_years == [2026]
    assert "matches_inserted" in result


def test_incremental_ingestion_skips_unchanged_csv(tmp_path, db_template_path, monkeypatch):
    """An unchanged CSV skips parse + insert; changed bytes or a DB lacking the year re-ingest."""
    from league_predictor.ingestion.service import run_current_year

    def _fresh_db():
        db_path = tmp_path / f"test_{len(list(tmp_path.glob('*.db')))}.db"
        shutil.copy(db_template_path, db_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        return conn

    tmp_db = _fresh_db()

    csv = tmp_path / "oe_2026.csv"
    csv.write_text("gameid\ng1\n")
    inserts: list = []

    def fake_insert(conn, df):
        # Write a real 2026 played facts row so the marker's claim ("this year is in the
        # DB") holds -- _db_has_year reads the facts (series joined to games).
        conn.execute(
            "INSERT OR IGNORE INTO series (series_id, team_a, team_b, scheduled_ts, status)"
            " VALUES ('s2026', 'A', 'B', '2026-05-01T00:00:00+00:00', 'completed')"
        )
        conn.execute(
            "INSERT OR IGNORE INTO games (game_id, series_id, game_no) VALUES ('g2026', 's2026', 1)"
        )
        inserts.append(df)
        return 3

    monkeypatch.setattr("league_predictor.ingestion.service.download_csv", lambda _year: csv)
    monkeypatch.setattr(
        "league_predictor.ingestion.oracles_elixir.load_csv", lambda _path: "sentinel-frame"
    )
    monkeypatch.setattr("league_predictor.ingestion.oracles_elixir.ingest_oe_results", fake_insert)
    monkeypatch.setattr(
        "league_predictor.ingestion.service.warn_unadjudicated_collisions",
        lambda _conn: ["unresolved handle"],
    )

    first = run_current_year(tmp_db, 2026)
    assert first["matches_inserted"] == 3
    assert first["years_skipped_unchanged"] == []

    second = run_current_year(tmp_db, 2026)
    assert len(inserts) == 1  # unchanged bytes: no second parse/insert
    assert second["years_skipped_unchanged"] == [2026]
    assert second["matches_inserted"] == 0
    assert second["unadjudicated_player_collisions"] == ["unresolved handle"]

    fresh = run_current_year(_fresh_db(), 2026)
    assert len(inserts) == 2  # marker present but DB lacks the year
    assert fresh["years_skipped_unchanged"] == []

    csv.write_text("gameid\ng1\ng2\n")  # a republished/updated CSV re-ingests
    third = run_current_year(tmp_db, 2026)
    assert len(inserts) == 3
    assert third["years_skipped_unchanged"] == []


def test_ingest_folds_brand_new_player_remint(tmp_path, db_template_path, monkeypatch):
    """The post-ingest fold pass detects and folds a brand-new remint at ingest time."""
    from league_predictor.ingestion.service import run_current_year

    db_path = tmp_path / "landmine.db"
    shutil.copy(db_template_path, db_path)
    tmp_db = sqlite3.connect(str(db_path))
    tmp_db.row_factory = sqlite3.Row

    csv = tmp_path / "oe_2026.csv"
    csv.write_text("gameid\ng1\n")

    def fake_insert(conn, _df):
        # 'old' is senior (2025). The 2026 republish lands 'new' in the same
        # (g2, T1, jng, LeeSin) seat as 'old' -> a duplicate game_players row.
        conn.executemany("INSERT OR IGNORE INTO teams (team_id) VALUES (?)", [("T1",), ("OPP",)])
        conn.executemany(
            "INSERT OR IGNORE INTO players (player_id) VALUES (?)", [("old",), ("new",)]
        )
        conn.executemany(
            "INSERT OR IGNORE INTO series (series_id, team_a, team_b, scheduled_ts, status) "
            "VALUES (?, 'OPP', 'T1', ?, 'completed')",
            [("s1", "2025-01-10T00:00:00+00:00"), ("s2", "2026-02-01T00:00:00+00:00")],
        )
        conn.executemany(
            "INSERT OR IGNORE INTO games (game_id, series_id, game_no) VALUES (?, ?, 1)",
            [("g1", "s1"), ("g2", "s2")],
        )
        conn.executemany(
            "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, result) "
            "VALUES (?, ?, 'T1', ?, 'jng', 1)",
            [("g1", "old", "Vi"), ("g2", "old", "LeeSin"), ("g2", "new", "LeeSin")],
        )
        return 2

    monkeypatch.setattr("league_predictor.ingestion.service.download_csv", lambda _year: csv)
    monkeypatch.setattr("league_predictor.ingestion.oracles_elixir.load_csv", lambda _path: "frame")
    monkeypatch.setattr("league_predictor.ingestion.oracles_elixir.ingest_oe_results", fake_insert)

    run_current_year(tmp_db, 2026)

    # The duplicate 'new' seat row is folded away; 'old' keeps both games.
    assert (
        tmp_db.execute("SELECT COUNT(*) FROM game_players WHERE player_id='new'").fetchone()[0] == 0
    )
    assert [
        r[0]
        for r in tmp_db.execute(
            "SELECT game_id FROM game_players WHERE player_id='old' ORDER BY game_id"
        )
    ] == ["g1", "g2"]
    # The adjudication is pinned durably in player_id_map.
    assert (
        tmp_db.execute("SELECT canonical_id FROM player_id_map WHERE alias_id='new'").fetchone()[0]
        == "old"
    )
    tmp_db.close()
