"""Tests for OE player-identity reconciliation: remint detection, folding, collision warnings."""

import sqlite3

import pandas as pd

from league_predictor.ingestion import player_reconciler as pr


def _add_game(
    conn: sqlite3.Connection,
    game_id: str,
    date: str,
    lineups: dict[str, dict[str, tuple[str, str]]],
    *,
    series_id: str | None = None,
) -> None:
    """Insert a series + game + game_players; ``lineups`` is {team_id: {role: (player, champ)}}."""
    teams = list(lineups)
    team1 = teams[0]
    team2 = teams[1] if len(teams) > 1 else "OPP"
    team_a, team_b = sorted((team1, team2))
    sid = series_id or f"s_{game_id}"
    conn.execute(
        "INSERT OR IGNORE INTO series (series_id, team_a, team_b, scheduled_ts, "
        "status, winner) VALUES (?, ?, ?, ?, 'completed', ?)",
        (sid, team_a, team_b, date, team1),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, blue_team, red_team, winner) "
        "VALUES (?, ?, 1, ?, ?, ?)",
        (game_id, sid, team1, team2, team1),
    )
    for team_id, lineup in lineups.items():
        for role, (player_id, champion) in lineup.items():
            conn.execute(
                "INSERT INTO game_players "
                "(game_id, player_id, team_id, champion_id, role, result) "
                "VALUES (?, ?, ?, ?, ?, 1)",
                (game_id, player_id, team_id, champion, role),
            )
    conn.commit()


def _dup_slot(
    conn: sqlite3.Connection, game_id: str, player_id: str, team_id: str, champion: str, role: str
) -> None:
    """Republish an existing slot under a second player_id (the OE remint shape)."""
    conn.execute(
        "INSERT INTO game_players "
        "(game_id, player_id, team_id, champion_id, role, result) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        (game_id, player_id, team_id, champion, role),
    )
    conn.commit()


def _player_rows(conn: sqlite3.Connection, player_id: str) -> list[tuple]:
    return conn.execute(
        "SELECT game_id, team_id, champion_id, role FROM game_players "
        "WHERE player_id = ? ORDER BY game_id",
        (player_id,),
    ).fetchall()


# --- Detection -------------------------------------------------------------


def test_detect_dup_remints_finds_champion_identical_pairs(schema_db):
    """A republished slot (same game/team/role/champion, two ids) is a remint."""
    _add_game(schema_db, "m1", "2025-01-10 09:00:00", {"T1": {"jng": ("old", "LeeSin")}})
    _add_game(schema_db, "m2", "2026-01-16 09:00:00", {"T1": {"jng": ("old", "Vi")}})
    _dup_slot(schema_db, "m2", "new", "T1", "Vi", "jng")

    assert pr.detect_dup_remints(schema_db) == {"new": "old"}


def test_detect_dup_remints_keeps_champion_conflicts_split(schema_db):
    """Same-slot rows that disagree on champion are a different data problem, never folded."""
    _add_game(schema_db, "m1", "2021-02-10 09:00:00", {"T1": {"jng": ("a", "LeeSin")}})
    _dup_slot(schema_db, "m1", "b", "T1", "Olaf", "jng")

    assert pr.detect_dup_remints(schema_db) == {}


def test_detect_canonical_is_earliest_first_game(schema_db):
    """Canonical = the id with the senior history, not the lexicographic min."""
    # 'a' sorts first but 'b' has the earlier history.
    _add_game(schema_db, "m1", "2024-05-01 09:00:00", {"T1": {"top": ("b", "Jax")}})
    _add_game(schema_db, "m2", "2026-02-01 09:00:00", {"T1": {"top": ("b", "Gnar")}})
    _dup_slot(schema_db, "m2", "a", "T1", "Gnar", "top")

    assert pr.detect_dup_remints(schema_db) == {"a": "b"}


def test_resolve_chains_compresses_to_root():
    assert pr._resolve_chains({"a": "b", "b": "c"}) == {"a": "c", "b": "c"}


def test_canonical_map_curation_wins_over_detection(schema_db, monkeypatch):
    """A curated fold overrides the detected canonical for the same alias."""
    _add_game(schema_db, "m0", "2024-05-01 09:00:00", {"T1": {"jng": ("old", "LeeSin")}})
    _add_game(schema_db, "m1", "2025-01-10 09:00:00", {"T1": {"jng": ("old", "Vi")}})
    _dup_slot(schema_db, "m1", "new", "T1", "Vi", "jng")
    monkeypatch.setattr(pr, "CONFIRMED_FOLDS", {"old": "elder"})

    # detection says new->old; curation folds old->elder; chains compress.
    assert pr.canonical_player_map(schema_db) == {"new": "elder", "old": "elder"}


# --- The fold primitive ----------------------------------------------------


def _seed_split_history(schema_db) -> None:
    """One player split across two ids: m1 old only, m2 both (dup), m3 new only."""
    _add_game(schema_db, "m1", "2025-01-10 09:00:00", {"T1": {"jng": ("old", "LeeSin")}})
    _add_game(schema_db, "m2", "2026-01-16 09:00:00", {"T1": {"jng": ("old", "Vi")}})
    _dup_slot(schema_db, "m2", "new", "T1", "Vi", "jng")
    _add_game(schema_db, "m3", "2026-02-01 09:00:00", {"T1": {"jng": ("new", "Sejuani")}})
    schema_db.executemany(
        "INSERT INTO player_aliases (player_id, source, source_name) VALUES (?, 'oe', ?)",
        [("old", "POP9"), ("new", "Pop9")],
    )
    schema_db.execute(
        "INSERT INTO rosters (team_id, player_id, role, start_date) "
        "VALUES ('T1', 'new', 'jng', '2026-01-16')"
    )
    schema_db.commit()


def test_fold_dedups_repoints_and_purges(schema_db):
    _seed_split_history(schema_db)

    report = pr.fold_player_ids(schema_db, {"new": "old"})

    # m2's republished row is deleted; m3's row is re-pointed; m1 untouched.
    assert _player_rows(schema_db, "new") == []
    assert [r[0] for r in _player_rows(schema_db, "old")] == ["m1", "m2", "m3"]
    assert report["dup_rows_deleted"] == 1
    assert report["rows_repointed"] == 1

    # Both handle variants now resolve to the canonical id.
    aliases = [
        r[0]
        for r in schema_db.execute(
            "SELECT DISTINCT player_id FROM player_aliases WHERE source='oe'"
        )
    ]
    assert aliases == ["old"]

    # Derived per-player state under the dead id is purged (replays re-derive it).
    assert (
        schema_db.execute("SELECT COUNT(*) FROM rosters WHERE player_id='new'").fetchone()[0] == 0
    )


def test_fold_is_idempotent(schema_db):
    _seed_split_history(schema_db)
    pr.fold_player_ids(schema_db, {"new": "old"})

    report = pr.fold_player_ids(schema_db, {"new": "old"})

    assert report["dup_rows_deleted"] == 0
    assert report["rows_repointed"] == 0
    assert [r[0] for r in _player_rows(schema_db, "old")] == ["m1", "m2", "m3"]


def test_fold_resolves_team_remint_republish(schema_db):
    """A slot republished under a team outside the series pair drops; the real-team slot stays."""
    # Series pair is T_REAL vs OPP; the canonical id's row sits on the dangling
    # T_STALE team id, the alias row on T_REAL.
    _add_game(schema_db, "m1", "2026-02-17 17:13:02", {"T_REAL": {"mid": ("alias", "Ryze")}})
    schema_db.execute(
        "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, result) "
        "VALUES ('m1', 'canon', 'T_STALE', 'Ryze', 'mid', 1)"
    )
    schema_db.commit()

    report = pr.fold_player_ids(schema_db, {"alias": "canon"})

    rows = schema_db.execute(
        "SELECT player_id, team_id FROM game_players WHERE game_id='m1' AND role='mid'"
    ).fetchall()
    assert [tuple(r) for r in rows] == [("canon", "T_REAL")]
    assert report["dup_rows_deleted"] == 1
    assert report["conflicts_skipped"] == 0


def test_fold_never_touches_blind_pick_mirrors(schema_db):
    """A fold pair on opposite in-record teams is a blind-pick mirror game; both rows stay."""
    _add_game(
        schema_db,
        "m1",
        "2025-12-20 08:40:54",
        {"TA": {"mid": ("alias", "Jayce")}, "TB": {"mid": ("canon", "Jayce")}},
    )

    report = pr.fold_player_ids(schema_db, {"alias": "canon"})

    assert len(_player_rows(schema_db, "alias")) == 1
    assert len(_player_rows(schema_db, "canon")) == 1
    assert report["conflicts_skipped"] == 1


def test_fold_skips_champion_conflict_matches(schema_db):
    """A same-game champion conflict inside a fold pair leaves both rows alone and reports."""
    _add_game(schema_db, "m1", "2025-01-10 09:00:00", {"T1": {"jng": ("old", "Vi")}})
    _dup_slot(schema_db, "m1", "new", "T1", "Olaf", "jng")

    report = pr.fold_player_ids(schema_db, {"new": "old"})

    assert report["conflicts_skipped"] == 1
    assert len(_player_rows(schema_db, "new")) == 1
    assert len(_player_rows(schema_db, "old")) == 1


def test_fold_map_survives_evidence_consumption(schema_db, monkeypatch):
    """The stored player_id_map carries the pair after folding consumes the dup evidence."""
    monkeypatch.setattr(pr, "CONFIRMED_FOLDS", {})
    _seed_split_history(schema_db)
    pr.fold_player_ids(schema_db)  # mapping derived from dup evidence

    # The dup evidence is gone, but the stored map still carries the pair.
    assert pr.detect_dup_remints(schema_db) == {}
    assert pr.canonical_player_map(schema_db) == {"new": "old"}

    # A republished vintage row under the dead alias re-folds via the map.
    schema_db.execute(
        "INSERT INTO game_players (game_id, player_id, team_id, champion_id, role, result) "
        "VALUES ('m3', 'new', 'T1', 'Sejuani', 'jng', 1)"
    )
    schema_db.commit()
    report = pr.fold_player_ids(schema_db)
    assert report["dup_rows_deleted"] == 1  # the republished slot folds away


# --- DataFrame fold (pre-insert seam) ---------------------------------------


def test_apply_canonical_ids_folds_dataframe():
    df = pd.DataFrame(
        {
            "playerid": ["new", "other", None],
            "playername": ["Pop9", "Other", None],
        }
    )

    out = pr.apply_canonical_ids(df, {"new": "old"})

    assert list(out["playerid"][:2]) == ["old", "other"]
    assert pd.isna(out["playerid"][2])  # missing ids pass through untouched
    # The input frame is not mutated.
    assert list(df["playerid"][:2]) == ["new", "other"]


def test_apply_canonical_ids_noop_returns_input():
    df = pd.DataFrame({"playerid": ["a", "b"]})
    assert pr.apply_canonical_ids(df, {"zzz": "y"}) is df
    assert pr.apply_canonical_ids(df, {}) is df


# --- Collision surfacing ----------------------------------------------------


def test_warn_flags_unadjudicated_casing_collisions(schema_db, monkeypatch):
    _add_game(schema_db, "m1", "2025-01-10 09:00:00", {"T1": {"jng": ("id1", "Vi")}})
    _add_game(schema_db, "m2", "2026-01-10 09:00:00", {"T2": {"jng": ("id2", "Olaf")}})
    schema_db.executemany(
        "INSERT INTO player_aliases (player_id, source, source_name) VALUES (?, 'oe', ?)",
        [("id1", "MYSTERY"), ("id2", "Mystery")],
    )
    schema_db.commit()
    monkeypatch.setattr(pr, "CONFIRMED_DISTINCT", frozenset())

    assert pr.warn_unadjudicated_collisions(schema_db) == ["mystery"]


def test_warn_suppresses_adjudicated_and_zero_game_ids(schema_db, monkeypatch):
    _add_game(schema_db, "m1", "2025-01-10 09:00:00", {"T1": {"jng": ("id1", "Vi")}})
    _add_game(schema_db, "m2", "2026-01-10 09:00:00", {"T2": {"jng": ("id2", "Olaf")}})
    schema_db.executemany(
        "INSERT INTO player_aliases (player_id, source, source_name) VALUES (?, 'oe', ?)",
        [
            ("id1", "TAEYOON"),
            ("id2", "Taeyoon"),  # adjudicated distinct
            ("id3", "GHOST"),
            ("id4", "Ghost"),  # id3/id4 have no games
        ],
    )
    schema_db.commit()
    monkeypatch.setattr(pr, "CONFIRMED_DISTINCT", frozenset({"taeyoon"}))

    assert pr.warn_unadjudicated_collisions(schema_db) == []


def test_warn_suppresses_pairs_already_folded(schema_db):
    _add_game(schema_db, "m1", "2025-01-10 09:00:00", {"T1": {"jng": ("old", "Vi")}})
    _add_game(schema_db, "m2", "2026-01-10 09:00:00", {"T1": {"jng": ("new", "Olaf")}})
    schema_db.executemany(
        "INSERT INTO player_aliases (player_id, source, source_name) VALUES (?, 'oe', ?)",
        [("old", "POP9"), ("new", "Pop9")],
    )
    schema_db.commit()
    pr.fold_player_ids(schema_db, {"new": "old"})

    assert pr.warn_unadjudicated_collisions(schema_db) == []
