"""Verify the Original feature family's index, its persisted blob and its on-demand fallback."""

from __future__ import annotations

import itertools
from datetime import date

import numpy as np
import pytest

from league_predictor import config
from league_predictor.asof import AsOf
from league_predictor.predraft.features.families import original
from league_predictor.predraft.features.families.original import _OriginalFeaturesIndex


def _key(day: str, game_id: str) -> str:
    """The entry key of a fixture map: its series starts at 10:00 on ``day``, one game."""
    return AsOf.game(f"{day}T10:00:00+00:00", f"ser_{game_id}", 1).key


def _index(rows: dict[tuple[str, str, str], list[float]], names: list[str]):
    keys = list(rows)
    return _OriginalFeaturesIndex(np.array([rows[k] for k in keys]), keys, names)


def test_features_reads_the_stored_row_by_name():
    index = _index({("A", "B", "2026-01-01"): [2.0, 0.0]}, ["diff_x", "diff_y"])
    assert index.features("A", "B", AsOf.of(date(2026, 1, 1))) == {"diff_x": 2.0, "diff_y": 0.0}


def test_features_matrix_matches_dict_rows_in_requested_column_order():
    index = _index(
        {("A", "B", "2026-01-01"): [2.0, 0.0], ("B", "A", "2026-01-01"): [-2.0, 3.0]},
        ["diff_x", "diff_y"],
    )
    q = AsOf.of(date(2026, 1, 1))
    assert index.features_matrix([("A", "B", q), ("B", "A", q)], ["diff_y", "diff_x"]).tolist() == [
        [0.0, 2.0],
        [3.0, -2.0],
    ]


def test_features_matrix_rejects_an_undeclared_column():
    index = _index({}, ["diff_x"])
    with pytest.raises(KeyError, match="unknown original feature"):
        index.features_matrix([], ["typo"])


def test_unknown_matchup_raises_rather_than_zero_filling():
    index = _index({}, ["diff_x"])
    with pytest.raises(KeyError):
        index.features("A", "B", AsOf.of(date(2026, 1, 1)))


@pytest.fixture
def facts_db(tmp_path):
    """File-backed facts-schema DB, so the index build persists a blob."""
    import sqlite3

    from league_predictor.db.schema import apply_current_schema
    from tests.predraft.conftest import ensure_event_type_column

    conn = sqlite3.connect(str(tmp_path / "facts.db"))
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    ensure_event_type_column(conn)  # _match_context reads games.event_type
    conn.commit()
    yield conn
    conn.close()


def test_match_context_reads_event_type_as_stage_and_prior_maps_as_score(facts_db):
    """``_match_context`` maps each game to (format, stage, wins_a, wins_b) before that map."""
    from tests.predraft.conftest import insert_game

    insert_game(
        facts_db,
        game_id="g_po",
        day="2026-06-01",
        team1="A",
        team2="B",
        winner="A",
        best_of=5,
        event_type="playoffs",
    )
    insert_game(
        facts_db,
        game_id="g_reg",
        day="2026-06-02",
        team1="A",
        team2="B",
        winner="A",
        best_of=1,
        event_type="regular",
    )
    series = insert_game(
        facts_db, game_id="g_null", day="2026-06-03", team1="A", team2="B", winner="A", best_of=3
    )
    insert_game(
        facts_db,
        game_id="g_two",
        day="2026-06-03",
        team1="A",
        team2="B",
        winner="B",
        series_id=series,
        game_no=2,
    )
    insert_game(
        facts_db,
        game_id="g_three",
        day="2026-06-03",
        team1="A",
        team2="B",
        winner="B",
        series_id=series,
        game_no=3,
    )
    facts_db.commit()

    ctx = original._match_context(facts_db)
    assert ctx["g_po"] == ("Bo5", "playoffs", 0, 0)
    assert ctx["g_reg"] == ("Bo1", "regular", 0, 0)
    assert ctx["g_null"] == ("Bo3", "regular", 0, 0)  # event_type NULL -> 'regular'
    assert ctx["g_two"] == ("Bo3", "regular", 1, 0)  # A took map one
    assert ctx["g_three"] == ("Bo3", "regular", 1, 1)  # never the map's own result


def test_is_playoffs_flows_from_event_type(facts_db):
    """The stage read from ``games.event_type`` drives the served ``is_playoffs`` column."""
    from league_predictor.predraft.features.snapshots import build_pre_draft_features
    from tests.predraft.conftest import insert_game

    insert_game(
        facts_db,
        game_id="g1",
        day="2026-05-01",
        team1="A",
        team2="B",
        winner="A",
        event_type="playoffs",
    )
    facts_db.commit()

    _fmt, stage, _wins_a, _wins_b = original._match_context(facts_db)["g1"]
    playoff_row = build_pre_draft_features(
        facts_db, "A", "B", "Bo5", stage, date(2026, 6, 1), prune=False
    )
    regular_row = build_pre_draft_features(
        facts_db, "A", "B", "Bo1", "regular", date(2026, 6, 1), prune=False
    )
    assert playoff_row["is_playoffs"] == 1.0
    assert regular_row["is_playoffs"] == 0.0


def _seed_scheduled_playoff(conn) -> None:
    """One played game plus a scheduled Bo5 playoff match for the same pair."""
    from tests.predraft.conftest import insert_game

    conn.execute("INSERT OR IGNORE INTO teams (team_id) VALUES ('A'), ('B')")
    conn.execute("INSERT OR IGNORE INTO leagues (league_id, name) VALUES ('LCK', 'LCK')")
    insert_game(conn, game_id="g1", day="2026-05-01", team1="A", team2="B", winner="A")
    conn.execute(
        "INSERT INTO upcoming_matches (ps_match_id, league_name, best_of, scheduled_ts,"
        " ps_status, team_a, team_b, stage, fetched_at) VALUES"
        " (1, 'LCK', 5, '2026-06-02T09:00:00+00:00', 'not_started', 'A', 'B', 'Playoffs', 'x')"
    )
    conn.commit()


def test_on_demand_fallback_prices_the_scheduled_matchs_true_stage(rebuild_db):
    """An uncovered matchup with a scheduled playoff match reads its true stage and format."""
    from league_predictor.predraft.features import FEATURE_FAMILIES

    _seed_scheduled_playoff(rebuild_db)
    index = FEATURE_FAMILIES["original"]().fallback_index(rebuild_db)

    row = index.features("A", "B", AsOf.of(date(2026, 6, 1)))
    reversed_side = index.features("B", "A", AsOf.of(date(2026, 6, 1)))

    assert row["is_playoffs"] == 1.0
    assert row["format_bo"] == 5.0
    # Both scoring orientations resolve the SAME scheduled match.
    assert reversed_side["is_playoffs"] == 1.0
    assert reversed_side["format_bo"] == 5.0


def test_on_demand_fallback_defaults_regular_when_nothing_is_scheduled(rebuild_db, facts_db):
    """No scheduled match — or no schedule table at all — keeps the neutral Bo1/regular."""
    from league_predictor.predraft.features import FEATURE_FAMILIES
    from tests.predraft.conftest import insert_game

    rebuild_db.execute("INSERT OR IGNORE INTO teams (team_id) VALUES ('A'), ('B')")
    rebuild_db.execute("INSERT OR IGNORE INTO leagues (league_id, name) VALUES ('LCK', 'LCK')")
    insert_game(rebuild_db, game_id="g1", day="2026-05-01", team1="A", team2="B", winner="A")
    rebuild_db.commit()
    row = (
        FEATURE_FAMILIES["original"]()
        .fallback_index(rebuild_db)
        .features("A", "B", AsOf.of(date(2026, 6, 1)))
    )
    assert row["is_playoffs"] == 0.0
    assert row["format_bo"] == 1.0

    # facts_db carries no upcoming_matches table; the resolver must default, not raise.
    insert_game(facts_db, game_id="g2", day="2026-05-01", team1="A", team2="B", winner="A")
    facts_db.commit()
    bare = (
        FEATURE_FAMILIES["original"]()
        .fallback_index(facts_db)
        .features("A", "B", AsOf.of(date(2026, 6, 1)))
    )
    assert bare["is_playoffs"] == 0.0
    assert bare["format_bo"] == 1.0


def test_swap_built_features_is_exactly_the_reverse_build(facts_db):
    from league_predictor.predraft.features.snapshots import SnapshotStore, build_pre_draft_features
    from tests.predraft.conftest import insert_game

    insert_game(
        facts_db,
        game_id="g_swap",
        day="2026-05-01",
        team1="A",
        team2="B",
        winner="A",
        event_type="playoffs",
    )
    facts_db.commit()
    as_of, store = AsOf.of(date(2026, 6, 1)), SnapshotStore(facts_db, teams=["A", "B"])
    forward = build_pre_draft_features(
        facts_db, "A", "B", "Bo5", "playoffs", as_of, prune=False, store=store
    )

    swapped = original._swap_built_features(store, forward, team_a="A", team_b="B", as_of=as_of)
    reference = build_pre_draft_features(
        facts_db, "B", "A", "Bo5", "playoffs", as_of, prune=False, store=store
    )

    assert swapped == reference


def _insert_map_with_roster(conn, *, game_id: str, day: str, a: str, b: str) -> None:
    """One eligible one-game series with 5-man rosters on both sides."""
    ta, tb = sorted((a, b))
    conn.execute(
        "INSERT INTO series (series_id, league_id, team_a, team_b, scheduled_ts, best_of, "
        "status) VALUES (?, 'LCK', ?, ?, ?, 1, 'completed')",
        (f"ser_{game_id}", ta, tb, f"{day}T10:00:00+00:00"),
    )
    conn.execute(
        "INSERT INTO games (game_id, series_id, game_no, winner, duration_s) "
        "VALUES (?, ?, 1, ?, 1800)",
        (game_id, f"ser_{game_id}", a),
    )
    for tid in (a, b):
        for r in ("top", "jng", "mid", "bot", "sup"):
            conn.execute(
                "INSERT INTO game_players (game_id, player_id, team_id, role, "
                "kills, deaths, assists) VALUES (?, ?, ?, ?, 5, 5, 10)",
                (game_id, f"{tid}_{r}", tid, r),
            )
    conn.commit()


def _populate_minimal(conn) -> None:
    """Round-robin matches with 5-man rosters: enough for ``build_pre_draft_features`` to run."""
    teams = ["t_aaa", "t_bbb", "t_ccc", "t_ddd"]
    n, day = 0, 1
    for _ in range(3):
        for a, b in itertools.combinations(teams, 2):
            _insert_map_with_roster(
                conn,
                game_id=f"m{n}",
                day=f"2026-03-{day:02d}",
                a=a if n % 2 == 0 else b,
                b=b if n % 2 == 0 else a,
            )
            n += 1
            day += 1


@pytest.fixture
def featurized_spans(monkeypatch):
    """Record every ``[start, through)`` span the index featurizes."""
    spans: list[tuple[date, date]] = []
    real = original._featurize

    def spy(conn, names, *, start, through):
        spans.append((start, through))
        return real(conn, names, start=start, through=through)

    monkeypatch.setattr(original, "_featurize", spy)
    return spans


def test_index_covers_every_eligible_map_in_both_orientations(facts_db):
    _populate_minimal(facts_db)
    index = original.build_original_index(facts_db, through=date(2026, 4, 1))

    assert len(index._rows) == 2 * 18
    assert index.feature_names == sorted(original.full_feature_universe())
    key = ("t_aaa", "t_bbb", _key("2026-03-01", "m0"))
    assert key in index._positions
    assert index.features(*key[:2], AsOf.parse(key[2]))["format_bo"] == 1.0
    # A map after ``through`` is not in the blob; the conn-backed index computes it on demand.
    _insert_map_with_roster(facts_db, game_id="late", day="2026-04-05", a="t_aaa", b="t_bbb")
    assert (
        index.features("t_aaa", "t_bbb", AsOf.parse(_key("2026-04-05", "late")))["is_t1_blue_side"]
        == 1.0
    )


def test_moved_through_extends_the_prior_blob_instead_of_full_rebuild(
    facts_db, tmp_path, monkeypatch, featurized_spans
):
    """A moved ``through`` featurizes only the tail after the newest valid blob; equals scratch."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache")
    t1, t2 = date(2026, 5, 3), date(2026, 5, 6)
    _insert_map_with_roster(facts_db, game_id="g_old1", day="2026-05-01", a="t_aaa", b="t_bbb")
    _insert_map_with_roster(facts_db, game_id="g_old2", day="2026-05-02", a="t_aaa", b="t_ccc")
    original.build_original_index(facts_db, through=t1)  # seed the blob
    assert featurized_spans == [(date.min, t1)]

    _insert_map_with_roster(facts_db, game_id="g_new", day="2026-05-05", a="t_bbb", b="t_ccc")
    featurized_spans.clear()
    extended = original.build_original_index(facts_db, through=t2)
    assert featurized_spans == [(t1, t2)]  # ONLY the tail after the prior cutoff

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache_scratch")
    featurized_spans.clear()
    fresh = original.build_original_index(facts_db, through=t2)
    assert featurized_spans == [(date.min, t2)]  # a genuine from-scratch reference
    assert extended._rows == fresh._rows
    assert extended._matrix.tobytes() == fresh._matrix.tobytes()
    assert ("t_bbb", "t_ccc", _key("2026-05-05", "g_new")) in extended._positions

    # The extension persisted under the new end: a repeat build is a pure load.
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache")
    featurized_spans.clear()
    again = original.build_original_index(facts_db, through=t2)
    assert featurized_spans == []
    assert again._rows == extended._rows
    assert (
        len(list((tmp_path / "cache" / "cache").glob("original_*.pkl"))) == 2
    )  # the group keeps two


def test_extension_is_refused_after_a_historical_change(
    facts_db, tmp_path, monkeypatch, featurized_spans
):
    """An ingest dated before the blob's cutoff fails its facts digest, forcing a full build."""
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache")
    t1, t2 = date(2026, 5, 3), date(2026, 5, 6)
    _insert_map_with_roster(facts_db, game_id="g_old", day="2026-05-01", a="t_aaa", b="t_bbb")
    original.build_original_index(facts_db, through=t1)

    _insert_map_with_roster(facts_db, game_id="g_backfill", day="2026-04-25", a="t_aaa", b="t_ccc")
    featurized_spans.clear()
    rebuilt = original.build_original_index(facts_db, through=t2)
    assert featurized_spans == [(date.min, t2)]  # full span, not a tail
    assert ("t_aaa", "t_ccc", _key("2026-04-25", "g_backfill")) in rebuilt._positions


def test_a_schema_change_mints_a_new_blob_group(facts_db, tmp_path, monkeypatch, featurized_spans):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache")
    _insert_map_with_roster(facts_db, game_id="g1", day="2026-05-01", a="t_aaa", b="t_bbb")
    through = date(2026, 5, 3)
    original.build_original_index(facts_db, through=through)
    featurized_spans.clear()
    original.build_original_index(facts_db, through=through)
    assert featurized_spans == []  # an exact hit

    monkeypatch.setattr(original, "_CACHE_VERSION", original._CACHE_VERSION + 1)
    original.build_original_index(facts_db, through=through)
    assert featurized_spans == [(date.min, through)]
    assert (
        len(list((tmp_path / "cache" / "cache").glob("original_*.pkl"))) == 2
    )  # two groups coexist


def test_in_memory_db_builds_without_persisting(tmp_path, monkeypatch, featurized_spans):
    import sqlite3

    from league_predictor.db.schema import apply_current_schema

    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "cache")
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    apply_current_schema(conn)
    _insert_map_with_roster(conn, game_id="g1", day="2026-05-01", a="t_aaa", b="t_bbb")

    index = original.build_original_index(conn, through=date(2026, 5, 3))
    assert len(index._rows) == 2
    assert featurized_spans == [(date.min, date(2026, 5, 3))]
    assert not (tmp_path / "cache").exists()
