"""Feature-name parity between the live builder and the training pipeline; needs the prod DB."""

from __future__ import annotations

import os
from datetime import date as _date
from pathlib import Path

import pytest

from league_predictor.config import DB_PATH

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not Path(DB_PATH).exists(),
        reason="Production DB not available",
    ),
]


@pytest.fixture(scope="module")
def prod_conn():
    from league_predictor.db import get_connection

    path = os.environ.get("REBUILD_SMOKE_DB") or DB_PATH
    conn = get_connection(path)
    required = {"games", "series", "oe_team_names"}
    tables = {
        row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    if not required.issubset(tables):
        conn.close()
        pytest.skip(
            "facts schema missing required tables; set REBUILD_SMOKE_DB to a rebuilt target"
        )
    yield conn
    conn.close()


def _sample_pairs(conn, limit: int = 8) -> list[tuple[str, str, str]]:
    """Pick recent finished matches with distinct unordered team pairs."""
    rows = conn.execute(
        """
        SELECT s.team_a AS team1_id, s.team_b AS team2_id,
               MAX(s.scheduled_ts) AS d
        FROM games g JOIN series s ON s.series_id = g.series_id
        WHERE date(s.scheduled_ts) >= '2026-03-01'
          AND g.winner IS NOT NULL
        GROUP BY s.team_a, s.team_b
        ORDER BY d DESC
        """
    ).fetchall()

    seen: set[frozenset[str]] = set()
    out: list[tuple[str, str, str]] = []
    for row in rows:
        key = frozenset({row["team1_id"], row["team2_id"]})
        if key in seen:
            continue
        seen.add(key)
        out.append((row["team1_id"], row["team2_id"], str(row["d"])[:10]))
        if len(out) >= limit:
            break
    return out


def test_live_builder_emits_is_t1_blue_side(prod_conn):
    """The live feature builder must always emit is_t1_blue_side."""
    from league_predictor.predraft.features.snapshots import build_pre_draft_features

    pairs = _sample_pairs(prod_conn, limit=1)
    if not pairs:
        pytest.skip("no sample matches")
    t1, t2, d = pairs[0]
    feats = build_pre_draft_features(
        conn=prod_conn,
        team1_id=t1,
        team2_id=t2,
        match_format="Bo3",
        stage="regular",
        as_of=_date.fromisoformat(d),
    )
    assert "is_t1_blue_side" in feats
    assert feats["is_t1_blue_side"] in (0.0, 1.0)
