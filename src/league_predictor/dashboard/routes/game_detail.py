"""The per-game sheet behind ``GET /api/games/{game_id}``: teams, box score, draft, prediction."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

from fastapi import HTTPException

from league_predictor.analytics.team_strength_display import latent_strength_to_display
from league_predictor.db.asof import team_state_asof
from league_predictor.db.draft_scores_read import preferred_context_predicate
from league_predictor.db.names import canonical_player_name_sql, canonical_team_name_sql
from league_predictor.db.predictions_read import (
    latest_rows_by_context,
    oos_corpus_edge,
    realized_prediction,
)
from league_predictor.predraft.pipeline.model_identity import current_rating_model_id
from league_predictor.probability import orient_blue
from league_predictor.vocab.roles import ROLE_RANK

if TYPE_CHECKING:
    import sqlite3


def build_game_detail(conn: sqlite3.Connection, game_id: str) -> dict[str, Any]:
    """Assemble the full per-game sheet: teams, box score, draft, prediction."""
    g = conn.execute(
        """
        SELECT g.game_id, g.series_id, g.game_no, g.blue_team, g.red_team,
               g.winner, g.patch, g.duration_s,
               s.league_id, s.best_of, s.team_a, s.team_b,
               date(s.scheduled_ts) AS game_date
        FROM games g JOIN series s ON s.series_id = g.series_id
        WHERE g.game_id = ?
        """,
        (game_id,),
    ).fetchone()
    if g is None:
        raise HTTPException(status_code=404, detail="Game not found")

    blue_team, red_team = g["blue_team"], g["red_team"]
    game_date = g["game_date"]

    # Sibling games in the series, for the per-game tabs.
    series_games = [
        {
            "game_id": row["game_id"],
            "game_no": row["game_no"],
            "winner_side": (
                "blue"
                if row["winner"] == row["blue_team"]
                else "red"
                if row["winner"] == row["red_team"]
                else None
            ),
        }
        for row in conn.execute(
            "SELECT game_id, game_no, blue_team, red_team, winner FROM games "
            "WHERE series_id = ? ORDER BY game_no",
            (g["series_id"],),
        )
    ]

    names = {
        blue_team: _team_name(conn, blue_team),
        red_team: _team_name(conn, red_team),
    }
    states = team_state_asof(
        conn, [blue_team, red_team], current_rating_model_id(), date.fromisoformat(game_date)
    )
    strengths = {
        tid: latent_strength_to_display(states[tid].strength if tid in states else None)
        for tid in (blue_team, red_team)
    }
    team_stats = {
        row["team_id"]: row
        for row in conn.execute("SELECT * FROM game_team_stats WHERE game_id = ?", (game_id,))
    }
    bans = _bans_by_side(conn, game_id)
    players_by_team = _players_by_team(conn, game_id)
    firstblood_team = _firstblood_team(conn, game_id)

    def side_block(team_id: str) -> dict:
        st = team_stats.get(team_id)
        players = players_by_team.get(team_id, [])
        return {
            "team_id": team_id,
            "name": names[team_id] or team_id,
            "won": (g["winner"] == team_id) if g["winner"] else None,
            "rating": strengths[team_id],
            "kills": st["teamkills"] if st else None,
            "deaths": st["teamdeaths"] if st else None,
            "gold": st["totalgold"] if st else None,
            "dragons": st["dragons"] if st else None,
            "barons": st["barons"] if st else None,
            "towers": st["towers"] if st else None,
            "heralds": st["heralds"] if st else None,
            "void_grubs": st["void_grubs"] if st else None,
            "firstblood": (firstblood_team == team_id) if firstblood_team else None,
            "firsttower": bool(st["firsttower"]) if st and st["firsttower"] is not None else None,
            "firstdragon": bool(st["firstdragon"])
            if st and st["firstdragon"] is not None
            else None,
            "firstbaron": bool(st["firstbaron"]) if st and st["firstbaron"] is not None else None,
            "bans": bans.get("blue" if team_id == blue_team else "red", []),
            "picks": [
                {k: p[k] for k in ("champion", "role", "player_id", "player_name")} for p in players
            ],
            "players": players,
        }

    return {
        "game": {
            "game_id": g["game_id"],
            "series_id": g["series_id"],
            "game_no": g["game_no"],
            "best_of": g["best_of"],
            "date": game_date,
            "league_id": g["league_id"],
            "patch": g["patch"],
            "duration_s": g["duration_s"],
        },
        "series_games": series_games,
        "blue": side_block(blue_team),
        "red": side_block(red_team),
        "draft": _draft_block(conn, game_id),
        "prediction": _prediction_block(
            conn,
            game_id,
            blue_team=blue_team,
            team_a=g["team_a"],
            team_b=g["team_b"],
            game_date=game_date,
        ),
        # Post-draft fusion (blue-oriented); None (dash) when either parent is missing.
        "post_p_blue": _postdraft_p_blue(conn, game_id),
    }


def _team_name(conn: sqlite3.Connection, team_id: str) -> str | None:
    """Canonical (newest-name-wins) display name for a team, or None if unknown."""
    row = conn.execute(f"SELECT {canonical_team_name_sql('?')} AS team_name", (team_id,)).fetchone()
    return row["team_name"]


def _bans_by_side(conn: sqlite3.Connection, game_id: str) -> dict[str, list[str | None]]:
    bans: dict[str, list[str | None]] = {"blue": [], "red": []}
    for row in conn.execute(
        "SELECT side, champion_id FROM draft_actions"
        " WHERE game_id = ? AND kind = 'ban' ORDER BY seq",
        (game_id,),
    ):
        bans[row["side"]].append(row["champion_id"])
    return bans


def _players_by_team(conn: sqlite3.Connection, game_id: str) -> dict[str, list[dict]]:
    """Role-ordered player lines per team for one game (names via aliases)."""
    rows = conn.execute(
        f"""
        SELECT gp.player_id, gp.team_id, gp.role, gp.champion_id,
               gp.kills, gp.deaths, gp.assists, gp.cspm, gp.dpm,
               gp.golddiffat15, gp.csdiffat15, gp.xpdiffat15,
               gp.killparticipation, gp.damageshare,
               {canonical_player_name_sql("gp.player_id")} AS player_name
        FROM game_players gp WHERE gp.game_id = ?
        """,
        (game_id,),
    ).fetchall()
    by_team: dict[str, list[dict]] = {}
    for row in rows:
        by_team.setdefault(row["team_id"], []).append(
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"] or row["player_id"],
                "role": row["role"],
                "champion": row["champion_id"],
                "kills": row["kills"],
                "deaths": row["deaths"],
                "assists": row["assists"],
                "cspm": row["cspm"],
                "dpm": row["dpm"],
                "gd15": row["golddiffat15"],
                "csd15": row["csdiffat15"],
                "xpd15": row["xpdiffat15"],
                "kp": row["killparticipation"],
                "dmg_share": row["damageshare"],
            }
        )
    for team_players in by_team.values():
        team_players.sort(key=lambda p: ROLE_RANK.get(p["role"], 99))
    return by_team


# sightline-ok: 11 - two unrelated facts that share a null-check, not one derivation
def _firstblood_team(conn: sqlite3.Connection, game_id: str) -> str | None:
    """First-blood team, read off the player-level flag; None when unrecorded."""
    sql = "SELECT team_id FROM game_players WHERE game_id = ? AND firstblood = 1 LIMIT 1"
    row = conn.execute(sql, (game_id,)).fetchone()
    return row["team_id"] if row else None


def _draft_block(conn: sqlite3.Connection, game_id: str) -> dict | None:
    ds = conn.execute(
        "SELECT ds.p_blue, ds.grade_blue, ds.grade_red, ds.edge_pp_blue "
        f"FROM draft_scores ds WHERE ds.game_id = ? AND {preferred_context_predicate()}",
        (game_id,),
    ).fetchone()
    if ds is None:
        return None
    picks = [
        {
            "side": row["side"],
            "role": row["role"],
            "percentile": row["percentile"],
            "actual_score": row["actual_score"],
            "best_score": row["best_score"],
            "worst_score": row["worst_score"],
        }
        for row in conn.execute(
            "SELECT side, role, percentile, actual_score, best_score, worst_score "
            "FROM pick_scores WHERE game_id = ? ORDER BY side, role",
            (game_id,),
        )
    ]
    return {
        "p_blue": ds["p_blue"],
        "grade_blue": ds["grade_blue"],
        "grade_red": ds["grade_red"],
        # Blue's calibrated expected edge (pp); red's is its negation. Null on
        # rows scored by a pre-edge grader.
        "edge_pp_blue": ds["edge_pp_blue"],
        "picks": picks,
    }


# sightline-ok: 11 - two unrelated facts that share a null-check, not one derivation
def _postdraft_p_blue(conn: sqlite3.Connection, game_id: str) -> float | None:
    """The stored post-draft fusion probability (blue-oriented) for the game, or None."""
    sql = "SELECT p_blue FROM postdraft_scores WHERE game_id = ?"
    row = conn.execute(sql, (game_id,)).fetchone()
    return row["p_blue"] if row else None


# sightline-ok: 14 - the orientation triple's home is probability.orient_blue, called below
def _prediction_block(
    conn: sqlite3.Connection,
    game_id: str,
    *,
    blue_team: str,
    team_a: str,
    team_b: str,
    game_date: str,
) -> dict | None:
    """Served prediction as a blue-side probability, through the decision rule.

    Returns:
        The walk-forward corpus row as ``source='walk'``, else a past-edge backfill row
        as ``source='live'``, else None — ``db.predictions_read.realized_prediction``,
        the same rule the games list applies.
    """
    rows = latest_rows_by_context(conn, game_id, id_prefix="served:")
    oos, backfill = rows.get("oos"), rows.get("backfill")
    p_a, source = realized_prediction(
        oos["p_a"] if oos else None,
        backfill["p_a"] if backfill else None,
        game_date=game_date,
        oos_edge=oos_corpus_edge(conn, id_prefix="served:"),
    )
    if source is None or p_a is None:
        return None
    chosen = rows["oos" if source == "walk" else "backfill"]
    return {
        "p_blue": orient_blue(p_a, blue_team, team_a, team_b),
        "asof_date": chosen["asof_date"],
        "source": source,
    }
