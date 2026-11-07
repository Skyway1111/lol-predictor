"""Shared games-page reads for the match browser and team detail."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from league_predictor.analytics.team_strength_display import latent_strength_to_display
from league_predictor.db.asof import team_state_strength_asof_select
from league_predictor.db.draft_scores_read import preferred_context_predicate
from league_predictor.db.names import canonical_team_name_sql
from league_predictor.db.predictions_read import (
    latest_p_a_prefix_select,
    oos_corpus_edge,
    realized_prediction,
)
from league_predictor.predraft.pipeline.model_identity import current_rating_model_id
from league_predictor.probability import orient_blue
from league_predictor.vocab.roles import ROLE_RANK

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping

    from league_predictor.dashboard.common import FilterScope


def _as_of_strength_sql(team_col: str) -> str:
    """Skill-filter strength for ``team_col`` strictly before the game's day (walk-forward)."""
    return team_state_strength_asof_select(team_col, "date(s.scheduled_ts)")


# The served prediction's two candidate rows; db.predictions_read owns which one wins.
_PRED_P_A_SQL = latest_p_a_prefix_select("oos", "g.game_id", id_prefix="served:")
_BACKFILL_P_A_SQL = latest_p_a_prefix_select("backfill", "g.game_id", id_prefix="served:")


def _pick_lists_for_games(
    conn: sqlite3.Connection, game_ids: list[str], blue_by_game: Mapping[str, str]
) -> dict[str, dict[str, list[str]]]:
    """Champion picks per game per side, batched for a page of games.

    Returns:
        ``{game_id: {'blue': [champ...], 'red': [champ...]}}`` — true draft pick order
        where ``draft_actions`` has it, else lane-ordered ``game_players.champion_id``.
    """
    if not game_ids:
        return {}
    placeholders = ",".join("?" * len(game_ids))
    result: dict[str, dict[str, list[str]]] = {gid: {"blue": [], "red": []} for gid in game_ids}

    drafted: set[str] = set()
    for row in conn.execute(
        f"""
        SELECT game_id, side, champion_id
        FROM draft_actions
        WHERE kind = 'pick' AND game_id IN ({placeholders})
        ORDER BY game_id, seq
        """,
        game_ids,
    ):
        gid = row["game_id"]
        if row["champion_id"]:
            result[gid][row["side"]].append(row["champion_id"])
            drafted.add(gid)

    # Lane-ordered fallback only for games with no draft_actions picks.
    undrafted = [gid for gid in game_ids if gid not in drafted]
    if undrafted:
        gp_placeholders = ",".join("?" * len(undrafted))
        rows = conn.execute(
            f"""
            SELECT game_id, team_id, role, champion_id
            FROM game_players
            WHERE game_id IN ({gp_placeholders}) AND champion_id IS NOT NULL
            """,
            undrafted,
        ).fetchall()
        rows.sort(key=lambda r: ROLE_RANK.get(r["role"], 99))
        for row in rows:
            gid = row["game_id"]
            side = "blue" if row["team_id"] == blue_by_game.get(gid) else "red"
            result[gid][side].append(row["champion_id"])
    return result


# sightline-ok: 14 - the orientation triple's home is probability.orient_blue, called below
def _pred_p_blue(pred_p_a: float | None, blue_team: str, team_a: str, team_b: str) -> float | None:
    """`p_a` is P(team_a); reorient to blue-side prob via `probability.orient_blue`. Null-safe."""
    return None if pred_p_a is None else orient_blue(pred_p_a, blue_team, team_a, team_b)


@dataclass(frozen=True)
class GamePage:
    team_id: str | None = None
    q: str | None = None
    limit: int = 25
    offset: int = 0


def fetch_games_page(
    conn: sqlite3.Connection,
    scope: FilterScope,
    page: GamePage,
) -> dict[str, Any]:
    """The shared games-list reader: `{total, games: [...]}` per the API contract.

    Args:
        scope: Date and league filters.
        page: Team and name filters, page size, and offset.

    Returns:
        Ratings and prediction as-of the game, grades from the draft model, all
        model-derived fields null-safe. One page query plus one batched pick query.
    """
    extra: list[str] = []
    params = list(scope.params)
    if page.team_id:
        extra.append("(g.blue_team = ? OR g.red_team = ?)")
        params.extend([page.team_id, page.team_id])
    if page.q:
        hit = "EXISTS(SELECT 1 FROM oe_team_names n WHERE n.team_id = {} AND n.team_name LIKE ?)"
        extra.append(f"({hit.format('g.blue_team')} OR {hit.format('g.red_team')})")
        params.extend([f"%{page.q}%"] * 2)
    where_sql = scope.where(*extra)

    page_from = "FROM games g JOIN series s ON s.series_id = g.series_id"
    total = conn.execute(f"SELECT COUNT(*) {page_from} WHERE {where_sql}", params).fetchone()[0]

    rows = conn.execute(
        f"""
        SELECT
            g.game_id, g.series_id, g.game_no, g.blue_team, g.red_team, g.winner,
            g.patch, g.duration_s,
            s.league_id, s.best_of, s.team_a, s.team_b,
            date(s.scheduled_ts) AS game_date,
            {canonical_team_name_sql("g.blue_team")} AS blue_name,
            {canonical_team_name_sql("g.red_team")}  AS red_name,
            {_as_of_strength_sql("g.blue_team")} AS blue_strength,
            {_as_of_strength_sql("g.red_team")}  AS red_strength,
            {_PRED_P_A_SQL} AS pred_p_a,
            {_BACKFILL_P_A_SQL} AS backfill_p_a,
            ds.p_blue AS draft_p_blue, ds.grade_blue, ds.grade_red, ds.edge_pp_blue,
            pds.p_blue AS post_p_blue
        {page_from}
        LEFT JOIN draft_scores ds
          ON ds.game_id = g.game_id AND {preferred_context_predicate()}
        LEFT JOIN postdraft_scores pds ON pds.game_id = g.game_id
        WHERE {where_sql}
        ORDER BY s.scheduled_ts DESC, g.game_no DESC
        LIMIT ? OFFSET ?
        """,
        [current_rating_model_id(), current_rating_model_id(), *params, page.limit, page.offset],
    ).fetchall()

    game_ids = [r["game_id"] for r in rows]
    blue_by_game = {r["game_id"]: r["blue_team"] for r in rows}
    picks = _pick_lists_for_games(conn, game_ids, blue_by_game)
    oos_edge = oos_corpus_edge(conn, id_prefix="served:")

    games = []
    for r in rows:
        gid = r["game_id"]
        game_picks = picks.get(gid, {"blue": [], "red": []})
        pred_p_a, pred_source = realized_prediction(
            r["pred_p_a"], r["backfill_p_a"], game_date=r["game_date"], oos_edge=oos_edge
        )
        games.append(
            {
                "game_id": gid,
                "series_id": r["series_id"],
                "game_no": r["game_no"],
                "best_of": r["best_of"],
                "date": r["game_date"],
                "league_id": r["league_id"],
                "patch": r["patch"],
                "duration_s": r["duration_s"],
                "blue": {
                    "team_id": r["blue_team"],
                    "name": r["blue_name"] or r["blue_team"],
                    "picks": game_picks["blue"],
                    "won": (r["winner"] == r["blue_team"]) if r["winner"] else None,
                },
                "red": {
                    "team_id": r["red_team"],
                    "name": r["red_name"] or r["red_team"],
                    "picks": game_picks["red"],
                    "won": (r["winner"] == r["red_team"]) if r["winner"] else None,
                },
                "rating_blue": latent_strength_to_display(r["blue_strength"]),
                "rating_red": latent_strength_to_display(r["red_strength"]),
                "pred_p_blue": _pred_p_blue(pred_p_a, r["blue_team"], r["team_a"], r["team_b"]),
                # 'walk' (the corpus row) or 'live' (current model, post-edge game);
                # None when pred_p_blue is null.
                "pred_source": pred_source,
                # Post-draft fusion of the pre-draft prior and the draft grade, stored
                # blue-oriented. None (dash) when either parent is missing.
                "post_p_blue": r["post_p_blue"],
                "grade_blue": r["grade_blue"],
                "grade_red": r["grade_red"],
                # Blue's calibrated expected edge (pp); red's is its negation.
                # Null on rows scored by a pre-edge grader.
                "edge_pp_blue": r["edge_pp_blue"],
            }
        )
    return {"total": total, "games": games}
