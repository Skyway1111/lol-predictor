"""Matchup and draft prediction endpoints; every response is 200 carrying ``available``."""

from __future__ import annotations

import logging
from datetime import date
from typing import TYPE_CHECKING, Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from league_predictor import clock
from league_predictor.asof import MatchContext
from league_predictor.dashboard.model_serving import DraftScorer, ServedOracle

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

_WARMING_NOTE = (
    "The oracle is warming up — the served model's feature indexes build once "
    "per dashboard run (a few minutes). Retrying automatically."
)


class MatchupRequest(BaseModel):
    team_a_id: str
    team_b_id: str
    as_of: date | None = None  # teams evaluated as snapshotted on this date; None = today
    # A series in progress: with a format the request prices the NEXT map at this score and
    # the series from it; without one the pair's scheduled match (else Bo1) is the context.
    best_of: Literal[1, 3, 5] | None = None
    score_a: int = Field(default=0, ge=0, le=2)
    score_b: int = Field(default=0, ge=0, le=2)
    stage: str | None = None


class DraftSide(BaseModel):
    picks: list[str] = []


class DraftBans(BaseModel):
    blue: list[str] = []
    red: list[str] = []


class DraftRequest(BaseModel):
    blue: DraftSide
    red: DraftSide
    bans: DraftBans | None = None


class DraftSuggestRequest(DraftRequest):
    side: Literal["blue", "red"] = "blue"  # the side making the next pick
    top_n: int = Field(default=5, ge=1, le=20)


class DraftRolloutRequest(DraftRequest):
    side: Literal["blue", "red"] = "blue"  # the side making the next pick
    # The candidates whose end-of-draft values are wanted — the whole engine
    # panel in one lockstep batch (capped well above the panel's 8 lines).
    champions: list[str] = Field(max_length=20)


def _never_500(fn, null_payload: dict) -> dict:
    """Call ``fn()``; the one seam mapping a ``ValueError``/other exception to a payload."""
    try:
        return fn()
    except ValueError as exc:
        return {"available": False, "note": str(exc), **null_payload}
    except Exception as exc:  # noqa: BLE001 - the never-500 firewall; any model failure is a payload
        return {"available": False, "note": f"{type(exc).__name__}: {exc}", **null_payload}


def build_predict_router(db_path: Path) -> APIRouter:
    """Construct routes and lazy handles (COST: milliseconds, no model inference)."""
    router = APIRouter(prefix="/api/predict", tags=["predict"])
    oracle = ServedOracle(db_path)
    scorer = DraftScorer(db_path)

    @router.post("/matchup")
    def predict_matchup(req: MatchupRequest):
        """The matchup model call site: P(team_a) for the next map, then Bo3/Bo5 from the state."""
        as_of = req.as_of or clock.utc_today()
        context = (
            MatchContext.of(
                f"Bo{req.best_of}",
                req.stage or "regular",
                {req.team_a_id: req.score_a, req.team_b_id: req.score_b},
            )
            if req.best_of is not None
            else None
        )

        def call() -> dict:
            pred = oracle.predict_series(req.team_a_id, req.team_b_id, as_of, context=context)
            if pred is None:
                failed = oracle.status == "error"
                return {
                    "available": False,
                    "status": "error" if failed else "warming",
                    "note": oracle.error if failed else _WARMING_NOTE,
                    "p_a": None,
                }
            return {
                "available": True,
                "as_of": pred.as_of.isoformat(),
                "p_a": pred.p_map,  # the next map's P(team_a) — the Bo1 number pre-series
                "p_a_bo3": pred.p_bo3,
                "p_a_bo5": pred.p_bo5,
                "score": list(pred.state),  # maps won so far (team_a, team_b) the numbers assume
            }

        return _never_500(call, {"status": "error", "p_a": None})

    @router.post("/draft")
    def predict_draft(req: DraftRequest):
        """The draft model call site: grade the (possibly partial) draft as it stands."""
        bans = req.bans or DraftBans()

        def call() -> dict:
            score = scorer.score(req.blue.picks, req.red.picks, bans.blue, bans.red)
            roles = scorer.assigned_roles(req.blue.picks, req.red.picks)
            return {
                "available": True,
                "p_blue": score.prob_blue_wins,
                "grade_blue": score.grade_blue,
                "grade_red": score.grade_red,
                "edge_pp_blue": score.edge_pp_blue,
                "roles": roles,  # champion -> role per side, the board's role tags
            }

        return _never_500(
            call, dict.fromkeys(("p_blue", "grade_blue", "grade_red", "edge_pp_blue", "roles"))
        )

    @router.post("/draft/suggest")
    def predict_draft_suggest(req: DraftSuggestRequest):
        """Fast one-ply suggestion lines for one side; end values come from the rollout call."""
        bans = req.bans or DraftBans()

        def call() -> dict:
            base, lines = scorer.suggest(
                req.blue.picks, req.red.picks, bans.blue, bans.red, side=req.side, top_n=req.top_n
            )
            return {
                "available": True,
                "side": req.side,
                "p_blue": base.prob_blue_wins,
                "suggestions": lines,
            }

        return _never_500(call, {"p_blue": None, "suggestions": []})

    @router.post("/draft/suggest/rollout")
    def predict_draft_suggest_rollout(req: DraftRolloutRequest):
        """Every visible line's end-of-draft value in one lockstep batch."""
        bans = req.bans or DraftBans()

        def call() -> dict:
            values = scorer.rollout_values(
                req.blue.picks,
                req.red.picks,
                bans.blue,
                bans.red,
                side=req.side,
                champions=req.champions,
            )
            return {
                "available": True,
                "values": [
                    {"champion": champion, "p_blue_end": value}
                    for champion, value in zip(req.champions, values, strict=True)
                ],
            }

        return _never_500(call, {"values": []})

    return router
