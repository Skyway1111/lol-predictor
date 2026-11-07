"""Tests for draft_grading/grader/swaps.py — counterfactual pick helpers."""

from __future__ import annotations

from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from league_predictor.draft_grading.grader.swaps import legal_alternatives
from league_predictor.draft_grading.state import DraftState
from tests.draft_grading.grader.conftest import make_grader_test_db


class TestLegalAlternatives:
    def test_excludes_picked(self, index: ChampionIndex):
        """Picked champions are not in the legal set."""
        db = make_grader_test_db(20)
        draft = DraftState.bulk_load(db, ["game_000"])["game_000"]
        alts = legal_alternatives(draft, index)

        picked = {s.champion_id for s in draft.blue.slots} | {
            s.champion_id for s in draft.red.slots
        }
        for champ in alts:
            assert champ not in picked

    def test_returns_nonempty(self, index: ChampionIndex):
        """There should be many legal alternatives for any position."""
        db = make_grader_test_db(20)
        draft = DraftState.bulk_load(db, ["game_000"])["game_000"]
        alts = legal_alternatives(draft, index)
        assert len(alts) > 100  # ~172 champions minus ~10 picked
