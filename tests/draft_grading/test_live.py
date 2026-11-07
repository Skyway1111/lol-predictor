"""Verify live draft serving: the hypothetical-draft sandbox and the dispatching loader."""

from __future__ import annotations

from league_predictor.draft_grading import live as draft_live


def test_score_hypothetical_draft_builds_a_neutral_partial_sandbox():
    from datetime import datetime

    from league_predictor.draft_grading.state import ROLE_ORDER

    when = datetime(2026, 7, 10, 12, 0)
    state = draft_live.build_hypothetical_draft(
        blue={"mid": "Ahri"},
        red={},
        blue_bans=["Aatrox"],
        red_bans=[],
        when=when,
    )
    assert state.date == when
    assert state.match_id == "sandbox"
    # Role-ordered slots; picked role carries the champion, the rest the neutral fill.
    assert tuple(s.role for s in state.blue.slots) == ROLE_ORDER
    by_role = {s.role: s.champion_id for s in state.blue.slots}
    assert by_role["mid"] == "Ahri"
    assert all(c == draft_live.UNPICKED for r, c in by_role.items() if r != "mid")
    assert all(s.champion_id == draft_live.UNPICKED for s in state.red.slots)
    # Bans: present list rides on the comp; an empty list collapses to None.
    assert state.blue.bans == ("Aatrox",)
    assert state.red.bans is None
    # Sandbox identities, never real teams.
    assert state.blue.team_id == "sandbox_blue"
    assert state.red.team_id == "sandbox_red"


# The dispatching loader (blend design D4).
