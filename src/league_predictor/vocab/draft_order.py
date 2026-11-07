"""Canonical 2017+ snake-draft pick order and the 20-step ban-and-pick order."""

from __future__ import annotations

from itertools import count

#: The 20 draft steps as ``(team_index, kind)``; team 0 is the first-pick team. The facts'
#: `draft_actions` walk this order in 92% of full sequences; the rest are red-first labels.
DRAFT_ACTIONS: list[tuple[int, str]] = [
    *[(0, "ban"), (1, "ban")] * 3,
    (0, "pick"),
    (1, "pick"),
    (1, "pick"),
    (0, "pick"),
    (0, "pick"),
    (1, "pick"),
    *[(1, "ban"), (0, "ban")] * 2,
    (1, "pick"),
    (0, "pick"),
    (0, "pick"),
    (1, "pick"),
]

# Team-local pick positions in the canonical action sequence.
_pick_indexes = (count(), count())
DRAFT_INTERLEAVE = [  # sightline-ok: 26 - pick positions derive from the canonical action order
    (team, next(_pick_indexes[team])) for team, kind in DRAFT_ACTIONS if kind == "pick"
]


def remaining_turns(n_first: int, n_second: int) -> list[int]:
    """Team indexes (0 = first pick) still to pick, in canonical snake order."""
    # A board filled out of order keeps each team's outstanding picks in its canonical slots.
    placed = (n_first, n_second)
    return [ti for ti, pi in DRAFT_INTERLEAVE if pi >= placed[ti]]


def ordered_picks(
    blue_picks: list[str],
    red_picks: list[str],
    *,
    first_pick_is_blue: bool,
) -> list[tuple[str, str, bool]]:
    """The 10 picks in true draft order as ``(champion, side, is_first_pick)``."""
    # ``side`` is "blue"/"red"; role resolution is each caller's own concern.
    if first_pick_is_blue:
        teams, sides = [blue_picks, red_picks], ["blue", "red"]
    else:
        teams, sides = [red_picks, blue_picks], ["red", "blue"]
    return [(teams[ti][pi], sides[ti], ti == 0) for ti, pi in DRAFT_INTERLEAVE]
