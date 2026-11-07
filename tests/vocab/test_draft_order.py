"""Tests for canonical snake-draft order."""

from __future__ import annotations

from league_predictor.vocab.draft_order import DRAFT_INTERLEAVE, ordered_picks, remaining_turns

_BLUE = ["BC1", "BC2", "BC3", "BC4", "BC5"]
_RED = ["RC1", "RC2", "RC3", "RC4", "RC5"]


def test_interleave_is_the_canonical_snake_order():
    assert DRAFT_INTERLEAVE == [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (0, 3),
        (0, 4),
        (1, 4),
    ]


def test_ordered_picks_blue_first():
    picks = ordered_picks(_BLUE, _RED, first_pick_is_blue=True)
    assert [p[0] for p in picks] == [
        "BC1",
        "RC1",
        "RC2",
        "BC2",
        "BC3",
        "RC3",
        "RC4",
        "BC4",
        "BC5",
        "RC5",
    ]
    assert [p[1] for p in picks] == [
        "blue",
        "red",
        "red",
        "blue",
        "blue",
        "red",
        "red",
        "blue",
        "blue",
        "red",
    ]
    # is_first_pick tracks the first-pick side, blue here.
    assert [p[2] for p in picks] == [s == "blue" for _, s, _ in picks]


def test_ordered_picks_red_first_swaps_sides():
    picks = ordered_picks(_BLUE, _RED, first_pick_is_blue=False)
    assert [p[1] for p in picks] == [
        "red",
        "blue",
        "blue",
        "red",
        "red",
        "blue",
        "blue",
        "red",
        "red",
        "blue",
    ]
    assert [p[2] for p in picks] == [s == "red" for _, s, _ in picks]


def test_remaining_turns_walks_the_snake_order():
    # Empty board: the full interleave. One first-pick pick placed: the
    # second-pick team's double comes next.
    assert remaining_turns(0, 0) == [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
    assert remaining_turns(1, 0) == [1, 1, 0, 0, 1, 1, 0, 0, 1]
    assert remaining_turns(4, 4) == [0, 1]
    assert remaining_turns(5, 4) == [1]
    assert remaining_turns(5, 5) == []


def test_remaining_turns_handles_off_sequence_boards():
    # A board filled out of order (dashboard slots are freely clickable) still
    # yields each team's outstanding picks in their canonical slots.
    assert remaining_turns(3, 0) == [1, 1, 1, 1, 0, 0, 1]
    assert remaining_turns(0, 5) == [0, 0, 0, 0, 0]
