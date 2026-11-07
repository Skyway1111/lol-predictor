"""Unit tests for the ingestion facts transform's grouping, orientation, and outcome rules."""

from __future__ import annotations

from league_predictor.ingestion.facts_transform import (
    RawMap,
    game_rows,
    group_series,
    infer_series_format,
    reconstruct_draft_actions,
    series_row,
)

# Team ids chosen so the string sort is obvious: A < B < C < D.
A, B, C, D = "oe:A", "oe:B", "oe:C", "oe:D"


def _map(game_id, day, t1, t2, winner, blue, gn, *, hh="10") -> RawMap:
    return RawMap(
        game_id=game_id,
        game_no=gn,
        date=f"{day} {hh}:00:00",
        team1_id=t1,
        team2_id=t2,
        winner=winner,
        blue_team=blue,
        league_id="LCK",
    )


# --- best-of inference (the one home of the rule) -----------------------------


def test_infer_lone_map_is_bo1():
    assert infer_series_format([1], [1]) == "Bo1"


def test_infer_two_map_run_is_bo3():
    assert infer_series_format([1, 2], [1, 0]) == "Bo3"


def test_infer_three_wins_or_gn4_is_bo5():
    assert infer_series_format([1, 2, 3], [1, 1, 1]) == "Bo5"  # 3 wins
    assert infer_series_format([3, 4], [1, 0]) == "Bo5"  # a map numbered >= 4


# --- grouping, orientation, outcome ------------------------------------------


def test_canonical_orientation_and_completed_winner():
    # Bo3 A vs B, A wins 2-0. Sorted pair is (A, B); winner derived from map majority.
    grouped = group_series(
        [
            _map("m1", "2026-01-10", A, B, A, A, 1, hh="09"),
            _map("m2", "2026-01-10", B, A, A, B, 2, hh="10"),
        ]
    )
    assert len(grouped.groups) == 1
    g = grouped.groups[0]
    assert (g.team_a, g.team_b) == (A, B)
    assert g.winner == A
    assert g.status == "completed"
    assert g.best_of == 3
    games = game_rows(g)
    assert (games[0].blue_team, games[0].red_team) == (A, B)
    assert (games[1].blue_team, games[1].red_team) == (B, A)  # blue is a per-game fact


def test_bo1_double_header_is_two_series():
    # Two game_number=1 maps of one pair on one day -> two 1-map series (reset splits).
    grouped = group_series(
        [
            _map("n1", "2026-01-05", C, D, C, C, 1, hh="09"),
            _map("n2", "2026-01-05", D, C, D, D, 1, hh="12"),
        ]
    )
    assert len(grouped.groups) == 2
    assert {g.winner for g in grouped.groups} == {C, D}
    assert all(g.best_of == 1 and g.status == "completed" for g in grouped.groups)
    # distinct minted ids (seq-within-day), both mapping their game
    assert len({g.series_id for g in grouped.groups}) == 2


def test_two_bo3s_same_pair_day_split_at_reset():
    grouped = group_series(
        [
            _map("p1", "2026-01-06", C, D, C, C, 1, hh="09"),
            _map("p2", "2026-01-06", D, C, C, D, 2, hh="10"),
            _map("p3", "2026-01-06", C, D, D, C, 1, hh="12"),
            _map("p4", "2026-01-06", D, C, D, D, 2, hh="13"),
        ]
    )
    assert len(grouped.groups) == 2
    assert all(len(g.maps) == 2 for g in grouped.groups)


def test_remake_gap_stays_one_series():
    # game_numbers 1 then 3 is one increasing run -> ONE series (a remade middle map).
    grouped = group_series(
        [
            _map("q1", "2026-01-07", A, C, A, A, 1, hh="09"),
            _map("q2", "2026-01-07", C, A, A, C, 3, hh="11"),
        ]
    )
    assert len(grouped.groups) == 1
    assert [m.game_no for m in grouped.groups[0].maps] == [1, 3]
    assert grouped.groups[0].winner == A


def test_incomplete_run_has_null_winner():
    # A two-map 1-1 run infers Bo3; neither side reaches 2 wins -> incomplete, NULL winner.
    grouped = group_series(
        [
            _map("s1", "2026-01-08", B, D, B, B, 1, hh="10"),
            _map("s2", "2026-01-08", D, B, D, D, 2, hh="11"),
        ]
    )
    g = grouped.groups[0]
    assert g.status == "incomplete"
    assert g.winner is None
    assert g.best_of == 3


def test_degenerate_self_pair_dropped():
    grouped = group_series(
        [
            _map("x1", "2026-01-08", A, A, A, A, 1),  # self-pair -- uncanonicalizable
            _map("x2", "2026-01-08", A, B, A, A, 1),
        ]
    )
    assert grouped.degenerate == 1
    assert len(grouped.groups) == 1


# --- best_of inference end-to-end through grouping ----------------------------


def test_lone_map_infers_bo1_completed():
    grouped = group_series([_map("f1", "2026-01-09", A, B, A, A, 1)])
    g = grouped.groups[0]
    assert g.best_of == 1
    assert g.status == "completed"
    assert g.winner == A


def test_grouping_is_deterministic():
    maps = [
        _map("m1", "2026-01-10", A, B, A, A, 1, hh="09"),
        _map("m2", "2026-01-10", B, A, A, B, 2, hh="10"),
    ]
    a = group_series(maps)
    b = group_series(maps)
    assert [series_row(g) for g in a.groups] == [series_row(g) for g in b.groups]


# --- reconstruct_draft_actions over the shared vocab.draft_order core ------------------


_ROLES = ("top", "jng", "mid", "bot", "sup")
_BLUE_PICKS = ["BC1", "BC2", "BC3", "BC4", "BC5"]
_RED_PICKS = ["RC1", "RC2", "RC3", "RC4", "RC5"]
_FULL_ROLES = dict(zip(_BLUE_PICKS, _ROLES, strict=True)) | dict(
    zip(_RED_PICKS, _ROLES, strict=True)
)


def _transform_picks(champ_to_role, *, first_pick_is_blue, blue_bans=None, red_bans=None):
    """The pick subsequence of the transform's reconstruction, in draft order."""
    actions = reconstruct_draft_actions(
        "g",
        blue_picks=_BLUE_PICKS,
        red_picks=_RED_PICKS,
        blue_bans=blue_bans or [],
        red_bans=red_bans or [],
        first_pick_is_blue=first_pick_is_blue,
        champ_to_role=champ_to_role,
    )
    return [(a.champion_id, a.side, a.role) for a in actions if a.kind == "pick"]


def test_pick_reconstruction_matches_draft_order_both_fp_sides():
    # Full role map (every champion resolved) on both first-pick sides: the transform's
    # pick rows must equal the shared vocab.draft_order sequence with roles attached.
    from league_predictor.vocab.draft_order import ordered_picks

    for first_pick_is_blue in (True, False):
        expected = [
            (champ, side, _FULL_ROLES[champ])
            for champ, side, _ in ordered_picks(
                _BLUE_PICKS, _RED_PICKS, first_pick_is_blue=first_pick_is_blue
            )
        ]
        assert _transform_picks(_FULL_ROLES, first_pick_is_blue=first_pick_is_blue) == expected


def test_roleless_fallback_keeps_order_all_roles_none():
    # One unmapped champion: order/sides are kept but every pick's role is None
    # (facts_transform's all-or-nothing role contract).
    partial = dict(_FULL_ROLES)
    del partial["RC3"]
    for first_pick_is_blue in (True, False):
        got = _transform_picks(partial, first_pick_is_blue=first_pick_is_blue)
        full = _transform_picks(_FULL_ROLES, first_pick_is_blue=first_pick_is_blue)
        assert [(c, s) for c, s, _ in got] == [(c, s) for c, s, _ in full]
        assert all(role is None for _, _, role in got)


def test_ban_holes_do_not_shift_the_picks():
    # Missing ban slots leave seq holes but must never shift the pick subsequence.
    with_holes = _transform_picks(
        _FULL_ROLES,
        first_pick_is_blue=True,
        blue_bans=["X1", "X2", "X3"],  # phase-2 blue bans empty
        red_bans=["Y1", "Y2", "Y3", "Y4", "Y5"],
    )
    assert with_holes == _transform_picks(_FULL_ROLES, first_pick_is_blue=True)
