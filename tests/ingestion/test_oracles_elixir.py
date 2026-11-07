"""Tests for Oracle's Elixir source handling."""

import math

import pandas as pd

from league_predictor.ingestion.oracles_elixir import (
    _first_pick_team,
    _parse_game,
    _team_draft_fields,
    _winner_team_id,
)


class TestWinnerNaNSafe:
    """Winner needs an explicit ``result == 1`` on exactly one side; otherwise it is NULL."""

    def test_normal_case_picks_winning_side(self):
        blue = {"teamid": "oe:team:t1", "result": 1}
        red = {"teamid": "oe:team:geng", "result": 0}
        assert _winner_team_id(blue, red) == "oe:team:t1"
        # Symmetric: red wins.
        blue2 = {"teamid": "oe:team:t1", "result": 0}
        red2 = {"teamid": "oe:team:geng", "result": 1}
        assert _winner_team_id(blue2, red2) == "oe:team:geng"

    def test_nan_result_yields_none_not_fabricated_red(self):
        blue = {"teamid": "oe:team:t1", "result": math.nan}
        red = {"teamid": "oe:team:geng", "result": math.nan}
        assert _winner_team_id(blue, red) is None

    def test_none_result_yields_none(self):
        blue = {"teamid": "oe:team:t1", "result": None}
        red = {"teamid": "oe:team:geng", "result": None}
        assert _winner_team_id(blue, red) is None

    def test_ambiguous_both_won_yields_none(self):
        blue = {"teamid": "oe:team:t1", "result": 1}
        red = {"teamid": "oe:team:geng", "result": 1}
        assert _winner_team_id(blue, red) is None


# Draft-field parse helpers: bans, pick order, and first-pick side come off OE team rows and
# are reconstructed into ``draft_actions``, which draft_grading/ reads via DraftState.


class TestTeamDraftFields:
    def test_returns_lists_for_full_bans_and_picks(self):
        team = {
            "ban1": "Aatrox",
            "ban2": "Ahri",
            "ban3": "Akali",
            "ban4": "Alistar",
            "ban5": "Amumu",
            "pick1": "Jax",
            "pick2": "Lee Sin",
            "pick3": "Azir",
            "pick4": "Kai'Sa",
            "pick5": "Leona",
        }
        bans, picks = _team_draft_fields(team)
        assert bans == ["Aatrox", "Ahri", "Akali", "Alistar", "Amumu"]
        assert picks == ["Jax", "Lee Sin", "Azir", "Kai'Sa", "Leona"]

    def test_preserves_pick_order_with_none_holes(self):
        team = {
            "ban1": "Aatrox",
            "ban2": None,
            "ban3": "Akali",
            "ban4": math.nan,
            "ban5": "Amumu",
            "pick1": "Jax",
        }
        bans, picks = _team_draft_fields(team)
        # None holes preserved positionally so consumers know ban4 was missing.
        assert bans == ["Aatrox", None, "Akali", None, "Amumu"]
        # picks only has pick1 set; pick2..pick5 absent from dict -> None.
        assert picks == ["Jax", None, None, None, None]

    def test_returns_none_when_columns_absent(self):
        # Older OE seasons have no ban*/pick* columns at all.
        team = {"some_other": "value"}
        bans, picks = _team_draft_fields(team)
        assert bans is None
        assert picks is None

    def test_returns_none_when_all_values_nan(self):
        team = {f"ban{i}": math.nan for i in range(1, 6)}
        bans, _ = _team_draft_fields(team)
        assert bans is None


class TestFirstPickTeam:
    def test_blue_first(self):
        blue = {"firstPick": 1, "teamid": "oe:team:blue"}
        red = {"firstPick": 0, "teamid": "oe:team:red"}
        assert _first_pick_team(blue, red) == "oe:team:blue"

    def test_red_first(self):
        blue = {"firstPick": 0, "teamid": "oe:team:blue"}
        red = {"firstPick": 1, "teamid": "oe:team:red"}
        assert _first_pick_team(blue, red) == "oe:team:red"

    def test_both_missing_returns_none(self):
        blue = {"firstPick": math.nan, "teamid": "oe:team:blue"}
        red = {"firstPick": math.nan, "teamid": "oe:team:red"}
        assert _first_pick_team(blue, red) is None

    def test_both_claim_returns_none(self):
        # Data error: both sides marked firstPick=1. Don't guess.
        blue = {"firstPick": 1, "teamid": "oe:team:blue"}
        red = {"firstPick": 1, "teamid": "oe:team:red"}
        assert _first_pick_team(blue, red) is None


# ``_parse_game`` keeps ``first_pick_is_blue`` three-state (bool | None): collapsing an unknown
# to False would fabricate a red-first pick order into the ``draft_actions`` facts.


_MIN_ROLES = ("top", "jng", "mid", "bot", "sup")


def _min_game_df(blue_fp, red_fp) -> pd.DataFrame:
    """A minimal one-game OE frame: two team rows and ten player rows, ``_parse_game`` columns."""
    day = "2026-01-01 10:00:00"
    blue_picks = ["C1", "C2", "C3", "C4", "C5"]
    red_picks = ["D1", "D2", "D3", "D4", "D5"]
    team = [
        {
            "position": "team",
            "side": "Blue",
            "teamid": "oe:blue",
            "date": day,
            "game": 1,
            "result": 1,
            "firstPick": blue_fp,
            **{f"pick{i}": c for i, c in enumerate(blue_picks, start=1)},
        },
        {
            "position": "team",
            "side": "Red",
            "teamid": "oe:red",
            "date": day,
            "game": 1,
            "result": 0,
            "firstPick": red_fp,
            **{f"pick{i}": c for i, c in enumerate(red_picks, start=1)},
        },
    ]
    players = [
        {
            "position": role,
            "side": side,
            "teamid": tid,
            "date": day,
            "game": 1,
            "result": res,
            "playerid": f"{tid}:{role}",
            "champion": champ,
        }
        for tid, side, res, picks in (
            ("oe:blue", "Blue", 1, blue_picks),
            ("oe:red", "Red", 0, red_picks),
        )
        for role, champ in zip(_MIN_ROLES, picks, strict=True)
    ]
    return pd.DataFrame(team + players)


class TestParseGameFirstPickThreeState:
    def test_ambiguous_first_pick_carries_none_not_false(self):
        parsed = _parse_game("g1", _min_game_df(blue_fp=math.nan, red_fp=math.nan))
        assert parsed is not None
        # Unknown must stay unknown -- a False here would fabricate a red-first draft.
        assert parsed.draft["first_pick_is_blue"] is None

    def test_blue_first_pick_carries_true(self):
        parsed = _parse_game("g1", _min_game_df(blue_fp=1, red_fp=0))
        assert parsed is not None
        assert parsed.draft["first_pick_is_blue"] is True

    def test_red_first_pick_carries_false(self):
        parsed = _parse_game("g1", _min_game_df(blue_fp=0, red_fp=1))
        assert parsed is not None
        assert parsed.draft["first_pick_is_blue"] is False


def test_nullable_fact_columns_preserve_oe_types_and_aliases():
    from league_predictor.ingestion.oracles_elixir import _game_player_rows, _game_team_stat_rows

    frame = pd.DataFrame({"teamid": ["A"], "total cs": [42.0]})
    (team,) = _game_team_stat_rows("g", frame)
    assert team.totalcs == 42
    assert isinstance(team.totalcs, int)
    assert team.dragons is None
    frame = pd.DataFrame(
        {
            "playerid": ["p"],
            "teamid": ["A"],
            "position": ["mid"],
            "champion": ["Ahri"],
            "kills": [float("nan")],
            "dpm": [123.5],
            "deaths": [3],
            "teamdeaths": [12],
        }
    )
    (player,) = _game_player_rows("g", frame)
    assert (player.player_id, player.role, player.champion_id) == ("p", "mid", "Ahri")
    assert player.kills is None
    assert player.dpm == 123.5
    assert player.deathshare == 0.25
