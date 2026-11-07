"""Parsing and pagination of the PandaScore upcoming-matches feed."""

from __future__ import annotations

from league_predictor.upcoming.pandascore import PandaMatch, PandaTeam, parse_matches


def _raw_match(**overrides) -> dict:
    """One PandaScore match dict with the fields parse_matches reads."""
    raw = {
        "id": 1397580,
        "league": {"id": 4996, "name": "EMEA Masters"},
        "scheduled_at": "2026-03-14T16:00:00Z",
        "number_of_games": 3,
        "status": "not_started",
        "opponents": [
            {
                "type": "Team",
                "opponent": {"id": 126215, "name": "WLGaming Esports", "acronym": "WLG"},
            },
            {
                "type": "Team",
                "opponent": {
                    "id": 126832,
                    "name": "Unicorns Of Love Sexy Edition",
                    "acronym": "USE",
                },
            },
        ],
    }
    raw.update(overrides)
    return raw


def test_parse_matches_reads_the_match_fields():
    matches = parse_matches([_raw_match()])
    assert matches == [
        PandaMatch(
            ps_match_id=1397580,
            league_name="EMEA Masters",
            scheduled_ts="2026-03-14T16:00:00+00:00",
            best_of=3,
            status="not_started",
            teams=(
                PandaTeam(ps_team_id=126215, name="WLGaming Esports"),
                PandaTeam(ps_team_id=126832, name="Unicorns Of Love Sexy Edition"),
            ),
        )
    ]


def test_parse_matches_captures_the_tournament_stage():
    """The tournament name is the stage signal is_playoffs prices; absent reads None."""
    raw = _raw_match(tournament={"id": 9, "name": "Playoffs"})
    (match,) = parse_matches([raw])
    assert match.tournament_name == "Playoffs"

    (bare,) = parse_matches([_raw_match()])
    assert bare.tournament_name is None


def test_parse_matches_keeps_partial_opponents_for_tbd_matches():
    """A bracket slot with one known team parses with a single-team tuple."""
    raw = _raw_match(opponents=[_raw_match()["opponents"][0]])
    (match,) = parse_matches([raw])
    assert len(match.teams) == 1
    assert match.teams[0].ps_team_id == 126215


def test_parse_matches_skips_player_opponents():
    """Showmatch player opponents never become teams."""
    raw = _raw_match(
        opponents=[{"type": "Player", "opponent": {"id": 9, "name": "Faker", "acronym": None}}]
    )
    (match,) = parse_matches([raw])
    assert match.teams == ()


def test_parse_matches_skips_matches_without_a_schedule():
    raw = _raw_match(scheduled_at=None)
    assert parse_matches([raw]) == []


def test_parse_matches_counts_each_teams_finished_maps():
    """A running match's ``games`` give the score: finished maps per winner, nothing else."""
    raw = _raw_match(
        status="running",
        games=[
            {"position": 1, "status": "finished", "winner": {"id": 126832, "type": "Team"}},
            {"position": 2, "status": "finished", "winner": {"id": 126832, "type": "Team"}},
            {"position": 3, "status": "running", "winner": {"id": None, "type": None}},
            {"position": 4, "status": "not_started", "winner": {"id": None, "type": None}},
        ],
    )
    (match,) = parse_matches([raw])
    assert match.wins == ((126832, 2),)
    assert parse_matches([_raw_match()])[0].wins == ()
