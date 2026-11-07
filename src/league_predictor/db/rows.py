"""Frozen typed rows for the multi-writer tables, each 1:1 with its table in ``db.schema``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelRow:
    """A content-addressed ``models`` registry row."""

    model_id: str
    kind: str
    artifact_sha: str | None = None


@dataclass(frozen=True, slots=True)
class PredictionRow:
    """One append-only ``predictions`` row; ``created_at`` is part of the natural key."""

    subject_kind: str
    subject_id: str
    model_id: str
    p_a: float
    asof_date: str
    context: str
    created_at: str


@dataclass(frozen=True, slots=True)
class TeamStateRow:
    """One team's latent strength on an appearance-day under a model."""

    team_id: str
    model_id: str
    asof_date: str
    strength: float | None = None


@dataclass(frozen=True, slots=True)
class SeriesRow:
    """A ``series`` fact row; orientation must be canonical (``team_a < team_b``)."""

    series_id: str
    team_a: str
    team_b: str
    league_id: str | None = None
    scheduled_ts: str | None = None
    best_of: int | None = None
    winner: str | None = None
    status: str | None = None


@dataclass(frozen=True, slots=True)
class GameRow:
    """A ``games`` fact row (one map); ``event_type`` is 'regular' or 'playoffs'."""

    game_id: str
    series_id: str
    game_no: int
    blue_team: str | None = None
    red_team: str | None = None
    winner: str | None = None
    patch: str | None = None
    duration_s: int | None = None
    event_type: str | None = None


@dataclass(frozen=True, slots=True)
class GamePlayerRow:
    """A ``game_players`` box-score row at (game_id, player_id) grain; stats default None."""

    game_id: str
    player_id: str
    team_id: str
    role: str | None = None
    champion_id: str | None = None
    result: int | None = None
    kills: int | None = None
    deaths: int | None = None
    assists: int | None = None
    teamkills: int | None = None
    teamdeaths: int | None = None
    damageshare: float | None = None
    killparticipation: float | None = None
    deathshare: float | None = None
    vspm: float | None = None
    controlwardsbought: int | None = None
    golddiffat15: float | None = None
    csdiffat15: float | None = None
    xpdiffat15: float | None = None
    golddiffat10: float | None = None
    csdiffat10: float | None = None
    xpdiffat10: float | None = None
    goldat15: float | None = None
    csat15: float | None = None
    xpat15: float | None = None
    dpm: float | None = None
    cspm: float | None = None
    earnedgoldshare: float | None = None
    firstblood: int | None = None
    firsttower: int | None = None
    firstdragon: int | None = None
    firstbaron: int | None = None
    wpm: float | None = None
    wcpm: float | None = None
    wardsplaced: int | None = None
    wardskilled: int | None = None
    damagetakenperminute: float | None = None
    damagemitigatedperminute: float | None = None
    firstbloodkill: int | None = None
    firstbloodassist: int | None = None
    killsat15: int | None = None
    assistsat15: int | None = None
    deathsat15: int | None = None


@dataclass(frozen=True, slots=True)
class DraftActionRow:
    """One reconstructed pick/ban action in a game's draft sequence."""

    game_id: str
    seq: int
    side: str
    kind: str
    champion_id: str | None = None
    role: str | None = None


@dataclass(frozen=True, slots=True)
class GameTeamStatRow:
    """A ``game_team_stats`` fact row at (game_id, team_id) grain; stats default None."""

    game_id: str
    team_id: str
    dragons: int | None = None
    barons: int | None = None
    towers: int | None = None
    heralds: int | None = None
    firstdragon: int | None = None
    firstbaron: int | None = None
    firsttower: int | None = None
    firstherald: int | None = None
    totalgold: int | None = None
    totalcs: int | None = None
    teamkills: int | None = None
    teamdeaths: int | None = None
    gamelength: int | None = None
    void_grubs: int | None = None
