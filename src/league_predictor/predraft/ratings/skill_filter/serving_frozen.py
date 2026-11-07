"""The flat, globally-sorted timelines a serving index retains, and the flattening pass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

# ``_KEY_STRIDE`` packs ``entity_id * _KEY_STRIDE + day.toordinal()`` into one strictly increasing
# int64 key; it must exceed any real ordinal (max 3_652_059), and 10_000_000 leaves headroom.
_KEY_STRIDE = 10_000_000


@dataclass
class _FrozenTimelines:
    """Flat, globally-sorted timelines: the only form the index retains once frozen.

    Attributes:
        player_key: each entity's snapshots occupy a contiguous band keyed by
            ``entity_id * _KEY_STRIDE + day.toordinal()`` (strictly increasing), so one
            global ``searchsorted`` lands at the strictly-before-``as_of`` boundary inside
            that band and ``index = pos - offset`` recovers the per-entity bisect result.
            The object timelines (~20x these tensors) are released once frozen.
    """

    player_key: torch.Tensor  # [T_p] int64 sorted (pid·STRIDE + day)
    player_state: torch.Tensor  # [T_p, d]
    player_offset: torch.Tensor  # [n_players+1] int64 band starts
    org_key: torch.Tensor  # [T_o] int64 sorted (oi·STRIDE + day)
    org_state: torch.Tensor  # [T_o, d_org]
    org_offset: torch.Tensor  # [n_orgs+1] int64
    team_key: torch.Tensor  # [T_r] int64 sorted (oi·STRIDE + day)
    team_five: torch.Tensor  # [T_r, 5] int64 player ids (-1 unknown)
    team_tier: torch.Tensor  # [T_r]
    team_games: torch.Tensor  # [T_r]
    team_cont: torch.Tensor  # [T_r] last-known lineup continuity
    team_offset: torch.Tensor  # [n_orgs+1] int64


@dataclass
class GameInputs:
    """Each corpus game's own pre-game inputs: a facts-game query reads what the replay trained on.

    Attributes:
        rows: ``(series_id, game_no) -> row``, the facts position an ``AsOf.game`` names.
        teams: ``(side 0, side 1)`` team ids per row, side 0 the lexicographically smaller.
        five: ``[N,2,5]`` the fielded player ids in role order, -1 unknown.
        ctx: ``[N,2,C]`` raw context channels: the game's tier, format, side, patch, churn.
        cont: ``[N,2]`` the start-of-day continuity of the fielded five.
    """

    rows: dict[tuple[str, int], int]
    teams: list[tuple[str, str]]
    five: torch.Tensor
    ctx: torch.Tensor
    cont: torch.Tensor


def game_inputs(torch, corpus) -> GameInputs:
    """The ``GameInputs`` table of every corpus game, in replay order."""
    inv_org = {i: t for t, i in corpus.vocab.org_index.items()}
    days = corpus.days
    positions = [pos for db in days for pos in db.positions]
    rows = {pos: i for i, pos in enumerate(positions)}
    assert len(rows) == len(positions), "two corpus games share one facts position"
    n_ctx = days[0].ctx.shape[-1] if days else 0
    return GameInputs(
        rows=rows,
        teams=[(inv_org[int(a)], inv_org[int(b)]) for db in days for a, b in db.orgs.tolist()],
        five=torch.cat([db.players for db in days]) if days else torch.zeros(0, 2, 5).long(),
        ctx=torch.cat([db.ctx for db in days]) if days else torch.zeros(0, 2, n_ctx),
        cont=torch.cat([db.roster_cont for db in days]) if days else torch.zeros(0, 2),
    )


def _freeze_timelines(torch, model, tl, org_index: dict[str, int]) -> _FrozenTimelines:
    """Flatten ``ServingTimelines`` into the globally-sorted tensors the batched gather reads.

    Returns:
        Per-entity bands concatenated in id order (days already ascending from the replay),
        so the ``key`` bands line up end-to-end into one sorted array; player ``prior`` rows
        stay on ``tl.prior``, indexed at gather time rather than copied.
    """
    cfg = model.config
    d, d_org = cfg.d, cfg.d_org
    n_players = tl.prior.shape[0]
    n_orgs = len(org_index)
    i64 = torch.int64

    p_keys: list[int] = []
    p_states, p_offset = [], [0]
    for pid in range(n_players):
        dates = tl.player_dates.get(pid)
        if dates:
            p_keys += [pid * _KEY_STRIDE + dt.toordinal() for dt in dates]
            p_states.append(torch.stack(tl.player_states[pid]))
        p_offset.append(len(p_keys))

    o_keys: list[int] = []
    o_states, o_offset = [], [0]
    for oi in range(n_orgs):
        dates = tl.org_dates.get(oi)
        if dates:
            o_keys += [oi * _KEY_STRIDE + dt.toordinal() for dt in dates]
            o_states.append(torch.stack(tl.org_states[oi]))
        o_offset.append(len(o_keys))

    t_keys: list[int] = []
    t_five, t_tier, t_games, t_cont, t_offset = [], [], [], [], [0]
    for oi in range(n_orgs):
        for rec in tl.team_records.get(oi, []):
            t_keys.append(oi * _KEY_STRIDE + rec.day.toordinal())
            t_five.append(rec.five)
            t_tier.append(rec.tier)
            t_games.append(rec.games)
            t_cont.append(rec.cont)
        t_offset.append(len(t_keys))

    f32 = torch.float32
    player_key = torch.tensor(p_keys, dtype=i64)
    org_key = torch.tensor(o_keys, dtype=i64)
    team_key = torch.tensor(t_keys, dtype=i64)
    # The whole gather rests on each key array being globally non-decreasing; pin it cheaply so an
    # out-of-order replay change fails loudly instead of making searchsorted silently wrong.
    for key in (player_key, org_key, team_key):
        assert key.numel() < 2 or bool((key[1:] >= key[:-1]).all()), (  # noqa: PLR2004 - a pair
            "frozen timeline keys are not sorted -- the searchsorted gather would be wrong"
        )
    return _FrozenTimelines(
        player_key=player_key,
        player_state=torch.cat(p_states) if p_states else torch.zeros(0, d),
        player_offset=torch.tensor(p_offset, dtype=i64),
        org_key=org_key,
        org_state=torch.cat(o_states) if o_states else torch.zeros(0, d_org),
        org_offset=torch.tensor(o_offset, dtype=i64),
        team_key=team_key,
        team_five=torch.tensor(t_five, dtype=i64) if t_five else torch.zeros(0, 5, dtype=i64),
        team_tier=torch.tensor(t_tier, dtype=f32),
        team_games=torch.tensor(t_games, dtype=f32),
        team_cont=torch.tensor(t_cont, dtype=f32),
        team_offset=torch.tensor(t_offset, dtype=i64),
    )
