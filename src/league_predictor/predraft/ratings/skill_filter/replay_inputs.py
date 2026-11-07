"""What one replay day consumes: the epoch-invariant tensors built from DayBatch + NormStats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from league_predictor.predraft.ratings.skill_filter.corpus_schema import MARGIN_IDX

if TYPE_CHECKING:
    import torch

    from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus, NormStats


def _to_slots(t):
    """Broadcast a per-team channel tensor [...,2,C] over the 5 role slots -> [...,2,5,C]."""
    return t.unsqueeze(2).expand(-1, -1, 5, -1)


def to_slots_k(t: torch.Tensor) -> torch.Tensor:
    """The K-batched twin of ``_to_slots``: a member tensor [K,G,2,C] -> [K,G,2,5,C]."""
    return t.unsqueeze(3).expand(-1, -1, -1, 5, -1)


@dataclass(frozen=True, slots=True)
class _ReplayDayStatics:
    """One day's replay inputs that depend only on its DayBatch and NormStats.

    Attributes:
        idx/unknown/flat_known/known_idx/orgs_flat/ctx_n: read back verbatim every game
            (clamped player rows, the unknown/known masks, flattened orgs, normalized context).
        outcome: the observed winner for each side.
        x_player_static/x_team_static: the update inputs with every channel EXCEPT the two
            per-epoch ones (surprise, opponent strength) already concatenated.
        team_perf: ``[G,2,P]`` credit split's org-credit input.
        margin_target/margin_mask: ``[G]`` auxiliary margin label + presence mask.
    """

    idx: torch.Tensor
    unknown: torch.Tensor
    flat_known: torch.Tensor
    known_idx: torch.Tensor
    orgs_flat: torch.Tensor
    ctx_n: torch.Tensor
    outcome: torch.Tensor
    x_player_static: torch.Tensor
    x_team_static: torch.Tensor
    team_perf: torch.Tensor
    margin_target: torch.Tensor
    margin_mask: torch.Tensor


def precompute_replay_statics(
    corpus: Corpus, *, norm: NormStats | None = None
) -> list[_ReplayDayStatics]:
    """Every day's epoch-invariant replay inputs, built ONCE per fit (no learned parameter)."""
    import torch

    norm = norm if norm is not None else corpus.norm
    return [_day_statics(torch, db, norm) for db in corpus.days]


def _day_statics(torch, db, norm: NormStats) -> _ReplayDayStatics:
    """Build one day's epoch-invariant replay inputs (see ``_ReplayDayStatics``).

    Returns:
        The statics, normalized with ``norm`` (training: the corpus's own stats; serving: the
            ARTIFACT's). This is the builder — ``precompute_replay_statics`` hoists it out
            of the epoch loop and ``_replay`` falls back to it per day, so both paths are
            byte-identical. Reads only ``db`` and ``norm``.
    """
    pids = db.players  # [G,2,5]
    unknown = (pids < 0).unsqueeze(-1)  # [G,2,5,1]
    idx = pids.clamp(min=0)
    flat_known = (~unknown.squeeze(-1)).reshape(-1)
    known_idx = idx.reshape(-1)[flat_known]
    orgs_flat = db.orgs.reshape(-1)
    ctx_n = (db.ctx - norm.ctx_mean) / norm.ctx_std
    perf_mask = db.perf_mask
    perf_n = ((db.perf - norm.player_mean) / norm.player_std) * perf_mask
    team_n = ((db.team_feat - norm.team_mean) / norm.team_std) * db.team_mask
    outcome = torch.stack([db.a_won, 1.0 - db.a_won], dim=1)  # [G,2]

    # Auxiliary margin label: masked mean of the normalized side-0 margin channels;
    # 0 (mask 0) where every channel is missing, so the loss simply skips it.
    t0 = (db.team_feat[:, 0, MARGIN_IDX] - norm.team_mean[MARGIN_IDX]) / norm.team_std[
        MARGIN_IDX
    ]  # [G, n_margin]
    mk0 = db.team_mask[:, 0, MARGIN_IDX]  # [G, n_margin]
    present = mk0.sum(dim=-1)  # [G]
    margin_target = (t0 * mk0).sum(dim=-1) / present.clamp(min=1.0)
    margin_mask = (present > 0).float()

    residual, team_perf = _credit_split(perf_n, perf_mask)
    out1 = outcome.unsqueeze(-1)
    x_team_static = torch.cat([team_n, db.team_mask, ctx_n, out1], dim=-1)
    x_player_static = torch.cat(
        [
            residual,
            perf_mask,
            _to_slots(team_n),
            _to_slots(db.team_mask),
            _to_slots(ctx_n),
            _to_slots(out1),
        ],
        dim=-1,
    )
    return _ReplayDayStatics(
        idx=idx,
        unknown=unknown,
        flat_known=flat_known,
        known_idx=known_idx,
        orgs_flat=orgs_flat,
        ctx_n=ctx_n,
        outcome=outcome,
        x_player_static=x_player_static,
        x_team_static=x_team_static,
        team_perf=team_perf,
        margin_target=margin_target,
        margin_mask=margin_mask,
    )


def _credit_split(perf_n, perf_mask):
    """Split the normalized box score into (within-team residual, mask-aware team mean)."""
    present = perf_mask.sum(dim=-2, keepdim=True).clamp(min=1.0)  # [...,1,P] present-slot count
    team_mean = perf_n.sum(dim=-2, keepdim=True) / present
    residual = (perf_n - team_mean) * perf_mask  # [...,5,P] re-masked: an absent entry stays 0
    return residual, team_mean.squeeze(-2)  # mean -> [...,P] org credit input
