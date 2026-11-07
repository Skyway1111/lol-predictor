"""The deployed latent-skill filter and its K-member functional mirror."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from league_predictor.predraft.ratings.skill_filter.corpus_schema import (
    CTX_CHANNELS,
    PLAYER_CHANNELS,
    TEAM_CHANNELS,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_N_CTX = len(CTX_CHANNELS)
# sightline-ok: 26 - an MLP input width, not a member list; a literal desyncs from the channels
X_PLAYER = 2 * len(PLAYER_CHANNELS) + 2 * len(TEAM_CHANNELS) + _N_CTX + 3
# sightline-ok: 26 - an MLP input width, not a member list; a literal desyncs from the channels
X_ORG = 2 * len(TEAM_CHANNELS) + _N_CTX + 3
H_HIST = 2
_PRECISION_BIAS = math.log(math.e - 1.0)
_UPDATE_K_INIT = math.log(math.expm1(0.1))
_UPDATE_K_SCALE_INIT = math.log(5.0)
_INITIAL_FORGETTING_TIMESCALE_DAYS = 200.0


def hist_features(roster_cont: torch.Tensor, debut: torch.Tensor) -> torch.Tensor:
    """Return log-continuity and debut count for each side."""
    return torch.stack(
        [torch.log1p(roster_cont.clamp(min=0.0)), debut.sum(dim=-1)],
        dim=-1,
    )


@dataclass(frozen=True)
class FilterConfig:
    """Structural dimensions for the one deployed Rating formula."""

    d: int = 2
    d_org: int = 4
    hidden: int = 32
    seed: int = 0


def _mlp(
    d_in: int, hidden: int, d_out: int, *, zero_out: bool, out_bias: float = 0.0
) -> nn.Sequential:
    """Build the filter's two-layer tanh MLP."""
    out = nn.Linear(hidden, d_out)
    if zero_out:
        nn.init.zeros_(out.weight)
    nn.init.constant_(out.bias, out_bias)
    return nn.Sequential(nn.Linear(d_in, hidden), nn.Tanh(), out)


def _mlp_apply(seq: nn.Sequential, x: torch.Tensor) -> torch.Tensor:
    """Apply an ``_mlp`` without per-child module dispatch in the hot replay."""
    lin0, lin2 = cast("nn.Linear", seq[0]), cast("nn.Linear", seq[2])
    return nn.functional.linear(
        torch.tanh(nn.functional.linear(x, lin0.weight, lin0.bias)), lin2.weight, lin2.bias
    )


def _bag_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None) -> torch.Tensor:
    """Apply K distinct linear layers to a tensor with a leading K axis."""
    K, out_f, in_f = w.shape
    lead = x.shape[1:-1]
    y = torch.bmm(x.reshape(K, -1, in_f), w.transpose(-1, -2))
    if b is not None:
        y = y + b[:, None, :]
    return y.reshape(K, *lead, out_f)


def _bag_mlp_apply(params: Mapping, prefix: str, x: torch.Tensor) -> torch.Tensor:
    """Apply K stacked ``_mlp`` parameter sets."""
    h = torch.tanh(_bag_linear(x, params[f"{prefix}.0.weight"], params[f"{prefix}.0.bias"]))
    return _bag_linear(h, params[f"{prefix}.2.weight"], params[f"{prefix}.2.bias"])


class _FilterMath(ABC):
    """Shared scoring and update formula for solo and stacked parameter layouts."""

    @abstractmethod
    def _mlp(self, name: str, x: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _scalar(self, name: str, ndim: int, /) -> torch.Tensor: ...

    @abstractmethod
    def _team_input(self, slots, org, ctx, *, gaps, debut) -> torch.Tensor: ...

    def team_scores(
        self,
        slots: torch.Tensor,
        org: torch.Tensor,
        ctx: torch.Tensor,
        *,
        gaps: torch.Tensor,
        debut: torch.Tensor,
    ) -> torch.Tensor:
        """Return one latent strength score per side."""
        team_input = self._team_input(slots, org, ctx, gaps=gaps, debut=debut)
        return self._mlp("f_mlp", team_input).squeeze(-1)

    @staticmethod
    def roster_dispersion(slots: torch.Tensor) -> torch.Tensor:
        """Return mean within-team latent standard deviation."""
        return slots.std(dim=-2).mean(dim=-1)

    @staticmethod
    def _swap_sym(x):
        """Pool two sides into swap-symmetric mean and absolute difference."""
        return torch.cat(
            [(x[..., 0, :] + x[..., 1, :]) * 0.5, (x[..., 0, :] - x[..., 1, :]).abs()], dim=-1
        )

    def matchup_uncertainty(self, hist: torch.Tensor) -> torch.Tensor:
        """Return learned matchup precision from settledness and dispersion."""
        pre = self._mlp("unc_mlp", self._swap_sym(hist[..., :H_HIST])).squeeze(-1)
        pre = pre + self._mlp("disp_mlp", self._swap_sym(hist[..., H_HIST : H_HIST + 1])).squeeze(
            -1
        )
        return nn.functional.softplus(pre)

    def pair_logit(self, f: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        """Return the antisymmetric side-A win logit."""
        return self.matchup_uncertainty(hist) * (f[..., 0] - f[..., 1])

    def expected_margin(self, f: torch.Tensor) -> torch.Tensor:
        """Return expected side-A dominance implied by the strength gap."""
        return torch.tanh(self._scalar("margin_beta", 1) * (f[..., 0] - f[..., 1]))

    def margin_head(
        self,
        slots: torch.Tensor,
        org: torch.Tensor,
        ctx: torch.Tensor,
        *,
        gaps: torch.Tensor,
        debut: torch.Tensor,
    ) -> torch.Tensor:
        """Return the train-only antisymmetric margin residual."""
        team_input = self._team_input(slots, org, ctx, gaps=gaps, debut=debut)
        g = self._mlp("margin_mlp", team_input).squeeze(-1)
        return g[..., 0] - g[..., 1]

    def update_step_boost(self, roster_cont: torch.Tensor) -> torch.Tensor:
        """Increase player update size for newly assembled lineups."""
        freshness = torch.exp(
            -roster_cont.clamp(min=0.0) / torch.exp(self._scalar("log_update_k_scale", 2))
        )
        return 1.0 + nn.functional.softplus(self._scalar("update_k_logit", 2)) * freshness

    def player_deltas(
        self, slots: torch.Tensor, x: torch.Tensor, kfactor: torch.Tensor
    ) -> torch.Tensor:
        """Return the settledness-scaled gated player update."""
        z = torch.cat([slots, x], dim=-1)
        gate = torch.sigmoid(self._mlp("gate_mlp", z))
        cand = torch.tanh(self._mlp("cand_mlp", z))
        return torch.exp(self._scalar("log_alpha", 4)) * gate * cand * kfactor[..., None, None]

    def org_deltas(
        self, org: torch.Tensor, x_team: torch.Tensor, team_perf: torch.Tensor
    ) -> torch.Tensor:
        """Return the gated organization update with team-level box-score credit."""
        z = torch.cat([org, x_team], dim=-1)
        gate = torch.sigmoid(self._mlp("org_gate_mlp", z))
        cand = torch.tanh(self._mlp("org_cand_mlp", z))
        cand = cand + torch.tanh(self._mlp("org_credit_mlp", team_perf))
        return torch.exp(self._scalar("log_alpha", 3)) * gate * cand


class SkillFilter(nn.Module, _FilterMath):
    """All trainable parameters for the deployed Rating recipe."""

    def __init__(self, config: FilterConfig, n_leagues: int) -> None:
        super().__init__()
        self.config = config
        torch.manual_seed(config.seed)
        d, d_org, hidden = config.d, config.d_org, config.hidden

        self.global_prior = nn.Parameter(torch.zeros(d))
        self.org_prior = nn.Parameter(torch.zeros(d_org))
        self.league_emb = nn.Embedding(max(1, n_leagues), d)
        nn.init.normal_(self.league_emb.weight, std=0.01)
        initial_log_tau = math.log(_INITIAL_FORGETTING_TIMESCALE_DAYS)
        self.log_tau = nn.Parameter(torch.full((d,), initial_log_tau))
        self.log_tau_org = nn.Parameter(torch.full((d_org,), initial_log_tau))
        self.log_alpha = nn.Parameter(torch.tensor(math.log(0.1)))

        team_in = 5 * d + d_org + _N_CTX + 2
        self.f_mlp = _mlp(team_in, hidden, 1, zero_out=False)
        self.unc_mlp = _mlp(2 * H_HIST, hidden, 1, zero_out=True, out_bias=_PRECISION_BIAS)
        self.gate_mlp = _mlp(d + X_PLAYER, hidden, d, zero_out=True, out_bias=-2.0)
        self.cand_mlp = _mlp(d + X_PLAYER, hidden, d, zero_out=True)
        self.org_gate_mlp = _mlp(d_org + X_ORG, hidden, d_org, zero_out=True, out_bias=-2.0)
        self.org_cand_mlp = _mlp(d_org + X_ORG, hidden, d_org, zero_out=True)
        self.margin_beta = nn.Parameter(torch.tensor(0.5))
        self.margin_mlp = _mlp(team_in, hidden, 1, zero_out=True)
        self.disp_mlp = _mlp(2, hidden, 1, zero_out=True)
        self.update_k_logit = nn.Parameter(torch.tensor(_UPDATE_K_INIT))
        self.log_update_k_scale = nn.Parameter(torch.tensor(_UPDATE_K_SCALE_INIT))
        self.org_credit_mlp = _mlp(len(PLAYER_CHANNELS), hidden, d_org, zero_out=True)

    def _mlp(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return _mlp_apply(getattr(self, name), x)

    def _scalar(self, name: str, _ndim: int) -> torch.Tensor:
        return getattr(self, name)

    def evolve(self, h: torch.Tensor, prior: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        """Decay player states toward their priors over elapsed days."""
        return self._evolve(h, prior, gap_days, self.log_tau)

    def evolve_org(
        self, h: torch.Tensor, prior: torch.Tensor, gap_days: torch.Tensor
    ) -> torch.Tensor:
        """Decay organization states toward their priors over elapsed days."""
        return self._evolve(h, prior, gap_days, self.log_tau_org)

    @staticmethod
    def _evolve(h, prior, gap_days, log_tau):
        """Apply the shared closed-form decay."""
        decay = torch.exp(-gap_days.unsqueeze(-1) / torch.exp(log_tau))
        evolved = prior + (h - prior) * decay
        return torch.where(gap_days.unsqueeze(-1) > 0, evolved, h)

    def _team_input(self, slots, org, ctx, *, gaps, debut):
        """Assemble the per-side representation read by strength and margin heads."""
        mean_gap = torch.log1p(gaps.clamp(min=0.0)).mean(dim=-1, keepdim=True)
        debut_frac = debut.mean(dim=-1, keepdim=True)
        return torch.cat([slots.flatten(-2), org, ctx, mean_gap, debut_frac], dim=-1)

    def matchup_hist(
        self, roster_cont: torch.Tensor, debut: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate the deployed continuity, debut, and dispersion channels."""
        return torch.cat(
            [hist_features(roster_cont, debut), self.roster_dispersion(slots).unsqueeze(-1)],
            dim=-1,
        )

    # sightline-ok: 11 - the K-member mirror of the single filter; parity pinned by the deer tests

    def prior_for(self, debut_league: torch.Tensor) -> torch.Tensor:
        """Return global plus league-of-debut player priors."""
        known = debut_league >= 0
        emb = self.league_emb(debut_league.clamp(min=0))
        emb = torch.where(known.unsqueeze(-1), emb, torch.zeros_like(emb))
        return self.global_prior.unsqueeze(0) + emb


class BatchedSkillFilter(_FilterMath):
    """K stacked deployed filters used by DEER and serving."""

    def __init__(self, params: dict, config: FilterConfig) -> None:
        self.params = params
        self.config = config
        self.K = params["global_prior"].shape[0]

    @property
    def global_prior(self) -> torch.Tensor:
        """Return stacked player priors."""
        return self.params["global_prior"]

    @property
    def org_prior(self) -> torch.Tensor:
        """Return stacked organization priors."""
        return self.params["org_prior"]

    def _mlp(self, name: str, x: torch.Tensor) -> torch.Tensor:
        return _bag_mlp_apply(self.params, name, x)

    def _scalar(self, name: str, ndim: int):
        """Broadcast a scalar parameter across trailing dimensions."""
        return self.params[name].view(self.K, *([1] * ndim))

    def _expand_k(self, tensor):
        """Add a shared leading K member axis."""
        return tensor.expand(self.K, *tensor.shape)

    # sightline-ok: 11 - one-line delegations to the shared _evolve; player and org taus differ
    def evolve(self, h: torch.Tensor, prior: torch.Tensor, gap_days: torch.Tensor) -> torch.Tensor:
        """Decay stacked player states."""
        return self._evolve(h, prior, gap_days, self.params["log_tau"])

    # sightline-ok: 11 - one-line delegations to the shared _evolve; player and org taus differ
    def evolve_org(
        self, h: torch.Tensor, prior: torch.Tensor, gap_days: torch.Tensor
    ) -> torch.Tensor:
        """Decay stacked organization states."""
        return self._evolve(h, prior, gap_days, self.params["log_tau_org"])

    def _evolve(self, h, prior, gap_days, log_tau):
        """Apply stacked closed-form decay."""
        tau = torch.exp(log_tau).view(self.K, *([1] * (h.dim() - 2)), -1)
        decay = torch.exp(-gap_days.unsqueeze(-1) / tau)
        evolved = prior + (h - prior) * decay
        return torch.where(gap_days.unsqueeze(-1) > 0, evolved, h)

    def _team_input(self, slots, org, ctx, *, gaps, debut):
        """Assemble stacked per-side representations."""
        mean_gap = torch.log1p(gaps.clamp(min=0.0)).mean(dim=-1, keepdim=True)
        debut_frac = debut.mean(dim=-1, keepdim=True)
        shared = self._expand_k(torch.cat([ctx, mean_gap, debut_frac], dim=-1))
        return torch.cat([slots.flatten(-2), org, shared], dim=-1)

    def matchup_hist(
        self, roster_cont: torch.Tensor, debut: torch.Tensor, slots: torch.Tensor
    ) -> torch.Tensor:
        """Aggregate stacked continuity, debut, and dispersion channels."""
        base = self._expand_k(hist_features(roster_cont, debut))
        return torch.cat([base, self.roster_dispersion(slots).unsqueeze(-1)], dim=-1)

    # sightline-ok: 11 - the K-member mirror of the single filter; parity pinned by the deer tests

    def prior_for(self, debut_league: torch.Tensor) -> torch.Tensor:
        """Return stacked global plus league-of-debut priors."""
        gp = self.params["global_prior"]
        emb = self.params["league_emb.weight"]
        known = debut_league >= 0
        gathered = emb[:, debut_league.clamp(min=0)]
        gathered = torch.where(known[None, :, None], gathered, torch.zeros_like(gathered))
        return (gp[:, None, :] + gathered).contiguous()
