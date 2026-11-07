"""The as-of query path: one frozen replay answered in wide batched forwards."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, TypeVar

from league_predictor.asof import AsOf
from league_predictor.predraft.ratings.skill_filter.corpus_schema import CTX_CHANNELS
from league_predictor.predraft.ratings.skill_filter.serving_frozen import (
    _KEY_STRIDE,
    GameInputs,
    _freeze_timelines,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from datetime import date

#: A query: the two teams and the game being predicted, or a bare day.
Query = tuple[str, str, "AsOf | date"]

# Day-path defaults for the context channels a query without a facts game cannot know.
_QUERY_CTX_DEFAULTS = {
    "international": 0.0,
    "game_number": 1.0,
    "side_blue": 0.0,
    "patch_changed": 0.0,
    "churn_count": 0.0,
}
_UNSEEN_TEAM_TIER = 3.0  # league_tier()'s conservative unknown default
_TREND_DAYS = 60  # strength-trend lookback window (days)

# Serving featurization runs over all query points at once: one ``searchsorted`` gather plus the
# heads as wide GEMMs over a [Q,2,...] axis, not 62k one-row forwards (~112x on the full corpus).

# Cap on query points per batched forward, so peak activation memory stays bounded on a huge
# retrain corpus (the gather/forward is independent across query points, so chunking is exact).
_BATCH_CHUNK = 16_384


def derived_feature_names(d: int, d_org: int) -> list[str]:
    """The exact column set ``SkillFilterIndex.derived_features`` emits for a (d, d_org) filter.

    Returns:
        The source of truth the skill-filter feature family and the index agree on;
        latent/org column counts scale with the state dims, so the family tracks the
        artifact's config automatically.
    """
    names = [
        "rating_strength_a",
        "rating_strength_b",
        "rating_logit_diff",
        "rating_rd",
        "rating_rd_diff",
        "rating_spread_a",
        "rating_spread_b",
        "rating_trend_a",
        "rating_trend_b",
        "rating_trend_diff",
    ]
    for c in range(d):
        names += [f"rating_latent_a_c{c}", f"rating_latent_b_c{c}", f"rating_latent_diff_c{c}"]
    names += [f"rating_org_diff_c{c}" for c in range(d_org)]
    return names


def _dedup_queries(queries: Sequence[Query]) -> tuple[list[Query], list[int]]:
    """Distinct queries in first-seen order plus the inverse index scattering back to each row.

    Returns:
        Featurization runs once per distinct ``(team_a, team_b, as_of)``; day queries of one
        matchup collapse, as do a game's repeated reads.
    """
    order: dict[Query, int] = {}
    distinct: list[Query] = []
    inverse: list[int] = []
    for q in queries:
        j = order.get(q)
        if j is None:
            j = len(distinct)
            order[q] = j
            distinct.append(q)
        inverse.append(j)
    return distinct, inverse


_T = TypeVar("_T")


# itertools.batched requires Python 3.12; this package also supports 3.11.
def _chunks(seq: Sequence[_T], n: int) -> Iterator[Sequence[_T]]:
    """Yield successive ``n``-sized slices of ``seq`` (the query-point chunking)."""
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _query_day(as_of: AsOf | date) -> date:
    return as_of.day if isinstance(as_of, AsOf) else as_of


class SkillFilterIndex:
    """As-of index over one serving replay: antisymmetric scores, unseen entities read priors.

    Args:
        games: each corpus game's own inputs. A query naming a facts game in it reads that game's
            five, context, and continuity, the inputs the replay scored it with; any other query,
            or every query when ``None``, reads the team's last record with default context.
    """

    def __init__(
        self,
        model,
        norm,
        timelines,
        org_index: dict[str, int],
        games: GameInputs | None = None,
    ) -> None:
        import torch

        self._model = model
        self._norm = norm
        self._org_index = org_index
        self._games = games
        # ONLY the flat form is retained: ``timelines`` is frozen EAGERLY and released with this
        # frame; ``prior`` is the one field the gather still reads (debut/unseen fallback rows).
        self._prior = timelines.prior
        self._frozen_tl = _freeze_timelines(torch, model, timelines, org_index)

    @property
    def game_inputs(self) -> bool:
        """Whether a facts-game query reads that game's own inputs."""
        return self._games is not None

    # sightline-ok: 11 - the one-query form of each class's OWN batch; the index protocol's shape
    def raw_score(self, team_a: str, team_b: str, as_of: AsOf | date) -> float:
        """The matchup win logit (antisymmetric, leak-safe): the one-query ``raw_score_batch``."""
        return self.raw_score_batch([(team_a, team_b, as_of)])[0]

    # sightline-ok: 11 - the one-query form of each class's OWN batch; the index protocol's shape
    def derived_features(self, team_a: str, team_b: str, as_of: AsOf | date) -> dict[str, float]:
        """The one-query ``derived_features_batch``: exactly ``derived_feature_names(d, d_org)``."""
        return self.derived_features_batch([(team_a, team_b, as_of)])[0]

    # ── featurization: bisect strictly-before-``as_of`` snapshots, evolve them analytically to
    # the query date, run the heads over a [Q,2,...] batch (one searchsorted gather) ───── #

    def raw_score_batch(self, queries: Sequence[Query]) -> list[float]:
        """Matchup win logits for many queries in one batched forward, one float per input."""
        import torch

        if not queries:
            return []
        distinct, inverse = _dedup_queries(queries)
        with torch.no_grad():
            logits = self._batched_logits(torch, distinct)  # [n_distinct]
        return logits[torch.tensor(inverse)].tolist()

    def derived_features_batch(self, queries: Sequence[Query]) -> list[dict[str, float]]:
        """The skill-filter family for many queries in wide GEMMs; one fresh dict per input."""
        import torch

        if not queries:
            return []
        distinct, inverse = _dedup_queries(queries)
        with torch.no_grad():
            rows: list[dict[str, float]] = []
            for chunk in _chunks(distinct, _BATCH_CHUNK):
                rows.extend(self._derive_chunk(torch, chunk))
        return [dict(rows[j]) for j in inverse]

    def _batched_logits(self, torch, distinct: list[Query]):
        """The matchup win logit [n_distinct] for each distinct query, chunked + concatenated."""
        parts = []
        for chunk in _chunks(distinct, _BATCH_CHUNK):
            a, b = self._pair_states(torch, chunk)
            f, hist = self._batched_pair_strength(torch, a, b)
            parts.append(self._model.pair_logit(f, hist))
        return torch.cat(parts) if parts else torch.zeros(0)

    def _game_rows(self, chunk: Sequence[Query]) -> tuple[list[int], list[int]]:
        """Each query's ``GameInputs`` row and team_a's side in it; row -1 reads the day path.

        Raises:
            ValueError: a facts game queried with teams other than the two that played it.
        """
        games = self._games
        rows, sides = [], []
        for team_a, team_b, as_of in chunk:
            row, side = -1, 0
            if games is not None and isinstance(as_of, AsOf) and as_of.facts_game:
                row = games.rows.get((as_of.series_id, as_of.game_no), -1)
                teams = games.teams[row] if row >= 0 else (team_a, team_b)
                if (team_b, team_a) == teams:
                    side = 1
                elif (team_a, team_b) != teams:
                    raise ValueError(
                        f"{as_of.key} is a game of {teams}, queried as {team_a} vs {team_b}"
                    )
            rows.append(row)
            sides.append(side)
        return rows, sides

    def _pair_states(self, torch, chunk: Sequence[Query], *, back_days: int = 0):
        """Both sides' team states for a chunk of queries, ``back_days`` before each query day."""
        days = [_query_day(q[2]) - timedelta(days=back_days) for q in chunk]
        rows, sides = self._game_rows(chunk)
        row_t, side_t = torch.tensor(rows), torch.tensor(sides)
        a = self._batched_team_state(torch, [q[0] for q in chunk], days, row_t, side_t)
        b = self._batched_team_state(torch, [q[1] for q in chunk], days, row_t, 1 - side_t)
        return a, b

    def _batched_team_state(  # noqa: PLR0915 - linear gather/evolve tensor assembly
        self, torch, tids: Sequence[str], as_ofs: Sequence[date], rows=None, sides=None
    ):
        """Per-side team state for K parallel ``(tid, as_of)`` queries, evolved to the query date.

        Args:
            rows/sides: ``[K]`` each query's ``GameInputs`` row (-1 for the day path) and this
                team's side in it; omitted, every query reads the day path.

        Returns:
            ``(slots [K,5,d], gaps [K,5], debut [K,5], org [K,d_org], ctx [K,C], cont [K],
            )``, gathered by one ``searchsorted`` per entity kind and grown over the gap by
            ``evolve``/``evolve_org``. The four prior-fallback
            branches (unknown team, unknown player, no history before ``as_of``, no org
            snapshot) are ``where`` masks, as in ``replay.py``. A game row supplies the five,
            context, and continuity; a day query reads them from the team's last record.
        """
        m = self._model
        cfg = m.config
        d, d_org = cfg.d, cfg.d_org
        fr = self._frozen_tl
        K = len(tids)
        i64 = torch.int64
        if rows is None:
            rows, sides = torch.full((K,), -1, dtype=i64), torch.zeros(K, dtype=i64)

        oi = torch.tensor([self._org_index.get(t, -1) for t in tids], dtype=i64)
        seen_org = oi >= 0
        oi_c = oi.clamp(min=0)
        aod = torch.tensor([dt.toordinal() for dt in as_ofs], dtype=i64)
        glob = m.global_prior.detach()  # [d]

        # --- team appearance record: strictly-before-as_of bisect within the org's band ---
        tpos = torch.searchsorted(fr.team_key, oi_c * _KEY_STRIDE + aod)
        ri = tpos - fr.team_offset[oi_c]
        never_seen = (~seen_org) | (ri <= 0)  # [K] pure prior team
        if fr.team_key.numel() > 0:
            ridx = (tpos - 1).clamp(min=0, max=fr.team_key.numel() - 1)
            five = fr.team_five[ridx]  # [K,5]
            tier, games = fr.team_tier[ridx], fr.team_games[ridx]
            cont = fr.team_cont[ridx]
            rec_day = fr.team_key[ridx] - oi_c * _KEY_STRIDE  # [K]
        else:
            five = torch.full((K, 5), -1, dtype=i64)
            tier = games = cont = rec_day = torch.zeros(K)
        game = rows >= 0  # [K] the query names a facts game: its own five, context, continuity
        games_table, g_row = self._games, rows.clamp(min=0)
        if games_table is not None and bool(game.any()):
            five = torch.where(game.unsqueeze(-1), games_table.five[g_row, sides], five)
        # A team with no record reads pure priors, unless its facts game names the fielded five.
        prior_team = never_seen & ~game

        # --- player slots: gather strictly-before-as_of state, prior-fallback masked, evolve ---
        pflat = five.reshape(-1)  # [5K]
        aod5 = aod.repeat_interleave(5)
        unknown_p = pflat < 0
        pidx = pflat.clamp(min=0)
        ppos = torch.searchsorted(fr.player_key, pidx * _KEY_STRIDE + aod5)
        is_prior = unknown_p | ((ppos - fr.player_offset[pidx]) <= 0)  # slot reads its prior
        prior_row = torch.where(unknown_p.unsqueeze(-1), glob, self._prior[pidx])  # [5K,d]
        if fr.player_state.shape[0] > 0:
            psnap = (ppos - 1).clamp(min=0, max=fr.player_state.shape[0] - 1)
            snap_state = fr.player_state[psnap]
            snap_day = fr.player_key[psnap] - pidx * _KEY_STRIDE
        else:
            psnap = torch.zeros(5 * K, dtype=i64)
            snap_state, snap_day = prior_row, aod5
        state = torch.where(is_prior.unsqueeze(-1), prior_row, snap_state)
        gap = (aod5 - snap_day).to(torch.float32).masked_fill(is_prior, 0.0)
        debut = is_prior.to(torch.float32)
        slots = m.evolve(state.reshape(K, 5, d), prior_row.reshape(K, 5, d), gap.reshape(K, 5))

        # --- org residual: strictly-before-as_of org snapshot, evolved, prior-fallback masked ---
        o_prior = m.org_prior.detach()  # [d_org]
        opos = torch.searchsorted(fr.org_key, oi_c * _KEY_STRIDE + aod)
        use_prior_org = never_seen | ((opos - fr.org_offset[oi_c]) <= 0)
        if fr.org_state.shape[0] > 0:
            osnap = (opos - 1).clamp(min=0, max=fr.org_state.shape[0] - 1)
            o_gap = (aod - (fr.org_key[osnap] - oi_c * _KEY_STRIDE)).to(torch.float32)
            evolved_org = m.evolve_org(fr.org_state[osnap], o_prior.expand(K, d_org), o_gap)
        else:
            evolved_org = o_prior.expand(K, d_org)
        org = torch.where(use_prior_org.unsqueeze(-1), o_prior, evolved_org)  # [K,d_org]

        # Day-path context: tier/gap/games are live; the columns only a game knows are neutral.
        gap_team = (aod - rec_day).to(torch.float32).masked_fill(never_seen, 0.0)
        tier_v = torch.where(
            never_seen, torch.full((K,), _UNSEEN_TEAM_TIER), tier.to(torch.float32)
        )
        games_v = games.to(torch.float32).masked_fill(never_seen, 0.0)
        cols = []
        for ch in CTX_CHANNELS:
            if ch == "league_tier":
                cols.append(tier_v)
            elif ch == "gap_days":
                cols.append(torch.log1p(gap_team.clamp(min=0.0)))
            elif ch == "games_played":
                cols.append(torch.log1p(games_v.clamp(min=0.0)))
            else:
                cols.append(torch.full((K,), _QUERY_CTX_DEFAULTS[ch]))
        ctx = torch.stack(cols, dim=1)  # [K,C]
        cont = cont.to(torch.float32).masked_fill(never_seen, 0.0)
        if games_table is not None and bool(game.any()):
            ctx = torch.where(game.unsqueeze(-1), games_table.ctx[g_row, sides], ctx)
            cont = torch.where(game, games_table.cont[g_row, sides], cont)

        # --- a team with no record before as_of and no facts game -> pure prior state ---
        slots = torch.where(prior_team.view(K, 1, 1), glob.view(1, 1, d).expand(K, 5, d), slots)
        gaps = gap.reshape(K, 5).masked_fill(prior_team.view(K, 1), 0.0)
        debut = torch.where(prior_team.view(K, 1), torch.ones(K, 5), debut.reshape(K, 5))
        return slots, gaps, debut, org, ctx, cont

    def _batched_team_strength(self, torch, a: tuple, b: tuple):
        """Return per-side strength and the state reused by the matchup-history head."""
        m = self._model
        ctx_n = (torch.stack([a[4], b[4]], dim=1) - self._norm.ctx_mean) / self._norm.ctx_std
        gaps = torch.stack([a[1], b[1]], dim=1)  # [K,2,5]
        debut = torch.stack([a[2], b[2]], dim=1)  # [K,2,5]
        roster_cont = torch.stack([a[5], b[5]], dim=1)  # [K,2]
        slots = torch.stack([a[0], b[0]], dim=1)  # [K,2,5,d]
        org = torch.stack([a[3], b[3]], dim=1)  # [K,2,d_org]
        f = m.team_scores(slots, org, ctx_n, gaps=gaps, debut=debut)  # [K,2]
        return f, slots, debut, roster_cont

    def _batched_pair_strength(self, torch, a: tuple, b: tuple):
        """``(f, hist)`` for K matchups, built by the SAME heads the training replay calls."""
        m = self._model
        f, slots, debut, roster_cont = self._batched_team_strength(torch, a, b)
        hist = m.matchup_hist(roster_cont, debut, slots)
        return f, hist

    def _derive_chunk(self, torch, chunk: Sequence[Query]) -> list[dict[str, float]]:
        """The rich derived family over one chunk of K matchups, unpacked to one dict per query."""
        m = self._model
        cfg = m.config
        a, b = self._pair_states(torch, chunk)
        f, hist = self._batched_pair_strength(torch, a, b)
        logit = m.pair_logit(f, hist)  # [K]
        rd = m.matchup_uncertainty(hist)  # [K]
        rd_a = m.matchup_uncertainty(torch.stack([hist[:, 0], hist[:, 0]], dim=1))
        rd_b = m.matchup_uncertainty(torch.stack([hist[:, 1], hist[:, 1]], dim=1))
        lat_a, lat_b = a[0].mean(dim=1), b[0].mean(dim=1)  # [K,d] roster-mean latents
        spread_a = a[0].std(dim=1).mean(dim=1)  # [K]
        spread_b = b[0].std(dim=1).mean(dim=1)
        org_diff = a[3] - b[3]  # [K,d_org]
        # Trend reads f 60 days back through the same f-head; a facts game keeps its five and
        # context, so its trend is that lineup's state drift.
        a_bk, b_bk = self._pair_states(torch, chunk, back_days=_TREND_DAYS)
        f_back = self._batched_team_strength(torch, a_bk, b_bk)[0]  # [K,2]
        trend = f - f_back  # [K,2]

        sa_l, sb_l = f[:, 0].tolist(), f[:, 1].tolist()
        logit_l, rd_l = logit.tolist(), rd.tolist()
        rd_diff_l = (rd_a - rd_b).tolist()
        spa_l, spb_l = spread_a.tolist(), spread_b.tolist()
        ta_l, tb_l = trend[:, 0].tolist(), trend[:, 1].tolist()
        tdiff_l = (trend[:, 0] - trend[:, 1]).tolist()
        lata_l, latb_l = lat_a.tolist(), lat_b.tolist()
        org_l = org_diff.tolist()
        out: list[dict[str, float]] = []
        for i in range(len(chunk)):
            row = {
                "rating_strength_a": sa_l[i],
                "rating_strength_b": sb_l[i],
                "rating_logit_diff": logit_l[i],
                "rating_rd": rd_l[i],
                "rating_rd_diff": rd_diff_l[i],
                "rating_spread_a": spa_l[i],
                "rating_spread_b": spb_l[i],
                "rating_trend_a": ta_l[i],
                "rating_trend_b": tb_l[i],
                "rating_trend_diff": tdiff_l[i],
            }
            for c in range(cfg.d):
                row[f"rating_latent_a_c{c}"] = lata_l[i][c]
                row[f"rating_latent_b_c{c}"] = latb_l[i][c]
                row[f"rating_latent_diff_c{c}"] = lata_l[i][c] - latb_l[i][c]
            for c in range(cfg.d_org):
                row[f"rating_org_diff_c{c}"] = org_l[i][c]
            out.append(row)
        return out
