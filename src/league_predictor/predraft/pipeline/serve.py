"""The serving path: how a fitted ``ServedModel`` turns a matchup into P(team_a wins)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from league_predictor.asof import AsOf
from league_predictor.predraft.features import FamilyIndex, build_features, build_features_batch
from league_predictor.predraft.pipeline.model_contract import FittedModel, predict_symmetric
from league_predictor.predraft.pipeline.series_math import (
    PRE_SERIES,
    SeriesState,
    symmetric_series_prob,
)
from league_predictor.predraft.pipeline.staleness import dampen_stale_matchup
from league_predictor.predraft.pipeline.temperature import apply_temperature
from league_predictor.probability import logit, sigmoid

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date

    from league_predictor.predraft.pipeline.chosen_model import ServedModel
    from league_predictor.predraft.ratings.region_rating import RegionRatingTable

# Logit saturation for the rating blend -- how extreme a base prob's log-odds is allowed to get
# before the blend caps it (the ``region_rating._EPS`` convention).
_RATING_BLEND_EPS = 1e-6


def rating_logit(
    features_ab: Mapping[str, float | str], features_ba: Mapping[str, float | str]
) -> float:
    """The skill filter's matchup logit, symmetrized as ``0.5 * (ab - ba)``: side-agnostic."""
    return 0.5 * (float(features_ab["rating_logit_diff"]) - float(features_ba["rating_logit_diff"]))


def blend_toward_rating(p: float, z_rating: float, w: float) -> float:
    """Blend a base prob toward ``z_rating`` in logit space; ``w == 0`` returns ``p`` bit-exact."""
    if w == 0.0:
        return p
    return sigmoid((1.0 - w) * logit(p, eps=_RATING_BLEND_EPS) + w * z_rating)


def series_lead(
    features_ab: Mapping[str, float | str], features_ba: Mapping[str, float | str]
) -> float:
    """Maps team_a leads by in this series, symmetrized as ``0.5 * (ab - ba)``: side-agnostic."""
    return 0.5 * (float(features_ab["diff_series_wins"]) - float(features_ba["diff_series_wins"]))


def with_series_lead(p: float, lead: float, weight: float) -> float:
    """Shift ``p`` by ``weight * lead`` in logit space; ``weight == 0`` returns ``p`` bit-exact."""
    if weight == 0.0:
        return p
    return sigmoid(logit(p, eps=_RATING_BLEND_EPS) + weight * lead)


def blended_base(
    p: float,
    features_ab: Mapping[str, float | str],
    features_ba: Mapping[str, float | str],
    w: float,
    lead_weight: float = 0.0,
) -> float:
    """``blend_toward_rating``, then ``with_series_lead``; a zero weight skips its feature reads."""
    if w != 0.0:
        p = blend_toward_rating(p, rating_logit(features_ab, features_ba), w)
    if lead_weight != 0.0:
        p = with_series_lead(p, series_lead(features_ab, features_ba), lead_weight)
    return p


def game_prob(
    fitted: FittedModel,
    indexes: Mapping[str, FamilyIndex],
    team_a: str,
    team_b: str,
    as_of: AsOf | date,
    *,
    families: Sequence[str],
) -> float:
    """Side-agnostic per-map P(team_a wins) from a raw fitted bag — no rating blend."""
    x_ab = build_features(team_a, team_b, as_of, indexes=indexes, families=families)
    x_ba = build_features(team_b, team_a, as_of, indexes=indexes, families=families)
    return predict_symmetric(fitted, [x_ab], [x_ba])[0]


def series_prob(
    p_map: float, match_format: str | None, *, state: SeriesState = PRE_SERIES
) -> float:
    """The next map's prob aggregated to the series prob from ``state`` (maps won so far)."""
    return symmetric_series_prob(p_map, match_format, state=state)


# --------------------------------------------------------------------------- the ServedModel chain


def feature_computation(
    indexes: Mapping[str, FamilyIndex],
    team_a: str,
    team_b: str,
    as_of: AsOf | date,
    *,
    features: Sequence[str],
) -> dict[str, float | str]:
    """The chosen feature vector for one orientation: every family assembled, ``features`` kept."""
    full = build_features(team_a, team_b, as_of, indexes=indexes, families=list(indexes))
    return {name: full[name] for name in features}


def base_score(
    model: ServedModel, features_ab: dict[str, float | str], features_ba: dict[str, float | str]
) -> float:
    """Side-agnostic per-map P(team_a wins): bag scored both ways, blended, series lead added."""
    p = predict_symmetric(model.bag, [features_ab], [features_ba])[0]
    return blended_base(p, features_ab, features_ba, model.rating_blend, model.series_lead)


def base_score_batch(
    model: ServedModel,
    feats_ab: Sequence[Mapping[str, float | str]],
    feats_ba: Sequence[Mapping[str, float | str]],
) -> list[float]:
    """Batched ``base_score``, float-equivalent; both orientations in the same matchup order."""
    ps = predict_symmetric(model.bag, feats_ab, feats_ba)
    w, lead = model.rating_blend, model.series_lead
    return [
        blended_base(p, ab, ba, w, lead) for p, ab, ba in zip(ps, feats_ab, feats_ba, strict=True)
    ]


def adjust_map_prob(
    p_map: float,
    league: RegionRatingTable | None,
    indexes: Mapping[str, FamilyIndex],
    team_a: str,
    team_b: str,
    as_of: AsOf | date,
    *,
    temperature: float,
    damp_facts_games: bool,
) -> float:
    """Apply one artifact-bound post-base formula in canonical serve order.

    Args:
        damp_facts_games: the inactivity dampener also humbles a facts game; otherwise it
            reaches only a match whose lineups the rating cannot see.
    """
    query = AsOf.of(as_of)
    day = query.day  # the residual and the idle clock are day-strict readers
    if league is not None:
        p_map = league.adjust(p_map, team_a, team_b, day)
    p_map = apply_temperature(p_map, temperature)
    if query.facts_game and not damp_facts_games:
        return p_map
    return dampen_stale_matchup(p_map, indexes, team_a, team_b, as_of=day)


def predict(
    model: ServedModel,
    indexes: Mapping[str, FamilyIndex],
    team_a: str,
    team_b: str,
    *,
    as_of: AsOf | date,
) -> float:
    """The serving entry point — the per-game P(team_a) written to the ``predictions`` log.

    Args:
        as_of: the game being predicted (its position and, for a game the facts do not hold,
            its match context); a bare date is the start of that day.
    """
    as_of = AsOf.of(as_of)
    feats = model.feature_names
    ab = feature_computation(indexes, team_a, team_b, as_of, features=feats)
    ba = feature_computation(indexes, team_b, team_a, as_of, features=feats)
    base = base_score(model, ab, ba)
    return adjust_map_prob(
        base,
        model.league,
        indexes,
        team_a,
        team_b,
        as_of,
        temperature=model.temperature,
        damp_facts_games=model.damp_facts_games,
    )


def predict_batch(
    model: ServedModel,
    indexes: Mapping[str, FamilyIndex],
    matchups: Sequence[tuple[str, str, AsOf | date]],
) -> list[float]:
    """Batched twin of ``predict``, float-equivalent; one P(team_a) per matchup in input order."""
    if not matchups:
        return []  # the bag's native predict rejects an empty frame; a no-op batch is []
    families = list(indexes)
    n = len(matchups)
    queries = [(a, b, AsOf.of(d)) for a, b, d in matchups]
    queries += [(b, a, d) for a, b, d in queries]
    feats = build_features_batch(queries, indexes=indexes, families=families)
    bases = base_score_batch(model, feats[:n], feats[n:])
    league = model.league
    return [
        adjust_map_prob(
            base,
            league,
            indexes,
            a,
            b,
            d,
            temperature=model.temperature,
            damp_facts_games=model.damp_facts_games,
        )
        for base, (a, b, d) in zip(bases, queries[:n], strict=True)
    ]
