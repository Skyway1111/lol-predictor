"""Locked Served model configuration, artifact, and fitter."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor.predraft.features.manifest import (
    SCHEMA_VERSION,
    feature_schema_fingerprint,
)
from league_predictor.predraft.pipeline.chosen_features import SERVED_FEATURES
from league_predictor.predraft.pipeline.lightgbm_bag import (
    LightGBMBagBackend,
    LightGBMParams,
    bag_from_boosters,
)
from league_predictor.predraft.pipeline.walk_forward import (
    WindowParams,
    build_fit_context,
    fit_window,
)
from league_predictor.predraft.ratings.region_rating import (
    RegionRatingTable,
    fit_region_ratings,
)
from league_predictor.predraft.ratings.skill_filter.artifact import (
    RatingDependency,
    require_rating_dependency,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Iterable, Mapping
    from datetime import date

    from league_predictor.predraft.pipeline.walk_forward import FitContext

_FAMILIES = ("learned_rating", "original", "recency", "lineup")
_SEEDS = (42, 123, 7, 999, 314, 271, 828)

# Optuna TPE tune (50 trials) minimizing pooled base log-loss under leak-safe walk-forward.
# num_leaves=65 is inert: max_depth=5 caps a tree at 2**5 = 32 leaves.
_LIGHTGBM_PARAMS = {
    "n_estimators": 522,
    "num_leaves": 65,
    "max_depth": 5,
    "learning_rate": 0.007665452256291985,
    "min_child_samples": 98,
    "subsample": 0.5452333789491521,
    "subsample_freq": 1,
    "colsample_bytree": 0.3460092100949576,
    "reg_alpha": 6.462042682582747,
    "reg_lambda": 2.8916844284867547,
}

# Every refresh trains on the full eligible history before its cutoff.
_WINDOW = WindowParams(tau=1e9, floor=0.0, cap_days=None)


@dataclass(frozen=True, slots=True)
class _ServedRecipe:
    """The only Served training and scoring recipe."""

    version: int = 5  # model meaning; moves on ANY behavior change
    families: tuple[str, ...] = _FAMILIES
    features: tuple[str, ...] = tuple(SERVED_FEATURES)
    seeds: tuple[int, ...] = _SEEDS
    window: WindowParams = _WINDOW
    # Learned region-rating residual coefficients (benchmarks.md);
    # fit on the served-OOS walk strictly before its evaluation window, 0.0 = off.
    region_beta_cross_tier: float = 0.4563
    region_beta_within_tier1: float = 0.0
    # rating_blend and series_lead: fit undamped on served maps before the evaluation window.
    rating_blend: float = 0.36
    # Logit per map of series lead: served under-reacts to the maps already played.
    series_lead: float = 0.16
    temperature: float = 1.0381057133080291
    # The inactivity dampener humbles a match whose lineups the rating cannot see. A facts
    # game's rating reads its fielded fives, and damping it costs log-loss (served-2026-10).
    damp_facts_games: bool = False

    lgb_params: dict = field(default_factory=lambda: dict(_LIGHTGBM_PARAMS))

    def to_dict(self) -> dict:
        """The exact recipe dict persisted under the artifact's ``recipe`` key."""
        return {
            "version": int(self.version),
            "config_schema_version": CONFIG_SCHEMA_VERSION,
            "families": list(self.families),
            "features": list(self.features),
            "lgb_params": dict(self.lgb_params),
            "seeds": list(self.seeds),
            "window": {
                "tau": self.window.tau,
                "floor": self.window.floor,
                "cap_days": self.window.cap_days,
            },
            "region_beta_cross_tier": float(self.region_beta_cross_tier),
            "region_beta_within_tier1": float(self.region_beta_within_tier1),
            "rating_blend": float(self.rating_blend),
            "series_lead": float(self.series_lead),
            "temperature": float(self.temperature),
            "damp_facts_games": bool(self.damp_facts_games),
        }


_RECIPE = _ServedRecipe()


def chosen_model() -> _ServedRecipe:
    """Return the one Served recipe: 124 features, seven seeds, and full history."""
    return _RECIPE


#: The serialization-schema version of the persisted recipe dict (a DIFFERENT axis from
#: ``version``, the model meaning). Bump whenever ``to_dict`` changes shape.
CONFIG_SCHEMA_VERSION = 5

SERVED_TEMPERATURE = _RECIPE.temperature


def _artifact_recipe_from_meta(meta: Mapping, meta_path: Path) -> _ServedRecipe:
    """Decode a family's persisted ``recipe`` formula."""
    config = meta.get("recipe")
    if config is None:
        raise RuntimeError(f"served artifact {meta_path} has no recipe; retrain served")
    schema = config.get("config_schema_version")
    if schema not in (3, 4, CONFIG_SCHEMA_VERSION):
        raise RuntimeError(f"served recipe schema {schema!r} is unsupported; retrain served")
    version = int(config["version"])
    temperature = float(config["temperature"])
    seed_meta = config["seeds"]
    if isinstance(seed_meta, dict):
        if seed_meta.get("per_seed"):
            raise RuntimeError("served per-seed recipe overrides are unsupported; retrain served")
        seed_meta = seed_meta["seeds"]
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise RuntimeError(
            "served artifact temperature must be finite and positive; retrain served"
        )
    params = dict(config["lgb_params"])
    LightGBMParams(**params)  # validate the persisted booster contract now, before a heavy fit
    return _ServedRecipe(
        version=version,
        families=tuple(config["families"]),
        features=tuple(config["features"]),
        lgb_params=params,
        seeds=tuple(int(seed) for seed in seed_meta),
        window=WindowParams(**config["window"]),
        region_beta_cross_tier=float(config["region_beta_cross_tier"]),
        region_beta_within_tier1=float(config["region_beta_within_tier1"]),
        rating_blend=float(config["rating_blend"]),
        temperature=temperature,
        # Schema 5 adds both; every earlier artifact had no series lead and damped facts games.
        series_lead=float(config["series_lead"]) if schema == CONFIG_SCHEMA_VERSION else 0.0,
        damp_facts_games=bool(config["damp_facts_games"])
        if schema == CONFIG_SCHEMA_VERSION
        else True,
    )


def _bag_backend_for_recipe(recipe) -> LightGBMBagBackend:
    """Construct the backend encoded by one persisted lifecycle recipe."""
    return LightGBMBagBackend(
        LightGBMParams(**recipe.lgb_params),
        seeds=recipe.seeds,
    )


def bag_backend() -> LightGBMBagBackend:
    """Construct the fixed seven-member backend."""
    return _bag_backend_for_recipe(chosen_model())


def served_version_stamp(through: date, feature_names: Iterable[str]) -> str:
    """Served version stamp ``served_<through>_<sha8 of feature set>``, shared by fit and eval."""
    sig = hashlib.sha1(repr(sorted(feature_names)).encode(), usedforsecurity=False).hexdigest()[:8]
    return f"served_{through:%Y%m%d}_{sig}"


#: On-disk family layout marker (``served_{SLOT}.json`` meta with a ``recipe`` key +
#: one ``.member{i}.lgb`` per bag member). ``load`` REQUIRES it; any other ``format`` raises.
_SERVED_FORMAT = 3


def _family_stem(path: Path | str) -> Path:
    """Return the suffix-less root of a served artifact family."""
    return Path(path).with_suffix("")


def _member_path(stem: Path, i: int) -> Path:
    """The i-th bag member's native-LightGBM file: ``<stem>.member{i}.lgb``."""
    return stem.with_name(f"{stem.name}.member{i}.lgb")


def _read_artifact_meta(
    path: Path | str, *, require_schema: bool = True
) -> tuple[Path, Path, dict]:
    """Read and validate the common Served family metadata envelope.

    Args:
        require_schema: demand the family was fit under the running feature schema. The bag
            load requires it (its columns must mean what they meant); a recipe read for a
            walk-forward refit does not, since the refit builds every column afresh.
    """
    stem = _family_stem(path)
    meta_path = stem.with_suffix(".json")
    if not meta_path.is_file():
        raise FileNotFoundError(f"no served family meta at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("format") != _SERVED_FORMAT:
        raise RuntimeError(
            f"served artifact {meta_path} declares format {meta.get('format')!r}; only format "
            f"{_SERVED_FORMAT} loads; retrain served"
        )
    schema_v = meta.get("feature_schema_version")
    schema_fp = meta.get("feature_schema_fingerprint")
    if require_schema and (schema_v != SCHEMA_VERSION or schema_fp != feature_schema_fingerprint()):
        raise RuntimeError(
            f"served artifact {meta_path} was fit under feature schema "
            f"(v{schema_v}, {schema_fp}) but running code is "
            f"(v{SCHEMA_VERSION}, {feature_schema_fingerprint()}); retrain served"
        )
    return stem, meta_path, meta


def _artifact_rating_dependency(
    meta_path: Path,
    meta: Mapping,
    *,
    actual: RatingDependency | None,
) -> RatingDependency | None:
    """Validate and return the rating dependency encoded by one artifact family."""
    dependency_meta = meta.get("rating_dependency")
    dependency = None if dependency_meta is None else RatingDependency.from_meta(dependency_meta)
    if "learned_rating" not in meta["recipe"]["families"]:
        return dependency
    if actual is None:
        raise RuntimeError(
            f"learned-rating served artifact {meta_path} requires the current rating_dependency"
        )
    if dependency is None:
        raise RuntimeError(
            f"learned-rating served artifact {meta_path} has no rating_dependency; retrain served"
        )
    require_rating_dependency(actual, dependency)
    return dependency


def _league_to_meta(league: RegionRatingTable | None) -> dict | None:
    """Serialize the residual table to a JSON-safe dict."""
    if league is None:
        return None
    return league.to_meta()


def _league_from_meta(data: dict | None) -> RegionRatingTable | None:
    """Reconstruct the persisted residual table exactly."""
    if data is None:
        return None
    return RegionRatingTable.from_meta(data)


def _recover_seeds(bag) -> list[int] | None:
    """Per-member LightGBM seeds when the bag is freshly fit, else ``None``; provenance only."""
    seeds: list[int] = []
    for member in bag._members:
        model = getattr(member, "_model", None)
        rs = getattr(model, "random_state", None)
        if rs is None:
            return None
        seeds.append(int(rs))
    return seeds


class ServedModel:
    """The fitted bag plus the formula values bound into its slot artifact; scored by ``serve``."""

    def __init__(
        self,
        bag,
        *,
        families,
        feature_names,
        league=None,
        rating_dependency: RatingDependency | None = None,
        rating_blend: float = 0.0,
        version: str = "",
        series_lead: float = 0.0,
        damp_facts_games: bool = False,
    ) -> None:
        self.bag = bag
        self.families = list(families)
        self.feature_names = list(feature_names)
        self.league = league  # RegionRatingTable | None (serve applies .adjust)
        self.temperature = SERVED_TEMPERATURE
        self.rating_dependency = rating_dependency
        self.rating_blend = (
            rating_blend  # base blend weight toward the rating logit (base_score applies it)
        )
        self.version = version  # 'served_<through>' stamp; feeds the served:<version>_<sha8> id
        self.series_lead = series_lead  # logit per map of series lead (base_score applies it)
        self.damp_facts_games = damp_facts_games  # the stale dampener's scope (adjust_map_prob)

    def save(self, path: Path | str) -> None:
        """Write the ``format: 3`` family at ``path`` (the ``served_{SLOT}.json`` meta path).

        Args:
            path: meta path; the meta carries the full fixed ``recipe`` and
                every other ``{stem}.*`` sibling is DELETED, so the on-disk glob is exactly
                what ``artifacts.family_sha`` hashes.
        """
        if "learned_rating" in self.families and self.rating_dependency is None:
            raise RuntimeError(
                "learned-rating served artifact has no rating_dependency; retrain served"
            )
        # The recipe stamp describes THIS model: its families/features/blend/temperature
        # over the fixed fit constants. A production fit follows chosen_model(), so the stamp
        # equals the source recipe exactly there.
        recipe_meta = chosen_model().to_dict()
        recipe_meta["families"] = list(self.families)
        recipe_meta["features"] = list(self.feature_names)
        recipe_meta["rating_blend"] = float(self.rating_blend)
        recipe_meta["series_lead"] = float(self.series_lead)
        recipe_meta["temperature"] = float(self.temperature)
        recipe_meta["damp_facts_games"] = bool(self.damp_facts_games)
        stem = _family_stem(path)
        stem.parent.mkdir(parents=True, exist_ok=True)
        boosters = self.bag.member_boosters
        family = [stem.with_suffix(".json")]
        for i, booster in enumerate(boosters):
            member = _member_path(stem, i)
            booster.save_model(str(member))
            family.append(member)
        bag_meta = {
            "members": len(boosters),
            # The bag's aligned-frame contract: every member scores one shared frame in this
            # column order (a pre-metadata bag is refused, so these always exist).
            "feature_names": list(self.bag._feature_names),
            "categorical_names": list(self.bag._categorical_names),
            "seeds": _recover_seeds(self.bag),
        }
        meta = {
            "format": _SERVED_FORMAT,
            "version": self.version,
            "recipe": recipe_meta,
            "rating_dependency": None
            if self.rating_dependency is None
            else self.rating_dependency.to_meta(),
            "bag": bag_meta,
            "league": _league_to_meta(self.league),
            # Load asserts both values against the running manifest, so an
            # artifact fit under a different feature schema fails loud (retrain served).
            "feature_schema_version": SCHEMA_VERSION,
            "feature_schema_fingerprint": feature_schema_fingerprint(),
        }
        stem.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        keep = {p.name for p in family}
        for stray in stem.parent.glob(f"{stem.name}.*"):
            if stray.name not in keep:
                stray.unlink()

    @classmethod
    def load(
        cls,
        path: Path | str,
        *,
        actual_rating_dependency: RatingDependency | None = None,
    ) -> ServedModel:
        """Load a ``format: 3`` served family (the meta at ``path``) -- the only layout accepted.

        Raises:
            FileNotFoundError: no meta at ``path``.
            RuntimeError: the meta's ``format`` or feature-schema stamp differs from running
                code -- an artifact fit under a different feature schema never serves.
        """
        import lightgbm as lgb

        stem, meta_path, meta = _read_artifact_meta(path)
        recipe = _artifact_recipe_from_meta(meta, meta_path)
        bag_meta = meta["bag"]
        n = int(bag_meta["members"])
        if n < 1:
            raise RuntimeError("served artifact contains no bag members; retrain served")
        boosters = [lgb.Booster(model_file=str(_member_path(stem, i))) for i in range(n)]
        bag = bag_from_boosters(
            boosters,
            feature_names=bag_meta["feature_names"],
            categorical_names=bag_meta["categorical_names"],
        )
        dependency = _artifact_rating_dependency(meta_path, meta, actual=actual_rating_dependency)
        model = cls(
            bag,
            families=list(recipe.families),
            feature_names=list(recipe.features),
            league=_league_from_meta(meta["league"]),
            rating_dependency=dependency,
            rating_blend=recipe.rating_blend,
            version=meta["version"],
            series_lead=recipe.series_lead,
            damp_facts_games=recipe.damp_facts_games,
        )
        model.temperature = recipe.temperature
        return model


def fit_served_model(
    conn: sqlite3.Connection, *, through: date, ctx: FitContext | None = None
) -> ServedModel:
    """Fit the fixed Served recipe on every eligible map before ``through``.

    Args:
        ctx: ``walk_forward.FitContext`` supplying shared indexes and the featurized corpus;
            omitting it pays a local index build.
    """
    config = chosen_model()
    if ctx is None:
        ctx = build_fit_context(conn, config, through=through, months=0)
    idx, drop, corpus = ctx.indexes, ctx.drop, ctx.corpus
    bag = fit_window(
        bag_backend(),
        train_end=through,
        window=config.window,
        drop_features=drop,
        corpus=corpus,
    )
    if bag is None:
        raise ValueError(f"empty training window through {through} for the served model")
    # The cross-region residual is built leak-safe through the training cutoff; identity on
    # same-pool matchups and on same-tier matchups below tier 1, never domestic.
    league = fit_region_ratings(conn, through=through).with_betas(
        cross_tier=config.region_beta_cross_tier,
        within_tier1=config.region_beta_within_tier1,
    )
    rating_dependency = None
    if "learned_rating" in config.families:
        rating_dependency = getattr(idx["learned_rating"], "rating_dependency", None)
        if rating_dependency is None:
            raise RuntimeError(
                "learned-rating served fit has no exact rating_dependency; retrain rating"
            )
        if not config.damp_facts_games and not getattr(idx["learned_rating"], "game_inputs", False):
            raise RuntimeError(
                f"served recipe {config.version} leaves facts games undamped, which holds only "
                "for a rating that reads each game's own five (rating recipe 2); retrain rating "
                "before retraining served"
            )
    return ServedModel(
        bag,
        families=[f.name for f in ctx.fams],
        feature_names=list(config.features),
        league=league,
        rating_dependency=rating_dependency,
        rating_blend=config.rating_blend,
        version=served_version_stamp(through, config.features),
        series_lead=config.series_lead,
        damp_facts_games=config.damp_facts_games,
    )
