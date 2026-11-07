"""The fixed-recipe LightGBM seed bag; members combine in logit space, uncalibrated."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd

from league_predictor.probability import logit_array, sigmoid_array

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lightgbm import Booster


def to_frame(
    X: Sequence[Mapping[str, float | str]],
    *,
    feature_names: Sequence[str],
    categorical_names: Sequence[str],
) -> pd.DataFrame:
    """Typed DataFrame with columns in ``feature_names`` order, categoricals as ``category``.

    Args:
        X: dict rows or a ``dataset.ColumnarRows`` matrix view; both yield the same frame.

    Raises:
        KeyError: a declared feature is absent from every row.
        ValueError: a categorical is declared over ``ColumnarRows`` (a float64 matrix).
    """
    from league_predictor.predraft.pipeline.dataset import ColumnarRows

    cols = list(feature_names)
    cat = set(categorical_names)
    if isinstance(X, ColumnarRows):
        # The columnar fast path wraps the corpus matrix directly. A float64 matrix cannot
        # carry categorical labels, so a declared categorical here is a wiring bug.
        if cat:
            raise ValueError(
                f"ColumnarRows is a float64 matrix and cannot supply categorical columns "
                f"{sorted(cat)}; categorical features require dict rows"
            )
        df = pd.DataFrame(X.matrix, columns=X.names)
    else:
        # Build natively through pandas' C record reader. An empty ``X`` cannot infer its
        # columns, so seed them explicitly and keep the schema consistent for ``fit``.
        df = pd.DataFrame(list(X)) if len(X) else pd.DataFrame(columns=cols)
    # Correctness guard: a declared feature absent from the rows is a builder bug, not a
    # silent NaN-fill. Checked over column labels, once per frame rather than per row.
    for name in cols:
        if name not in df.columns:
            raise KeyError(name)
    df = df[cols]  # feature_names order; drops any undeclared extras
    # one consolidated astype: declared categoricals -> ``category``, the rest -> float. Per
    # column it would re-consolidate the BlockManager; under Copy-on-Write this stays lazy.
    return cast(
        "pd.DataFrame", df.astype({name: "category" if name in cat else float for name in cols})
    )


@dataclass(frozen=True)
class LightGBMParams:
    """Small-data-safe LightGBM knobs; the defaults are LightGBM's own no-op values."""

    n_estimators: int = 300
    learning_rate: float = 0.03
    num_leaves: int = 15
    max_depth: int = -1
    min_child_samples: int = 30
    subsample: float = 0.8
    subsample_freq: int = 0
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    random_state: int = 42


_DEFAULT_PARAMS = LightGBMParams()


class _FittedLightGBM:
    """A fitted LightGBM behind the ``FittedModel`` interface; returns P(team_a wins)."""

    def __init__(self, model) -> None:
        self._model = model

    def predict_proba_frame(self, frame):
        """Predict from a prebuilt typed frame; returns the positive-class column as an array."""
        pos = list(self._model.classes_).index(1)
        return self._model.predict_proba(frame)[:, pos]

    @property
    def booster(self):
        """The native LightGBM ``Booster`` -- the served artifact's per-member save unit."""
        return self._model.booster_


class _BoosterModel:
    """A bare reloaded ``Booster`` behind the ``FittedModel`` predict path; predict only."""

    def __init__(self, booster) -> None:
        self._booster = booster

    def predict_proba_frame(self, frame):
        """Predict P(team_a wins) from a prebuilt typed frame (raw binary-objective predict)."""
        import numpy as np

        return np.asarray(self._booster.predict(frame), dtype=float)

    @property
    def booster(self):
        """The native LightGBM ``Booster`` -- the served artifact's per-member save unit."""
        return self._booster


class _FittedBag:
    """The fitted per-seed members; predict = sigmoid(mean of per-seed logits)."""

    def __init__(
        self,
        members: Sequence,
        *,
        feature_names: Sequence[str],
        categorical_names: Sequence[str],
    ) -> None:
        self._members = list(members)
        self._feature_names = list(feature_names)
        self._categorical_names = list(categorical_names)

    def predict_proba(self, X: Sequence[Mapping[str, float | str]]) -> list[float]:
        frame = to_frame(
            X, feature_names=self._feature_names, categorical_names=self._categorical_names
        )
        per_member = [m.predict_proba_frame(frame) for m in self._members]
        # Vectorised clip + logit-mean over an (n_members, n_rows) stack; the 1e-15 clip stops
        # one saturated member from dragging the mean by ~34.5/N.
        avg_logit = logit_array(np.vstack(per_member), eps=1e-15).mean(axis=0)
        return [float(p) for p in sigmoid_array(avg_logit)]

    @property
    def member_boosters(self) -> list:
        """Each member's native ``Booster`` in member order -- the artifact's save units."""
        return [m.booster for m in self._members]


class LightGBMBagBackend:
    """Fits one LightGBM per seed and serves their logit average."""

    name = "lightgbm_bag"

    def __init__(
        self,
        params: LightGBMParams = _DEFAULT_PARAMS,
        *,
        seeds: Sequence[int] = (42,),
    ) -> None:
        self.params = params
        self.seeds = tuple(seeds)

    # sightline-ok: 14 - dense LightGBM rows and sparse Impact designs share names, not a data type
    def fit(
        self,
        X: Sequence[Mapping[str, float | str]],
        y: Sequence[int],
        *,
        sample_weight: Sequence[float] | None,
        feature_names: Sequence[str],
        categorical_names: Sequence[str],
    ) -> _FittedBag:
        frame = to_frame(X, feature_names=feature_names, categorical_names=categorical_names)
        y_list = list(y)
        weights = None if sample_weight is None else list(sample_weight)

        def _fit_seed(s: int):
            from lightgbm import LGBMClassifier

            model = LGBMClassifier(
                **asdict(replace(self.params, random_state=s)), n_jobs=1, verbose=-1
            )
            model.fit(
                frame, y_list, sample_weight=weights, categorical_feature=list(categorical_names)
            )
            return _FittedLightGBM(model)

        # LightGBM releases the GIL during native training and each member fits with
        # ``n_jobs=1``, so concurrent seeds give byte-identical members in ~1/N the wall time.
        with ThreadPoolExecutor(max_workers=len(self.seeds)) as pool:
            members = list(pool.map(_fit_seed, self.seeds))
        return _FittedBag(members, feature_names=feature_names, categorical_names=categorical_names)


def bag_from_boosters(
    boosters: Sequence[Booster], *, feature_names: Sequence[str], categorical_names: Sequence[str]
) -> _FittedBag:
    """Rebuild a ``_FittedBag`` from native ``Booster``s -- the served-artifact reload path."""
    members = [_BoosterModel(b) for b in boosters]
    return _FittedBag(members, feature_names=feature_names, categorical_names=categorical_names)
