"""SHA-verified persistence for skill-filter weights plus everything serving needs."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor.predraft.ratings.skill_filter.corpus_schema import NormStats
from league_predictor.predraft.ratings.skill_filter.model import (
    FilterConfig,
    SkillFilter,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

RATING_DEPENDENCY_STAGE_VERSION = "rating-member-order-v1"

_STRUCTURAL_CONFIG_KEYS = frozenset({"d", "d_org", "hidden", "seed"})
_DEPLOYED_NORM_KEYS = frozenset(
    {"player_mean", "player_std", "team_mean", "team_std", "ctx_mean", "ctx_std"}
)
_LEGACY_FORMULA = {
    "elo_mode": False,
    "elo_k": 0.1,
    "use_margins": True,
    "perf_encoder": False,
    "variance_readout": "off",
    "org_factor": "off",
    "dispersion_readout": "raw",
    "familiarity_readout": "off",
    "familiarity_pool": "mean",
    "staleness_readout": "off",
    "drop_channels": (),
    "international_head": False,
    "update_kfactor": "settled",
    "credit_split": True,
    "box_pace": False,
    "box_role_relative": False,
    "query_ctx": False,
}


def _deployed_config(raw: dict) -> FilterConfig:
    """Translate the verbose frozen config into the fixed runtime formula."""
    extra = set(raw) - _STRUCTURAL_CONFIG_KEYS
    if extra:
        legacy = {key: raw.get(key) for key in _LEGACY_FORMULA}
        if extra != set(_LEGACY_FORMULA) or legacy != _LEGACY_FORMULA:
            raise ValueError("unsupported Rating artifact formula; retrain rating")
    missing = _STRUCTURAL_CONFIG_KEYS - set(raw)
    if missing:
        raise ValueError(f"Rating artifact config is missing structural fields: {sorted(missing)}")
    return FilterConfig(**{key: raw[key] for key in _STRUCTURAL_CONFIG_KEYS})


def _deployed_norm(raw: dict) -> NormStats:
    """Load deployed normalization while discarding retired experiment variants."""
    missing = _DEPLOYED_NORM_KEYS - set(raw)
    if missing:
        raise ValueError(f"Rating artifact norm is missing deployed fields: {sorted(missing)}")
    return NormStats(**{key: raw[key] for key in _DEPLOYED_NORM_KEYS})


@dataclass(frozen=True)
class RatingDependency:
    """The exact ordered rating bag a downstream served artifact consumed."""

    artifact_sha: str
    member_count: int
    ordered_member_digest: str
    stage_version: str = RATING_DEPENDENCY_STAGE_VERSION

    def to_meta(self) -> dict[str, str | int]:
        """Return the JSON-safe persisted representation."""
        return dataclasses.asdict(self)

    @classmethod
    def from_meta(cls, meta: Mapping) -> RatingDependency:
        """Load a dependency descriptor without weakening its stage contract."""
        return cls(**meta)

    @property
    def identity(self) -> str:
        """Content identity of the full downstream dependency contract."""
        payload = json.dumps(self.to_meta(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


def require_rating_dependency(actual: RatingDependency, expected: RatingDependency) -> None:
    """Reject any rating SHA, K, member-order, or stage mismatch."""
    if actual != expected:
        raise RuntimeError(
            "served rating dependency mismatch: "
            f"expected {expected.to_meta()}, got {actual.to_meta()}; retrain served"
        )


def _ordered_member_digest(payload: Mapping) -> tuple[int, str]:
    """Hash member order, semantic configs, and every ordered state tensor."""
    states = payload.get("members")
    configs = payload.get("member_configs")
    if states is None:
        states = [payload["state_dict"]]
        configs = [payload["config"]]
    elif configs is None:
        configs = [payload["config"]] * len(states)
    digest = hashlib.sha256()
    digest.update(f"{RATING_DEPENDENCY_STAGE_VERSION}\n".encode())
    for position, (config, state) in enumerate(zip(configs, states, strict=True)):
        digest.update(f"member:{position}\n".encode())
        digest.update(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())
        digest.update(b"\n")
        for name in sorted(state):
            tensor = state[name].detach().cpu().contiguous()
            digest.update(str(name).encode())
            digest.update(b"\0")
            digest.update(str(tensor.dtype).encode())
            digest.update(b"\0")
            digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
            digest.update(b"\0")
            digest.update(tensor.numpy().tobytes())
            digest.update(b"\n")
    return len(states), digest.hexdigest()


def rating_dependency(path: Path, *, expected_sha: str | None = None) -> RatingDependency:
    """Read the exact rating artifact dependency used by downstream serving."""
    from league_predictor import artifacts

    payload, _meta = _verified_payload(Path(path), expected_sha)
    count, ordered_digest = _ordered_member_digest(payload)
    return RatingDependency(
        artifact_sha=artifacts.family_sha([Path(path)]),
        member_count=count,
        ordered_member_digest=ordered_digest,
    )


# sightline-ok: 56 - writes the research-fit single-model shape scoring's slot fallback serves


def save_bag_artifact(
    path: Path,
    models: Sequence[SkillFilter],
    config: FilterConfig,
    norm: NormStats,
    league_index: dict[str, int],
    *,
    signature: tuple[int, str, str],
    notes: dict[str, object],
    recipe: dict | None = None,
) -> str:
    """Write a fixed-formula K-member bag and return its SHA.

    Args:
        recipe: the recipe dict describing THIS fit (production passes
            ``RATING_RECIPE.to_dict()``); ``None`` stamps none (research fits).
    """
    import torch

    payload = {
        "recipe": recipe,
        "config": dataclasses.asdict(config),
        "n_leagues": models[0].league_emb.num_embeddings,
        "members": [m.state_dict() for m in models],
        "norm": dataclasses.asdict(norm),
        "league_index": dict(league_index),
        "signature": tuple(signature),
        "notes": dict(notes),
        "torch_version": str(torch.__version__),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return artifact_sha(path)


def load_bag_artifact(
    path: Path, *, expected_sha: str | None = None
) -> tuple[list[SkillFilter], dict]:
    """Rebuild a fixed-formula K-member bag and verify its SHA when given."""
    payload, meta = _verified_payload(path, expected_sha)
    member_cfgs = payload.get("member_configs")
    if member_cfgs is None:
        member_cfgs = [payload["config"]] * len(payload["members"])
    models = []
    for cfg_dict, state_dict in zip(member_cfgs, payload["members"], strict=True):
        model = SkillFilter(_deployed_config(cfg_dict), n_leagues=payload["n_leagues"])
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
    return models, meta


def _verified_payload(path: Path, expected_sha: str | None) -> tuple[dict, dict]:
    """SHA-check, load, and build the shared ``meta`` dict; a mismatch raises ``ValueError``."""
    import torch

    sha = artifact_sha(path)
    if expected_sha is not None and sha != expected_sha:
        raise ValueError(
            f"artifact SHA mismatch for {path}: expected {expected_sha}, "
            f"got {sha} — refusing to load unpinned weights"
        )
    payload = torch.load(path, weights_only=True)
    meta = {
        "norm": _deployed_norm(payload["norm"]),
        "league_index": payload["league_index"],
        "signature": tuple(payload["signature"]),
        "notes": payload["notes"],
        "sha": sha,
        "recipe": payload.get("recipe"),
    }
    return payload, meta


def artifact_sha(path: Path) -> str:
    """SHA-256 of the artifact file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
