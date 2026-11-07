"""Model artifact files: one family per token under ``models/<token>/``, content-addressed ids."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor import config

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


def artifact_model_id(kind: str, artifact_sha: str) -> str:
    """'<kind>:<sha12>', the content-addressed id of the rating, draft, and impact models."""
    return f"{kind}:{artifact_sha[:12]}"


def served_model_id(version: str, content_sha8: str) -> str:
    """'served:<version>_<sha8>'; the sha makes the bytes, not the version stamp, the identity."""
    return f"served:{version}_{content_sha8}"


def family_sha(paths: Iterable[Path]) -> str:
    """sha256 over each file's name and bytes, in name order. Minters slice ``[:12]``."""
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: p.name):
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            while chunk := handle.read(1 << 20):
                digest.update(chunk)
    return digest.hexdigest()


def _read_meta(meta_path: Path) -> dict:
    with meta_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _torch_payload(primary: Path) -> dict:
    import torch

    return torch.load(primary, weights_only=True)


def _single_file_family(primary: Path) -> list[Path]:
    return [primary]


def _served_family(primary: Path) -> list[Path]:
    """The ``.json`` meta plus one ``.member{i}.lgb`` per bag member."""
    meta = _read_meta(primary)
    if meta.get("format") != 3:
        raise RuntimeError(f"served meta {primary} is not format 3; retrain served")
    root = primary.with_suffix("")
    n = int(meta["bag"]["members"])
    return [primary, *(root.with_name(f"{root.name}.member{i}.lgb") for i in range(n))]


def _draft_family(primary: Path) -> list[Path]:
    """The metadata, encoder weights, and frozen knowledge arrays."""
    meta = _read_meta(primary)
    if meta.get("format") != 4:
        raise RuntimeError(f"draft meta {primary} is not format 4; retrain draft")
    root = primary.with_suffix("")
    return [
        primary,
        root.with_name(f"{root.name}.encoder.pt"),
        root.with_name(f"{root.name}.knowledge.npz"),
    ]


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """One token's files: the primary suffix, its family, and how its id is spelled."""

    primary_suffix: str
    enumerate_family: Callable[[Path], list[Path]]
    id_version_key: str | None = None  # meta key stamped into the id (served only)
    read_payload: Callable[[Path], dict] = _read_meta


SPECS: dict[str, ArtifactSpec] = {
    "served": ArtifactSpec(".json", _served_family, id_version_key="version"),
    "rating": ArtifactSpec(".pt", _single_file_family, read_payload=_torch_payload),
    "draft": ArtifactSpec(".json", _draft_family),
    "impact": ArtifactSpec(".json", _single_file_family),
}
TOKENS = tuple(SPECS)


def validate_token(token: str) -> None:
    if token not in SPECS:
        raise KeyError(f"unknown model {token!r}; expected one of: {', '.join(TOKENS)}")


def token_dir(token: str) -> Path:
    """``<MODELS_DIR>/<token>``, read from ``config`` at call time."""
    return Path(config.MODELS_DIR) / token


def artifact_root(token: str) -> Path:
    """The suffix-less stem ``<MODELS_DIR>/<token>/<token>`` every family file extends."""
    validate_token(token)
    return token_dir(token) / token


def primary_path(token: str) -> Path:
    """The family's meta file, or its single artifact."""
    validate_token(token)
    return token_dir(token) / f"{token}{SPECS[token].primary_suffix}"


def family_paths(token: str, primary: Path | None = None) -> list[Path]:
    """Every file of the family at ``primary`` (default: the live artifact)."""
    validate_token(token)
    return SPECS[token].enumerate_family(primary or primary_path(token))


def mint(token: str, primary: Path | None = None) -> tuple[str, str]:
    """The ``(model_id, full sha)`` the family's bytes mint."""
    primary = primary or primary_path(token)
    spec = SPECS[token]
    sha = family_sha(spec.enumerate_family(primary))
    if spec.id_version_key is None:
        return artifact_model_id(token, sha), sha
    return served_model_id(_read_meta(primary)[spec.id_version_key], sha[:8]), sha


def content_sha(token: str) -> str:
    return mint(token)[1]


def persisted_recipe(token: str) -> dict | None:
    """The live artifact's persisted recipe, or None when no artifact exists.

    Raises:
        RuntimeError: an artifact exists but persists no ``recipe``.
    """
    primary = primary_path(token)
    if not primary.is_file():
        return None
    recipe = SPECS[token].read_payload(primary).get("recipe")
    if recipe is None:
        raise RuntimeError(f"{token} artifact {primary} persists no 'recipe'; retrain {token}")
    return recipe
