"""The frozen per-family champion-knowledge member ``.knowledge.npz``."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from league_predictor.draft_grading.champion_attributes.ability_store import (
    AbilityFeatureStore,
    load_ability_vectors,
)
from league_predictor.draft_grading.champion_attributes.kit_vector import (
    KIT_MEMBER_FLAG,
    kit_matrix,
    kit_matrix_shape,
)
from league_predictor.draft_grading.grader.power_curve_features import (
    POWER_MEMBER_FLAG,
    PowerTables,
    power_table_shape,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from league_predictor.draft_grading.grader.embeddings import ChampionIndex

KNOWLEDGE_MEMBER_SUFFIX = ".knowledge.npz"
KNOWLEDGE_SCHEMA_VERSION = 1

#: The backfill remedy every missing-member error names.
_BACKFILL_TOOL = "lol-predict retrain draft"


def knowledge_member_path(stem: Path) -> Path:
    """``{stem}.knowledge.npz`` — name-APPEND (dotted side stems keep their name)."""
    stem = Path(stem)
    return stem.with_name(f"{stem.name}{KNOWLEDGE_MEMBER_SUFFIX}")


@dataclass(frozen=True)
class KnowledgeTables:
    """One family's frozen serving view of champion knowledge; UNKNOWN is the last row."""

    schema_version: int
    feature_names: tuple[str, ...]  # column order of z_matrix
    champions: tuple[str, ...]  # index order, sans UNKNOWN
    z_matrix: np.ndarray  # (len+1, D) float32, per-feature z-scored
    power: PowerTables | None  # (len+1, 5, 4, 11) float32 curves + their borrowed plane
    kit: np.ndarray | None  # (len+1, n_kit_families) float32, the dense named kit vector

    def z_vectors(self) -> dict[str, np.ndarray]:
        """Champion names mapped to their frozen standardized feature rows."""
        return {name: self.z_matrix[i] for i, name in enumerate(self.champions)}


def build_knowledge_tables(
    index: ChampionIndex,
    ability_store: AbilityFeatureStore,
    *,
    kit: bool = False,
    power: PowerTables | None = None,
) -> KnowledgeTables:
    """Freeze champion vectors, the optional kit matrix, and role power curves at fit time."""
    zvecs, _, _ = load_ability_vectors(ability_store)
    dim = len(ability_store.feature_names)
    z_matrix = np.zeros((len(index), dim), dtype=np.float32)
    for i, name in enumerate(index.known_champions):
        resolved = ability_store.resolve_name(name)
        if resolved is not None and resolved in zvecs:
            z_matrix[i] = zvecs[resolved]

    return KnowledgeTables(
        schema_version=KNOWLEDGE_SCHEMA_VERSION,
        feature_names=tuple(ability_store.feature_names),
        champions=tuple(index.known_champions),
        z_matrix=z_matrix,
        power=power,
        kit=kit_matrix(index, ability_store) if kit else None,
    )


def member_flags(tables: KnowledgeTables) -> dict[str, bool]:
    """The meta flags a family stamps for these tables; ``_MemberCheck`` reads the same map."""
    return {
        "lane_pair_features": False,
        "fight_pair_features": False,
        POWER_MEMBER_FLAG: tables.power is not None,
        KIT_MEMBER_FLAG: tables.kit is not None,
        "attribute_embeddings": False,
    }


def save_knowledge_tables(stem: Path, tables: KnowledgeTables) -> str:
    """Write ``{stem}.knowledge.npz``; return its sha256 hex for the meta stamp."""
    path = knowledge_member_path(stem)
    arrays: dict[str, np.ndarray] = {
        "schema_version": np.asarray(tables.schema_version, dtype=np.int64),
        "feature_names": np.array(tables.feature_names, dtype=np.str_),
        "champions": np.array(tables.champions, dtype=np.str_),
        "z_matrix": tables.z_matrix,
    }
    if tables.power is not None:
        arrays["power_table"] = tables.power.table
        arrays["power_borrowed"] = tables.power.borrowed
    if tables.kit is not None:
        arrays["kit_matrix"] = tables.kit
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        np.savez(handle, **arrays)  # pyright: ignore[reportArgumentType]  # stub folds **kwds into allow_pickle
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class _MemberCheck:
    """A loaded knowledge member, the family meta it must agree with, and how to object."""

    payload: Mapping
    meta: Mapping
    corrupt: Callable[[str], Exception]

    def core(self) -> tuple[int, tuple[str, ...], tuple[str, ...], np.ndarray]:
        """``(schema_version, champions, feature_names, z_matrix)``, each checked vs the meta."""
        payload, meta, corrupt = self.payload, self.meta, self.corrupt
        version = int(payload.get("schema_version", -1))
        if version != KNOWLEDGE_SCHEMA_VERSION:
            raise corrupt(f"schema_version ({version} != {KNOWLEDGE_SCHEMA_VERSION})")

        champions = tuple(str(c) for c in payload["champions"])
        if list(champions) != list(meta.get("champions", [])):
            raise corrupt("champion universe (member champions != meta['champions'])")

        feature_names = tuple(str(f) for f in payload["feature_names"])
        if len(set(feature_names)) != len(feature_names):
            raise corrupt("feature_names uniqueness")

        z_matrix = payload["z_matrix"]
        n_rows = len(champions) + 1
        if z_matrix.shape != (n_rows, len(feature_names)) or z_matrix.dtype != np.float32:
            raise corrupt(
                f"z_matrix shape/dtype ({z_matrix.shape}, {z_matrix.dtype}; expected "
                f"({n_rows}, {len(feature_names)}) float32)"
            )
        if not np.isfinite(z_matrix).all():
            raise corrupt("z_matrix finite")
        if np.any(z_matrix[-1] != 0.0):
            raise corrupt("UNKNOWN row (must be exact zeros — the corpus mean under z-scoring)")
        if len(feature_names) != int(meta.get("knowledge_dim", -1)):
            raise corrupt(
                f"knowledge_dim ({len(feature_names)} != meta {meta.get('knowledge_dim')!r})"
            )
        return version, champions, feature_names, z_matrix

    def power(self, n_rows: int) -> PowerTables | None:
        """The frozen power curves and their borrowed plane, present iff the meta's flag is."""
        corrupt = self.corrupt
        table, borrowed = self.payload.get("power_table"), self.payload.get("power_borrowed")
        if (table is None) != (borrowed is None):
            raise corrupt("power_table pairing (a table and its borrowed plane ride together)")
        if bool(self.meta.get(POWER_MEMBER_FLAG, False)) != (table is not None):
            raise corrupt(f"power_table presence (must match meta {POWER_MEMBER_FLAG})")
        if table is None or borrowed is None:
            return None
        expected = power_table_shape(n_rows)
        if table.shape != expected or table.dtype != np.float32:
            raise corrupt(
                f"power_table shape/dtype ({table.shape}, {table.dtype}; expected "
                f"{expected} float32)"
            )
        if not np.isfinite(table).all():
            raise corrupt("power_table finite")
        if np.any(table[-1] != 0.0):
            raise corrupt("power_table UNKNOWN row (must be exact zeros)")
        if borrowed.shape != expected[:2] or borrowed.dtype != np.bool_:
            raise corrupt(
                f"power_borrowed shape/dtype ({borrowed.shape}, {borrowed.dtype}; expected "
                f"{expected[:2]} bool)"
            )
        if borrowed[-1].any():
            raise corrupt("power_borrowed UNKNOWN row (must be all False)")
        return PowerTables(table, borrowed)

    def kit(self, n_rows: int) -> np.ndarray | None:
        """The frozen kit vectors, present exactly when the meta's kit flag is."""
        corrupt = self.corrupt
        kit = self.payload.get("kit_matrix")
        if bool(self.meta.get(KIT_MEMBER_FLAG, False)) != (kit is not None):
            raise corrupt(f"kit_matrix presence (must match meta {KIT_MEMBER_FLAG})")
        if kit is not None:
            expected = kit_matrix_shape(n_rows)
            if kit.shape != expected or kit.dtype != np.float32:
                raise corrupt(
                    f"kit_matrix shape/dtype ({kit.shape}, {kit.dtype}; expected "
                    f"{expected} float32)"
                )
            if not np.isfinite(kit).all():
                raise corrupt("kit_matrix finite")
            if np.any(kit[-1] != 0.0):
                raise corrupt("kit_matrix UNKNOWN row (must be exact zeros)")
        return kit


def load_knowledge_tables(stem: Path, meta: Mapping) -> KnowledgeTables:
    """Load and verify ``{stem}.knowledge.npz`` against the family meta.

    Raises:
        RuntimeError: the stem carries no member.
        RuntimeError: any integrity check fails; names the check and the stem.
    """
    stem = Path(stem)
    path = knowledge_member_path(stem)

    def corrupt(check: str) -> RuntimeError:
        return RuntimeError(
            f"knowledge member {path.name} fails its {check} check — the family at "
            f"{stem} must not serve. Refit the slot, or re-run {_BACKFILL_TOOL} "
            "for a pre-migration family."
        )

    if not path.is_file():
        raise RuntimeError(
            f"family stem {stem} has no {KNOWLEDGE_MEMBER_SUFFIX} member — a "
            f"missing family must be rebuilt via {_BACKFILL_TOOL} "
            "(new fits write it at save)."
        )
    stamped_sha = meta.get("knowledge_sha")
    if not stamped_sha:
        raise corrupt("knowledge_sha stamp (member present but meta unstamped)")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != stamped_sha:
        raise corrupt("sha256 (bytes differ from the meta's knowledge_sha stamp)")

    try:
        with np.load(path, allow_pickle=False) as npz:
            payload = {key: npz[key] for key in npz.files}
    except Exception as exc:
        raise corrupt(f"npz parse ({exc})") from exc

    member = _MemberCheck(payload, meta, corrupt)
    version, champions, feature_names, z_matrix = member.core()
    n_rows = len(champions) + 1

    return KnowledgeTables(
        schema_version=version,
        feature_names=feature_names,
        champions=champions,
        z_matrix=z_matrix,
        power=member.power(n_rows),
        kit=member.kit(n_rows),
    )
