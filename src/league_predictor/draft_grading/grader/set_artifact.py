"""The format-4 family: K seed encoders over one computed slot input, saved and loaded."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch

from league_predictor.draft_grading.grader.artifact import (
    _encoder_path,
    family_meta_path,
    resolve_device,
)
from league_predictor.draft_grading.grader.embeddings import ChampionIndex
from league_predictor.draft_grading.grader.grades import EdgeCurve
from league_predictor.draft_grading.grader.knowledge_tables import (
    knowledge_member_path,
    load_knowledge_tables,
    member_flags,
    save_knowledge_tables,
)

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Mapping, Sequence

    from league_predictor.draft_grading.grader.knowledge_tables import KnowledgeTables

#: The loader only CALLS the class it is handed, so it constrains nothing beyond that.
_BuiltT = TypeVar("_BuiltT")

#: Family layout marker: 4 = seed-varied set encoders, served by ``set_variant.py``.
SET_FORMAT = 4

#: The ``.encoder.pt`` payload a format-4 family persists: ``SetWeights.to_state()`` exactly.
SET_WEIGHT_KEYS: tuple[str, ...] = (
    "config",
    "d_in",
    "n_champions",
    "states",
    "input_spec",
    "history",
    "label_scale",
)

#: Top-meta keys the load path rebuilds the served function from.
_SET_META_KEYS = frozenset(
    {
        "champions",
        "edge_curve",
        "edge_cuts_pp",
        "grade_semantics",
        "knowledge_dim",
        "knowledge_sha",
        "pro_excess_baseline",
        "recipe",
        "temperature",
        "thresholds",
    }
)


def set_family_files(root: Path) -> list[Path]:
    """The three files a format-4 family IS: the meta, the seed weights, the knowledge member."""
    root = Path(root)
    return [family_meta_path(root), _encoder_path(root), knowledge_member_path(root)]


def checked_weights(state: Mapping) -> dict[str, Any]:
    """The weights payload, refused unless the load path can rebuild every seed from it.

    Raises:
        ValueError: a missing key, no seed state, or an ``input_spec`` that does not name
            each of the ``d_in`` slot-input columns (the design's column vocabulary).
    """
    missing = sorted(set(SET_WEIGHT_KEYS) - set(state))
    if missing:
        raise ValueError(
            f"a format-4 weights payload is SetWeights.to_state() {list(SET_WEIGHT_KEYS)}; "
            f"missing {missing}"
        )
    if not state["states"]:
        raise ValueError("a format-4 family carries at least one seed state dict")
    columns = dict(state["input_spec"]).get("columns")
    d_in = int(state["d_in"])
    if columns is None or len(columns) != d_in:
        raise ValueError(
            f"input_spec['columns'] names each of the d_in={d_in} slot-input columns, so "
            f"design() can cut the matrix into audited blocks; got {columns and len(columns)}"
        )
    return dict(state)


def _require_weights_only(path: Path) -> None:
    """Prove the written member reloads under ``weights_only`` before any meta claims it.

    Raises:
        ValueError: the payload holds something ``torch.load(weights_only=True)`` refuses
            (a numpy array is the usual one), which would otherwise surface at serve time.
    """
    try:
        torch.load(path, weights_only=True)
    except Exception as exc:
        raise ValueError(
            f"{path.name} does not reload under weights_only=True ({exc}) — a family member "
            "carries tensors and primitives only; convert it inside SetWeights.to_state"
        ) from exc


def save_set_family(
    root: Path,
    *,
    weights: Mapping,
    champions: Sequence[str],
    knowledge: KnowledgeTables,
    temperature: float,
    thresholds: Sequence[float],
    edge_curve: EdgeCurve,
    edge_cuts_pp: tuple[float, float],
    recipe: Mapping,
    pro_excess_baseline: str = "team_form",
) -> None:
    """Write the three-member family; the fit stamps provenance onto the meta afterwards.

    Args:
        weights: ``SetWeights.to_state()``; tensors and primitives only.
        recipe: the recipe the fit ran under, stamped verbatim.
    """
    root = Path(root)
    state = checked_weights(weights)
    root.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, _encoder_path(root))
    _require_weights_only(_encoder_path(root))
    meta = {
        "format": SET_FORMAT,
        "recipe": dict(recipe),
        "champions": list(champions),
        "d_in": int(state["d_in"]),
        "seeds": len(state["states"]),
        "temperature": float(temperature),
        "thresholds": [float(t) for t in thresholds],
        "grade_semantics": "edge",
        "edge_cuts_pp": [float(x) for x in edge_cuts_pp],
        "edge_curve": edge_curve.to_meta(),
        "pro_excess_baseline": pro_excess_baseline,
        "knowledge_sha": save_knowledge_tables(root, knowledge),
        "knowledge_dim": len(knowledge.feature_names),
        **member_flags(knowledge),
    }
    family_meta_path(root).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    keep = {p.name for p in set_family_files(root)}
    # The three files ARE the family, so an undeclared sibling would corrupt its id.
    for stray in root.parent.glob(f"{root.name}.*"):
        if stray.name not in keep:
            stray.unlink()


def load_set_meta(root: Path) -> dict[str, Any]:
    """A format-4 family's top meta, validated.

    Raises:
        RuntimeError: another family format, a missing key the served function needs, or
            non-edge grade semantics.
    """
    root = Path(root)
    meta = json.loads(family_meta_path(root).read_text(encoding="utf-8"))
    if meta.get("format") != SET_FORMAT:
        raise RuntimeError(
            f"draft artifact {root} declares family format {meta.get('format')!r}; this "
            f"loader takes ONLY format {SET_FORMAT} (the seed-encoder family). Load through "
            "live.load_grader, the dispatching loader."
        )
    missing = sorted(_SET_META_KEYS - meta.keys())
    if missing:
        raise RuntimeError(
            f"draft artifact {root} is missing required metadata {missing}; refit the slot"
        )
    if meta["grade_semantics"] != "edge":
        raise RuntimeError(f"draft artifact {root} must use edge grade semantics; refit the slot")
    return meta


def load_set_family(
    predictor_cls: Callable[..., _BuiltT],
    path: Path,
    *,
    db: sqlite3.Connection,
    encoder_factory: Callable[..., torch.nn.Module],
    inputs_factory: Callable[..., Callable],
    device: torch.device | str | None = None,
) -> _BuiltT:
    """Load a format-4 family; the injected factories build its encoder and its slot inputs.

    Args:
        encoder_factory: ``(config, d_in, n_champions) -> nn.Module``, one seed's module.
        inputs_factory: ``(input_spec, knowledge, db) -> (drafts, stats_cache) -> Tensor``.
    """
    path = Path(path)
    meta = load_set_meta(path)
    knowledge = load_knowledge_tables(path, meta)
    state = checked_weights(torch.load(_encoder_path(path), weights_only=True))
    device = resolve_device(device)
    modules = []
    for seed_state in state["states"]:
        module = encoder_factory(state["config"], int(state["d_in"]), int(state["n_champions"]))
        module.load_state_dict(seed_state)
        modules.append(module.to(device).eval())
    return predictor_cls(
        modules,
        weights=state,
        build_inputs=inputs_factory(state["input_spec"], knowledge, db),
        champion_index=ChampionIndex.from_names(meta["champions"]),
        thresholds=meta["thresholds"],
        temperature=meta["temperature"],
        edge_curve=EdgeCurve.from_meta(meta["edge_curve"]),
        edge_cuts_pp=cast("tuple[float, float]", tuple(float(x) for x in meta["edge_cuts_pp"])),
        recipe=meta["recipe"],
        pro_excess_baseline=meta["pro_excess_baseline"],
        knowledge=knowledge,
        artifact_stem=path.stem,
    )
