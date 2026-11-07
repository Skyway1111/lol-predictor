"""Fit the set encoder: five pinned seeds, Kendall-weighted heads, a walk-forward early stop."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812 - the torch.nn.functional convention

from league_predictor.draft_grading.grader.predictor import FORWARD_ROWS
from league_predictor.draft_grading.grader.randomness import pin_fit_randomness
from league_predictor.draft_grading.grader.set_encoder import DraftSetEncoder
from league_predictor.draft_grading.grader.slot_inputs import N_SLOTS, SlotInputs
from league_predictor.torch_graphs import capture_training, graph_training
from league_predictor.vocab.roles import ROLE_ORDER

if TYPE_CHECKING:
    from league_predictor.draft_grading.grader.dense_labels import DenseTargets
    from league_predictor.draft_grading.recipe import SetTrainConfig

logger = logging.getLogger(__name__)

#: The set encoder's seed base; the pre-train uses 42 and the booster bag 1000+i
#: (`training/bundle.py`), so each family's draws stay their own.
SEED_BASE = 2000

#: The chronological tail held out of every seed's fit for the early stop.
VAL_FRACTION = 0.1

#: Heads whose labels are z-scored per role before the fit; the rest are probabilities or
#: class ids and carry their own scale.
_SCALED_SLOT_HEADS = ("dpm", "damage_share")


@dataclass(frozen=True)
class SetWeights:
    """One fitted set family: every seed's state, what it was fit on, and how it stopped.

    Attributes:
        input_spec: `SlotInputs.to_spec`: ``columns``, one name per ``d_in`` column in row
            order, and the ``blocks`` that partition them. A serving build rebuilds the
            layout from it (`SlotInputs.build_for`), never from source.
        label_scale: per head, the train-slice ``(mu, sd)`` its labels were z-scored by;
            provenance for a head read, never part of the served score.
        history: per seed, the early-stop epoch and the per-epoch losses.
    """

    config: dict[str, Any]
    d_in: int
    n_champions: int
    states: list[dict[str, torch.Tensor]]
    input_spec: dict[str, Any]
    history: list[dict[str, Any]] = field(default_factory=list)
    label_scale: dict[str, list[list[float]]] = field(default_factory=dict)

    def to_state(self) -> dict[str, Any]:
        """A ``weights_only``-loadable payload: tensors and primitives, never a numpy array."""
        return {
            "config": self.config,
            "d_in": self.d_in,
            "n_champions": self.n_champions,
            "states": self.states,
            "input_spec": self.input_spec,
            "history": self.history,
            "label_scale": self.label_scale,
        }


def _split_at(n: int) -> int:
    """First row of the chronological early-stop tail; the fit slice is everything before."""
    return max(1, min(int(n * (1.0 - VAL_FRACTION)), n - 1)) if n > 1 else 1


def _role_scale(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-role ``(mu, sd)`` over the masked rows of a ``(n, 10)`` slot label."""
    mu = np.zeros(len(ROLE_ORDER))
    sd = np.ones(len(ROLE_ORDER))
    for role in range(len(ROLE_ORDER)):
        columns = [role, role + len(ROLE_ORDER)]
        seen = values[:, columns][mask[:, columns]]
        if seen.size:
            mu[role] = seen.mean()
            sd[role] = seen.std() or 1.0
    return mu, sd


def _scaled_labels(
    dense: DenseTargets, config: SetTrainConfig, split: int
) -> tuple[dict[str, np.ndarray], dict[str, list[list[float]]]]:
    """Z-score the continuous heads' labels on the fit slice; class and long_game pass through."""
    out: dict[str, np.ndarray] = {}
    scale: dict[str, list[list[float]]] = {}
    for name in config.aux_heads:
        values, mask = dense.labels[name], dense.masks[name]
        if name in _SCALED_SLOT_HEADS:
            mu, sd = _role_scale(values[:split], mask[:split])
            tiled_mu, tiled_sd = np.tile(mu, 2), np.tile(sd, 2)
            out[name] = ((values - tiled_mu) / tiled_sd).astype(np.float32)
            scale[name] = [mu.tolist(), sd.tolist()]
        elif name == "gd15":
            seen = values[:split][mask[:split]]
            mu = float(seen.mean()) if seen.size else 0.0
            sd = (float(seen.std()) if seen.size else 1.0) or 1.0
            out[name] = ((values - mu) / sd).astype(np.float32)
            scale[name] = [[mu], [sd]]
        else:
            out[name] = values
    return out, scale


def _head_loss(
    name: str, prediction: torch.Tensor, target: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor | None:
    """One head's masked loss over a batch, or None when the batch masks every entry."""
    if not indices.numel():
        return None
    if name == "class":
        flat = prediction.reshape(-1, prediction.shape[-1])[indices]
        return F.cross_entropy(flat, target.reshape(-1)[indices])
    picked, wanted = prediction.reshape(-1)[indices], target.reshape(-1)[indices]
    if name == "long_game":
        return F.binary_cross_entropy_with_logits(picked, wanted)
    return F.mse_loss(picked, wanted)


class _Corpus:
    """The fit's device-resident tensors, sliced by row index."""

    def __init__(
        self,
        inputs: torch.Tensor,
        *,
        outcomes: np.ndarray,
        offsets: np.ndarray | None,
        labels: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        weights: np.ndarray,
        device: torch.device,
    ):
        self.x = inputs.to(device)
        self.y = torch.as_tensor(outcomes, dtype=torch.float32, device=device)
        self.w = torch.as_tensor(weights, dtype=torch.float32, device=device)
        self.offset = (
            None
            if offsets is None
            else torch.as_tensor(offsets, dtype=torch.float32, device=device)
        )
        self.labels = {
            name: torch.as_tensor(
                value, dtype=torch.long if name == "class" else torch.float32, device=device
            )
            for name, value in labels.items()
        }
        self.masks = masks

    def batches(self, order: np.ndarray, chunk: int):
        """Upload one epoch's row order and compact label indices without GPU mask scans."""
        boundaries = np.arange(0, len(order) + chunk, chunk)
        selections = {}
        for name, mask in self.masks.items():
            width = int(np.prod(mask.shape[1:]))
            positions = np.flatnonzero(mask[order])
            offsets = np.searchsorted(positions, boundaries * width)
            indices = torch.as_tensor(positions % (chunk * width), device=self.x.device)
            selections[name] = indices, offsets
        rows = torch.as_tensor(order, device=self.x.device)
        for i, start in enumerate(boundaries[:-1]):
            yield (
                rows[start : start + chunk],
                {
                    name: indices[offsets[i] : offsets[i + 1]]
                    for name, (indices, offsets) in selections.items()
                },
            )


def _main_loss(score: torch.Tensor, corpus: _Corpus, rows: torch.Tensor | slice) -> torch.Tensor:
    """The weighted outcome objective shared by fitting and early stopping."""
    logit = score if corpus.offset is None else score + corpus.offset[rows]
    return F.binary_cross_entropy_with_logits(logit, corpus.y[rows], weight=corpus.w[rows])


def _batch_terms(
    model: DraftSetEncoder, corpus: _Corpus, idx: torch.Tensor, selections: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """The main head's weighted BCE over the residual target, plus every configured head."""
    score, aux = model(corpus.x[idx])
    terms = {"main": _main_loss(score, corpus, idx)}
    for name, prediction in aux.items():
        loss = _head_loss(name, prediction, corpus.labels[name][idx], selections[name])
        if loss is not None:
            terms[name] = loss
    return terms


def _validation_loss(model: DraftSetEncoder, corpus: _Corpus, split: int, chunk: int) -> float:
    """The main head's own objective on the chronological tail, in float32, never AMP."""
    model.eval()
    total, n = 0.0, corpus.y.shape[0]
    with torch.no_grad():
        for start in range(split, n, chunk):
            rows = slice(start, min(start + chunk, n))
            score, _ = model(corpus.x[rows])
            loss = _main_loss(score, corpus, rows)
            total += float(loss) * len(corpus.y[rows])
    return total / max(n - split, 1)


@graph_training()
def _fit_one_seed(  # noqa: PLR0915, C901 - optimizer, checkpoint, and graph lifetimes belong to one fit
    corpus: _Corpus,
    config: SetTrainConfig,
    *,
    d_in: int,
    n_champions: int,
    seed: int,
    split: int,
    epochs: int | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """One seed's fit: Kendall-weighted heads, AdamW, early stop on the chronological tail.

    Args:
        split: first row of the early-stop tail; ``split == len(corpus)`` fits every row.
        epochs: run exactly this many epochs and keep the last, the fixed-round path the
            final fit takes once the folds have chosen the count. ``None`` early-stops.
    """
    device = corpus.x.device
    torch.manual_seed(seed)
    model = DraftSetEncoder(config, d_in, n_champions).to(device)
    log_vars = {
        name: torch.nn.Parameter(torch.zeros(1, device=device))
        for name in ("main", *config.aux_heads)
    }
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.parameters()), "weight_decay": config.weight_decay},
            {"params": list(log_vars.values()), "weight_decay": 0.0},
        ],
        lr=config.lr,
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None  # pyright: ignore[reportPrivateImportUsage]  torch.amp public API, stub omits re-export
    use_graph = use_amp and not config.identity_residual and split >= config.batch_size
    complete = True
    if use_graph:
        eager = model.forward
        with torch.autocast("cuda", cache_enabled=False):
            capture_training(model, (corpus.x[: config.batch_size],))
        captured = model.forward
        model.forward = lambda x: (
            captured(x) if complete and len(x) == config.batch_size else eager(x)
        )
    rng = np.random.default_rng(seed)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val, best_epoch, stale = float("inf"), 0, 0
    history: dict[str, Any] = {"seed": seed, "val_main": [], "train": []}
    for epoch in range(1, (epochs or config.max_epochs) + 1):
        model.train()
        totals: dict[str, float] = {}
        permutation = rng.permutation(split)
        for idx, selections in corpus.batches(permutation, config.batch_size):
            # An absent head needs grad=None, so AdamW must not update its moments or decay.
            complete = all(indices.numel() for indices in selections.values())
            with torch.amp.autocast("cuda", enabled=use_amp):  # pyright: ignore[reportPrivateImportUsage]  torch.amp public API, stub omits re-export
                terms = _batch_terms(model, corpus, idx, selections)
                # Kendall uncertainty weighting, the phase-1 idiom: each head pays
                # 0.5·exp(-s)·L + 0.5·s, clamped so a solved head takes no runaway weight.
                loss = model.identity_penalty()
                for name, term in terms.items():
                    log_var = log_vars[name].clamp(min=config.log_var_floor).squeeze()
                    loss = loss + 0.5 * torch.exp(-log_var) * term + 0.5 * log_var
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            for name, term in terms.items():
                totals[name] = totals.get(name, 0.0) + float(term.detach())

        batches = max(1, -(-split // config.batch_size))
        history["train"].append({k: round(v / batches, 5) for k, v in totals.items()})
        history["log_vars"] = {
            name: round(float(p.detach().clamp(min=config.log_var_floor)), 5)
            for name, p in log_vars.items()
        }
        if epochs is not None:
            best_epoch, best_state = epoch, model.state_dict()
            continue
        value = _validation_loss(model, corpus, split, config.batch_size)
        history["val_main"].append(round(value, 6))
        if value < best_val:
            best_val, best_epoch, stale = value, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    history["best_epoch"] = best_epoch
    history["best_val_main"] = None if epochs is not None else round(best_val, 6)
    history["epochs_run"] = len(history["train"])
    if use_graph:
        optimizer.zero_grad(set_to_none=True)
        del model.forward, captured, loss, terms, term
        gc.collect()
        torch.cuda.empty_cache()
    return {k: v.detach().cpu().clone() for k, v in best_state.items()}, history


def fit_set_encoder(
    inputs: torch.Tensor,
    outcomes: np.ndarray,
    offsets: np.ndarray | None,
    dense: DenseTargets | None,
    config: SetTrainConfig,
    *,
    weights: np.ndarray | None = None,
    n_champions: int = 0,
    epochs: int | None = None,
    device: str | torch.device | None = None,
) -> SetWeights:
    """Fit ``config.seeds`` encoders on one chronological corpus; the served score is their mean.

    Args:
        inputs: ``(N, 10, D)`` slot rows in facts order, `SlotInputs.build`'s output.
        offsets: the per-row residual target the main head rides; required by, and only
            read under, ``outcome_offset='team_form'``.
        dense: the auxiliary labels; required when the config carries any head.
        weights: per-row importance weights on the main head; ones when absent.
        epochs: fit every row for exactly this many epochs and keep the last. The final fit
            takes this path at the folds' chosen count, so it never chooses an epoch on the
            rows the OOF calibration assumes it never saw. ``None`` early-stops on the tail.

    Returns:
        Every seed's state, with the input spec and the label scales it was fit against.

    Raises:
        ValueError: the corpus is empty, or the offsets or dense labels the config needs
            are missing or misaligned.
    """
    pin_fit_randomness()
    n = int(outcomes.shape[0])
    slot_inputs = SlotInputs(config)
    if n == 0 or tuple(inputs.shape) != (n, N_SLOTS, slot_inputs.width):
        raise ValueError(
            f"fit_set_encoder: inputs {tuple(inputs.shape)} against {n} outcomes and a "
            f"({N_SLOTS}, {slot_inputs.width}) slot row."
        )
    if (config.outcome_offset != "none") != (offsets is not None):
        raise ValueError(
            f"fit_set_encoder: outcome_offset={config.outcome_offset!r} and the offsets "
            "travel together — the main head rides one or fits the plain outcome."
        )
    if config.aux_heads and dense is None:
        raise ValueError(f"fit_set_encoder: heads {config.aux_heads} but no dense labels.")

    # A fixed-epoch fit reads every row; only an early-stopped one holds the tail back.
    split = n if epochs is not None else _split_at(n)
    labels, scale = _scaled_labels(dense, config, split) if dense is not None else ({}, {})
    masks = {name: dense.masks[name] for name in labels} if dense is not None else {}
    chosen = device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus = _Corpus(
        inputs,
        outcomes=outcomes,
        offsets=offsets,
        labels=labels,
        masks=masks,
        weights=np.ones(n) if weights is None else weights,
        device=torch.device(chosen),
    )
    states, history = [], []
    for i in range(config.seeds):
        state, report = _fit_one_seed(
            corpus,
            config,
            d_in=slot_inputs.width,
            n_champions=n_champions,
            seed=SEED_BASE + i,
            split=split,
            epochs=epochs,
        )
        states.append(state)
        history.append(report)
        logger.info(
            "set encoder seed %d: epoch %d/%d, val main %s, log vars %s",
            SEED_BASE + i,
            report["best_epoch"],
            report["epochs_run"],
            report["best_val_main"] or "not held out",
            report.get("log_vars", {}),
        )
    return SetWeights(
        config=config.to_dict(),
        d_in=slot_inputs.width,
        n_champions=n_champions,
        states=states,
        input_spec=slot_inputs.to_spec(),
        history=history,
        label_scale=scale,
    )


def set_scores(
    weights: SetWeights,
    config: SetTrainConfig,
    inputs: torch.Tensor,
    *,
    device: str | torch.device | None = None,
    chunk: int = 8 * FORWARD_ROWS,
) -> np.ndarray:
    """The seed-mean antisymmetric score per draft — the served quantity, in float32.

    Args:
        chunk: rows per forward, a multiple of `grader.predictor.FORWARD_ROWS`. Every
            forward runs at exactly this many rows, the tail zero-padded, so row ``i``
            always sits at ``i % chunk`` of one fixed shape and reads the same in any
            company (`draft-architecture.md`, the D6 pin). No layer mixes rows.

    Returns:
        ``(N,)`` float64. A mirrored board's score is the exact negation: each seed's is,
        and the mean of exact negations is one.

    Raises:
        ValueError: ``chunk`` is not a multiple of ``FORWARD_ROWS``.
    """
    if chunk <= 0 or chunk % FORWARD_ROWS:
        raise ValueError(f"set_scores(chunk={chunk}) must be a positive multiple of FORWARD_ROWS")
    target = torch.device(
        device or config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    x = inputs.to(target)
    n = x.shape[0]
    totals = torch.zeros(n, device=target)
    models = []
    for state in weights.states:
        model = DraftSetEncoder(config, weights.d_in, weights.n_champions).to(target)
        model.load_state_dict(state)
        models.append(model.eval())
    with torch.no_grad():
        for start in range(0, n, chunk):
            block = x[start : start + chunk]
            padded = torch.cat([block, block.new_zeros((chunk - block.shape[0], *block.shape[1:]))])
            for model in models:
                totals[start : start + block.shape[0]] += model(padded)[0][: block.shape[0]]
    return (totals / len(weights.states)).cpu().numpy().astype(np.float64)
