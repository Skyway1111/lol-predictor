"""Format-4 serving."""  # sightline-ok: 27 - composes artifacts, encoders and draft contracts

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, cast

import numpy as np
import torch

from league_predictor.draft_grading.grader import set_encoder, slot_inputs
from league_predictor.draft_grading.grader.grades import assign_grade
from league_predictor.draft_grading.grader.predictor import (
    FORWARD_ROWS,
    DraftScore,
    SwapRequest,
)
from league_predictor.draft_grading.grader.set_artifact import (
    checked_weights,
    load_set_family,
    save_set_family,
)
from league_predictor.draft_grading.grader.slot_inputs import IDENTITY_COLUMN
from league_predictor.draft_grading.grader.swaps import swap_champion, validate_swap_query
from league_predictor.draft_grading.recipe import GATE_K
from league_predictor.draft_grading.state import DraftState
from league_predictor.probability import sigmoid_mirrored
from league_predictor.vocab.roles import MIRROR_SLOTS, N_SLOTS, ROLE_ORDER

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Callable, Mapping, Sequence

    from league_predictor.draft_grading.grader.bulk_stats import BulkStatsCache
    from league_predictor.draft_grading.grader.embeddings import ChampionIndex
    from league_predictor.draft_grading.grader.grades import EdgeCurve
    from league_predictor.draft_grading.grader.knowledge_tables import KnowledgeTables

#: Rows the construction-time antisymmetry probe runs; the property is per row.
_PROBE_ROWS = 4


class _SeedGraph:
    """The seed bag's fixed-size CUDA forward, with unchanged arithmetic and live weights."""

    def __init__(self, modules: Sequence[torch.nn.Module], sample: torch.Tensor):
        self.inputs = sample.clone()
        stream = torch.cuda.Stream(device=sample.device)
        stream.wait_stream(torch.cuda.current_stream(sample.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                for module in modules:
                    module(self.inputs)
        torch.cuda.current_stream(sample.device).wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.outputs = torch.stack([module(self.inputs)[0] for module in modules])

    def __call__(self, rows: torch.Tensor):
        self.inputs.copy_(rows)
        self.graph.replay()
        return self.outputs


@contextmanager
def _served(modules: Sequence[torch.nn.Module]):
    """Run the block with every module in ``eval``, each left in the mode it arrived in.

    Yields:
        Nothing. Serving must never read a fit-time module's dropout, and a predictor must
        never leave its caller's module in another mode than it found it.
    """
    modes = [module.training for module in modules]
    for module in modules:
        module.eval()
    try:
        yield
    finally:
        for module, training in zip(modules, modes, strict=True):
            module.train(training)


class SetDraftPredictor:
    """Serve a format-4 family: K seed encoders over the computed slot input, then the curve.

    Attributes:
        temperature: the fitted scale on the seed-mean score; the served probability is
            ``sigmoid(temperature * score)``, read off the favoured side so a mirrored
            draft scores exactly ``1 - p``.
    """

    def __init__(
        self,
        modules: Sequence[torch.nn.Module],
        *,
        weights: Mapping,
        build_inputs: Callable[..., torch.Tensor],
        champion_index: ChampionIndex,
        thresholds: Sequence[float],
        temperature: float = 1.0,
        edge_curve: EdgeCurve | None = None,
        edge_cuts_pp: tuple[float, float] | None = None,
        recipe: Mapping | None = None,
        pro_excess_baseline: str = "team_form",
        knowledge: KnowledgeTables | None = None,
        artifact_stem: str = "",
    ):
        if (edge_curve is None) != (edge_cuts_pp is None):
            raise ValueError(
                "edge_curve and edge_cuts_pp travel together: an edge-semantics predictor "
                "carries both, an uncalibrated fit-time predictor neither"
            )
        self._weights = checked_weights(weights)
        self.modules = tuple(modules)
        if len(self.modules) != len(self._weights["states"]):
            raise ValueError(
                f"{len(self.modules)} seed modules against {len(self._weights['states'])} "
                "persisted seed states — the bag and its weights are one thing"
            )
        self._build_inputs = build_inputs
        self.index = champion_index
        self.thresholds = cast(
            "tuple[float, float, float, float]", tuple(float(t) for t in thresholds)
        )
        self.temperature = float(temperature)
        self.edge_curve = edge_curve
        self.edge_cuts_pp = edge_cuts_pp
        self.recipe = dict(recipe or {})
        self.k = float(self.recipe.get("k", GATE_K))
        self.pro_excess_baseline = pro_excess_baseline
        self.knowledge = knowledge
        self.artifact_stem = artifact_stem
        self._graphs: dict[tuple[torch.nn.Module, ...], _SeedGraph] = {}
        self._forward_lock = Lock()
        self._assert_antisymmetric()

    # ------------------------- What the family declares ------------------------- #

    @property
    def grade_semantics(self) -> str:
        """Return ``edge`` for persisted semantics or ``quantile`` for an uncalibrated fit."""
        return "edge" if self.edge_curve is not None else "quantile"

    @property
    def design_columns(self) -> list[str]:
        """Slot-major design names: each slot's own input columns, blue top..sup then red."""
        return slot_inputs.design_columns(self._weights["input_spec"])

    @property
    def design_blocks(self) -> dict[str, np.ndarray]:
        """Each frozen slot-input block's ``design`` columns: its own span, over ten slots."""
        return slot_inputs.design_blocks(self._weights["input_spec"], int(self._weights["d_in"]))

    def edge_pp(self, score: float) -> float:
        """Calibrated edge (pp of win rate) at a served score; antisymmetric in the score.

        Raises:
            RuntimeError: an uncalibrated predictor carries no curve, so there is no honest
                number to return, never a silent 0.
        """
        if self.edge_curve is None:
            raise RuntimeError(
                f"draft artifact {self.artifact_stem or '<unsaved>'} carries no frozen edge "
                "curve (quantile semantics) - edge_pp is undefined for it. Refit the slot."
            )
        return self.edge_curve.edge_pp(score)

    # ------------------------------ Inference ------------------------------ #

    def _assert_antisymmetric(self) -> None:
        """Refuse a seed whose half-swap does not negate the score; the served mirror IS that.

        Raises:
            RuntimeError: naming the seed and the largest deviation found.
        """
        probe = self._probe()
        # Two calls, never one mirror-closed batch: that batch is its own mirror, so a
        # row reading its company would pass it.
        mirrored = self.seed_scores(probe[:, MIRROR_SLOTS, :])
        for seed, scores in enumerate(self.seed_scores(probe)):
            worst = float(np.abs(mirrored[seed] + scores).max())
            if worst != 0.0:
                raise RuntimeError(
                    f"seed {seed} scores the half-swapped input off its own negation by "
                    f"{worst:.3e} — the served function reads p once from the score, so the "
                    "mirror would be wrong by that much. Encode each side through one shared "
                    "call on (side, enemy), never as one permuted ten-row reduction."
                )

    def _probe(self) -> torch.Tensor:
        """A random LEGAL ``(_PROBE_ROWS, 10, d_in)`` input for the antisymmetry gate."""
        # The identity arm's column is an embedding row, not a feature: a normal draw
        # there indexes the table out of range, so it draws ids over the universe.
        generator = torch.Generator().manual_seed(0)
        shape = (_PROBE_ROWS, N_SLOTS)
        probe = torch.randn(*shape, int(self._weights["d_in"]), generator=generator)
        columns = list(self._weights["input_spec"]["columns"])
        if IDENTITY_COLUMN in columns:
            ids = torch.randint(self.index.UNKNOWN + 1, shape, generator=generator)
            probe[..., columns.index(IDENTITY_COLUMN)] = ids.to(probe.dtype)
        return probe

    def seed_scores(self, x: torch.Tensor) -> np.ndarray:
        """``(K, N)`` each seed's score, every forward in ``eval`` at ``FORWARD_ROWS`` rows.

        Returns:
            float64 rows; the fixed shape keeps a row's score free of the company it rode
            in (the D6 pin), and ``eval`` keeps a fit-time module's dropout out of it.
        """
        n = int(x.shape[0])
        out = np.empty((len(self.modules), n), dtype=np.float64)
        if not n:
            return out
        pad = -n % FORWARD_ROWS
        if pad:
            x = torch.cat([x, x.new_zeros((pad, *x.shape[1:]))])
        with self._forward_lock, _served(self.modules), torch.no_grad():
            device = next(self.modules[0].parameters()).device
            if device.type == "cuda" and all(
                next(module.parameters()).device == device for module in self.modules
            ):
                rows = x.to(device)
                if self.modules not in self._graphs:
                    self._graphs[self.modules] = _SeedGraph(self.modules, rows[:FORWARD_ROWS])
                graph = self._graphs[self.modules]
                for start in range(0, n, FORWARD_ROWS):
                    count = min(FORWARD_ROWS, n - start)
                    out[:, start : start + count] = (
                        graph(rows[start : start + FORWARD_ROWS])[:, :count].cpu().numpy()
                    )
                return out
            for seed, module in enumerate(self.modules):
                rows = x.to(next(module.parameters()).device, non_blocking=True)
                parts = [
                    module(rows[start : start + FORWARD_ROWS])[0]
                    for start in range(0, n + pad, FORWARD_ROWS)
                ]
                out[seed] = torch.cat(parts)[:n].detach().cpu().numpy().astype(np.float64)
        return out

    def reads(
        self, drafts: list[DraftState], stats_cache: BulkStatsCache | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(p[N], members[K, N])``: the served probability, tempered, and each seed's own."""
        return self.read_inputs(self._build_inputs(drafts, stats_cache))

    def read_inputs(self, inputs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """The served probability and seed members over already-built slot rows."""
        seeds = self.seed_scores(inputs)
        return sigmoid_mirrored(self.temperature * seeds.mean(axis=0)), sigmoid_mirrored(seeds)

    def _score(self, p: float) -> DraftScore:
        """One probability as a graded score through this family's own frozen curve."""
        return DraftScore(
            prob_blue_wins=p,
            grade_blue=assign_grade(p, self.thresholds),  # type: ignore[arg-type]
            grade_red=assign_grade(1.0 - p, self.thresholds),  # type: ignore[arg-type]
            edge_pp_blue=None if self.edge_curve is None else self.edge_curve.edge_pp(p),
        )

    def require_stats_cache(self, stats_cache: BulkStatsCache | None, surface: str) -> None:
        """Batch surfaces refuse the per-cell SQL fallback — a silent cost cliff.

        Raises:
            RuntimeError: naming the surface and the cache to build.
        """
        if stats_cache is None:
            raise RuntimeError(
                f"SetDraftPredictor.{surface} requires a stats_cache — build ONE "
                "BulkStatsCache(conn, baseline=predictor.pro_excess_baseline) "
                "(grader/features.py) per connection and pass it to every batch call."
            )

    def predict(self, draft: DraftState, stats_cache: BulkStatsCache | None = None) -> DraftScore:
        """Score a single draft: the seed-mean score, tempered, graded by the frozen curve."""
        return self._score(float(self.reads([draft], stats_cache)[0][0]))

    def predict_batch(
        self, drafts: list[DraftState], stats_cache: BulkStatsCache | None = None
    ) -> list[DraftScore]:
        """Score N drafts in one forward per seed; row ``i`` equals ``predict(drafts[i])``."""
        self.require_stats_cache(stats_cache, "predict_batch")
        if not drafts:
            return []
        return [self._score(float(p)) for p in self.reads(drafts, stats_cache)[0]]

    def predict_swaps(
        self,
        draft: DraftState,
        side: str,
        role: str,
        alternatives: list[str],
        *,
        stats_cache: BulkStatsCache | None = None,
    ) -> np.ndarray:
        """Batched single-champion-swap scoring against the known enemy, in one forward."""
        validate_swap_query(side, role, role)
        enemy = draft.red if side == "blue" else draft.blue
        return self.predict_swaps_marginalized(
            draft,
            side,
            role,
            alternatives,
            enemy_role=role,
            enemy_responses=[(enemy.slots[ROLE_ORDER.index(role)].champion_id, 1.0)],
            stats_cache=stats_cache,
        )

    def predict_swaps_marginalized(
        self,
        draft: DraftState,
        side: str,
        role: str,
        alternatives: list[str],
        *,
        enemy_role: str,
        enemy_responses: list[tuple[str, float]],
        stats_cache: BulkStatsCache | None = None,
    ) -> np.ndarray:
        """One request through the batch path — the same rows, so the same numbers."""
        request = SwapRequest(
            side=side,
            role=role,
            alternatives=list(alternatives),
            enemy_role=enemy_role,
            enemy_responses=list(enemy_responses),
        )
        return self.predict_swaps_marginalized_batch(draft, [request], stats_cache=stats_cache)[0]

    def predict_swaps_marginalized_batch(
        self,
        draft: DraftState,
        requests: list[SwapRequest],
        stats_cache: BulkStatsCache | None = None,
    ) -> list[np.ndarray]:
        """Per request, the marginalized swap scores over the shared draft, in one forward."""
        return self.predict_swaps_marginalized_many([(draft, requests)], stats_cache=stats_cache)[0]

    def predict_swaps_marginalized_many(
        self,
        batches: list[tuple[DraftState, list[SwapRequest]]],
        stats_cache: BulkStatsCache | None = None,
    ) -> list[list[np.ndarray]]:
        """Score swap-request batches from many drafts in one forward; every row is a board.

        Returns:
            ``outputs[batch_index][request_index]``, each entry the enemy-response weighted
            mean over the counterfactual boards that request names.
        """
        self.require_stats_cache(stats_cache, "predict_swaps_marginalized_many")
        results: list[list[np.ndarray]] = []
        boards: list[DraftState] = []
        scored: list[tuple[int, int, int, int, np.ndarray]] = []
        for batch_idx, (draft, requests) in enumerate(batches):
            results.append([np.empty(0, dtype=np.float64) for _ in requests])
            for request_idx, request in enumerate(requests):
                validate_swap_query(request.side, request.role, request.enemy_role)
                if not request.alternatives:
                    continue
                if not request.enemy_responses:
                    results[batch_idx][request_idx] = np.zeros(
                        len(request.alternatives), dtype=np.float64
                    )
                    continue
                scored.append(
                    (
                        batch_idx,
                        request_idx,
                        len(request.enemy_responses),
                        len(request.alternatives),
                        np.array([w for _c, w in request.enemy_responses], dtype=np.float64),
                    )
                )
                boards.extend(swap_boards(draft, request))
        if boards:
            probs = self.reads(boards, stats_cache)[0]
            offset = 0
            for batch_idx, request_idx, n_enemies, n_alts, weights in scored:
                block = probs[offset : offset + n_enemies * n_alts].reshape(n_enemies, n_alts)
                results[batch_idx][request_idx] = (block * weights[:, None]).sum(axis=0)
                offset += n_enemies * n_alts
        return results

    # ------------------------------- Artifact ------------------------------- #

    def design(self, drafts: list[DraftState], stats_cache: BulkStatsCache) -> np.ndarray:
        """The exact computed slot inputs, flattened in the frozen column order."""
        rows = self._build_inputs(drafts, stats_cache)
        return rows.detach().cpu().numpy().reshape(len(drafts), -1)

    def save(self, root: Path) -> None:
        """Persist this family; the format-4 layout contract is ``grader/set_artifact.py``.

        Raises:
            ValueError: an uncalibrated or knowledge-less predictor, which must not serve.
        """
        if self.edge_curve is None or self.edge_cuts_pp is None:
            raise ValueError("persisted draft families require edge_curve and edge_cuts_pp")
        if self.knowledge is None:
            raise ValueError(
                "persisted draft families require their KnowledgeTables — the frozen serving "
                "view of champion knowledge (knowledge_tables.build_knowledge_tables)"
            )
        save_set_family(
            Path(root),
            weights=self._weights,
            champions=self.index.known_champions,
            knowledge=self.knowledge,
            temperature=self.temperature,
            thresholds=self.thresholds,
            edge_curve=self.edge_curve,
            edge_cuts_pp=self.edge_cuts_pp,
            recipe=self.recipe,
            pro_excess_baseline=self.pro_excess_baseline,
        )

    @classmethod
    def load(
        cls,
        path: Path,
        db: sqlite3.Connection,
        *,
        encoder_factory: Callable[..., torch.nn.Module] | None = None,
        inputs_factory: Callable[..., Callable] | None = None,
        device: torch.device | str | None = None,
    ) -> SetDraftPredictor:
        """Load a saved format-4 family; a ``None`` factory takes the production wiring."""
        if encoder_factory is None or inputs_factory is None:
            # Through the module at call time: the fitted pair has one home, the encoder's.
            fitted_encoder, fitted_inputs = set_encoder.production_wiring()
            encoder_factory = encoder_factory or fitted_encoder
            inputs_factory = inputs_factory or fitted_inputs
        return load_set_family(
            cls,
            Path(path),
            db=db,
            encoder_factory=encoder_factory,
            inputs_factory=inputs_factory,
            device=device,
        )


def swap_boards(draft: DraftState, request: SwapRequest) -> list[DraftState]:
    """Row ``k*N + i``: the real board with alternative ``i`` facing enemy response ``k``."""
    enemy_side = "red" if request.side == "blue" else "blue"
    return [
        swap_champion(
            swap_champion(draft, enemy_side, request.enemy_role, enemy),
            request.side,
            request.role,
            alternative,
        )
        for enemy, _weight in request.enemy_responses
        for alternative in request.alternatives
    ]
