"""Skill-filter serving backends: artifact + corpus → a queryable index."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from league_predictor.predraft.ratings.skill_filter.index_snapshot import (
    facts_identity,
    load_index_snapshot,
)
from league_predictor.predraft.ratings.skill_filter.recipe import serves_game_inputs
from league_predictor.predraft.ratings.skill_filter.serving_bag import BaggedSkillFilterIndex
from league_predictor.predraft.ratings.skill_filter.serving_index import SkillFilterIndex

if TYPE_CHECKING:
    import sqlite3
    from collections.abc import Mapping, Sequence
    from datetime import date

    from league_predictor.predraft.ratings.skill_filter.corpus_schema import Corpus
    from league_predictor.predraft.ratings.skill_filter.model import FilterConfig


def _league_remap(meta: Mapping, corpus) -> dict[int, int]:
    """Corpus league id -> the ARTIFACT's embedding row; -1 when the league is unknown to it."""
    return {
        ci: meta["league_index"].get(name, -1) for name, ci in corpus.vocab.league_index.items()
    }


def _serving_index(model, meta: dict, corpus, games) -> SkillFilterIndex:
    """Replay one model into its serving index."""
    # sightline-ok: 35 - torch-loading module stays off the cold import path
    from league_predictor.predraft.ratings.skill_filter.replay import replay_for_serving

    timelines = replay_for_serving(
        corpus, model, norm=meta["norm"], league_remap=_league_remap(meta, corpus)
    )
    return SkillFilterIndex(model, meta["norm"], timelines, corpus.vocab.org_index, games)


def _bag_serving_indexes(models, meta: dict, corpus) -> list[SkillFilterIndex]:
    """Every bag member's ``SkillFilterIndex``, batching the replays wherever members allow.

    Returns:
        Members grouped by replay-relevant config (seed excepted — seed only shaped init
        draws already frozen into the weights); each group of two or more rides one K-batched
        replay, while singletons replay solo. Per-member equivalent to a
        solo build (pinned by ``TestBagServingReplay``), member order preserved. Each group's
        replay yields one K-wide ``BagServingTimelines`` frozen member at a time, so no
        per-member object graph accumulates before ``_share_frozen_structure`` collapses them.
    """
    import dataclasses

    import torch

    # sightline-ok: 35 - torch-loading module stays off the cold import path
    from league_predictor.predraft.ratings.skill_filter.replay_bag import replay_bag_for_serving
    from league_predictor.predraft.ratings.skill_filter.serving_frozen import game_inputs

    games = game_inputs(torch, corpus) if serves_game_inputs(meta["recipe"]) else None

    groups: dict[FilterConfig, list[int]] = {}
    for i, m in enumerate(models):
        groups.setdefault(dataclasses.replace(m.config, seed=0), []).append(i)
    indexes: list = [None] * len(models)
    for idxs in groups.values():
        if len(idxs) >= 2:  # noqa: PLR2004 - a bag replay needs at least a pair
            bag_tl = replay_bag_for_serving(
                corpus,
                [models[i] for i in idxs],
                norm=meta["norm"],
                league_remap=_league_remap(meta, corpus),
            )
            for k, i in enumerate(idxs):
                indexes[i] = SkillFilterIndex(
                    models[i],
                    meta["norm"],
                    bag_tl.member(k),
                    corpus.vocab.org_index,
                    games,
                )
    members = [
        idx if idx is not None else _serving_index(m, meta, corpus, games)
        for idx, m in zip(indexes, models, strict=True)
    ]
    _share_frozen_structure(members)
    return members


# The member-INDEPENDENT ``_FrozenTimelines`` fields: pure functions of the corpus, never of a
# member's weights, so ``_share_frozen_structure`` aliases them to one tensor each per bag.
_STRUCT_FIELDS = (
    "player_key",
    "player_offset",
    "org_key",
    "org_offset",
    "team_key",
    "team_five",
    "team_tier",
    "team_games",
    "team_cont",
    "team_offset",
)


def _share_frozen_structure(members: Sequence[SkillFilterIndex]) -> None:
    """Alias every member's member-independent frozen tensors (``_STRUCT_FIELDS``) to member 0's.

    Raises:
        AssertionError: those fields are pure functions of the corpus, so the identity is
            ASSERTED before aliasing — a replay change making structure weight-dependent
            fails loudly here instead of silently serving one member's structure to all.
    """
    import torch

    if len(members) < 2:  # noqa: PLR2004 - nothing to cross-check below a pair
        return
    head = members[0]._frozen_tl
    for m in members[1:]:
        fr = m._frozen_tl
        for f in _STRUCT_FIELDS:
            ours, shared = getattr(fr, f), getattr(head, f)
            assert torch.equal(ours, shared), (
                f"frozen '{f}' differs across bag members -- structure must be corpus-determined"
            )
            setattr(fr, f, shared)


class BaggedSkillFilterBackend:
    """Replay a K-member artifact over one shared corpus into a ``BaggedSkillFilterIndex``."""

    def __init__(
        self,
        artifact_path: Path | str,
        *,
        expected_sha: str | None = None,
        use_index_snapshot: bool = False,
    ) -> None:
        self.artifact_path = Path(artifact_path)
        self.expected_sha = expected_sha
        self.use_index_snapshot = use_index_snapshot

    def build_index(
        self,
        conn: sqlite3.Connection,
        *,
        through: date,
        query_dates: Sequence[date],
        corpus: Corpus | None = None,
    ) -> BaggedSkillFilterIndex:
        """Build the bag's serving index: one shared corpus, then ``_bag_serving_indexes``.

        Returns:
            Config-uniform member groups ride one K-batched no-grad replay each; the rest
            replay member by member, SERIALLY. Never threads: the replay day loop is
            GIL-bound Python glue over tiny per-day matmuls, so a member-per-thread pool
            only thrashes the GIL — measured slower (numbers in ``benchmarks.md``).
        """
        del query_dates
        # sightline-ok: 35 - torch-loading module stays off the cold import path
        from league_predictor.predraft.ratings.skill_filter.artifact import (
            artifact_sha,
            load_bag_artifact,
            rating_dependency,
        )
        from league_predictor.predraft.ratings.skill_filter.corpus import build_corpus
        from league_predictor.predraft.ratings.skill_filter.corpus_build import corpus_signature

        content_sha = self.expected_sha or artifact_sha(self.artifact_path)
        dependency = rating_dependency(self.artifact_path, expected_sha=content_sha)
        if self.use_index_snapshot:
            cached = load_index_snapshot(
                artifact_path=self.artifact_path,
                artifact_sha=content_sha,
                source_signature=corpus.signature
                if corpus is not None
                else corpus_signature(conn, through=through),
                source_identity=facts_identity(conn, through=through),
                through=through,
            )
            if cached is not None:
                cached.rating_dependency = dependency
                return cached
        if corpus is None:
            corpus = build_corpus(conn, through=through, norm_through=through)
        models, meta = load_bag_artifact(self.artifact_path, expected_sha=self.expected_sha)
        index = BaggedSkillFilterIndex(_bag_serving_indexes(models, meta, corpus))
        index.rating_dependency = dependency
        # Provenance consumed by ``index_snapshot``.  It is attached only after a complete replay,
        # so an interrupted build can never publish a partial serving materialization.
        index.source_signature = corpus.signature
        index.through = through
        return index
