"""Champion names and indexes, including facts aliases."""

from __future__ import annotations

from league_predictor.draft_grading.champion_attributes.ability_store import AbilityFeatureStore
from league_predictor.draft_grading.champion_attributes.normalize import (
    _CHAMPION_NAME_ALIASES,
    normalize_champion_name,
)
from league_predictor.draft_grading.state import DraftState


class ChampionIndex:
    """Champion name <-> index over a universe fixed at construction; misses are UNKNOWN.

    Attributes:
        UNKNOWN: the out-of-universe sentinel id; embedding-table sizes follow the universe.
    """

    def __init__(self, store: AbilityFeatureStore):
        self._build(store.champions)

    def _build(self, names: list[str]) -> None:
        """Fill the name<->index tables, alias lookup, and UNKNOWN sentinel from ``names``."""
        self._name_to_idx: dict[str, int] = {name: i for i, name in enumerate(names)}
        self._idx_to_name: dict[int, str] = dict(enumerate(names))
        self._lookup: dict[str, str] = {normalize_champion_name(name): name for name in names}
        for alias, target in _CHAMPION_NAME_ALIASES.items():
            if target in self._name_to_idx:
                self._lookup[normalize_champion_name(alias)] = target
        self._normalize = normalize_champion_name
        self.UNKNOWN: int = len(names)

    @classmethod
    def from_names(cls, names: list[str]) -> ChampionIndex:
        """Index over an explicit champion NAME LIST in index order; no store constructed."""
        index = cls.__new__(cls)
        index._build(list(names))
        return index

    def encode(self, champion: str) -> int:
        """Champion name -> index through the self-owned alias lookup; UNKNOWN if unresolved."""
        if champion in self._name_to_idx:
            return self._name_to_idx[champion]
        resolved = self._lookup.get(self._normalize(champion))
        if resolved is None:
            return self.UNKNOWN
        return self._name_to_idx[resolved]

    def encode_draft(self, draft: DraftState) -> list[int]:
        """A draft's 10 champion ids in position order (blue 0-4, red 5-9)."""
        return [
            self.encode(slot.champion_id) for comp in (draft.blue, draft.red) for slot in comp.slots
        ]

    def decode(self, index: int) -> str:
        """Index -> champion name."""
        return self._idx_to_name.get(index, "<UNKNOWN>")

    @property
    def known_champions(self) -> list[str]:
        """Known champion names in index order; the universe ``from_names`` round-trips."""
        return list(self._name_to_idx.keys())

    def __len__(self) -> int:
        """Total entries including UNKNOWN."""
        return len(self._name_to_idx) + 1
