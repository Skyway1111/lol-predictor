"""The snapshot store's group budget: groups past it go by last use."""

from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import date

from league_predictor import config
from league_predictor.predraft.pipeline import fit_corpus_store as store

_END = date(2026, 1, 1)


@dataclass(frozen=True)
class _Key:
    name: str

    def as_tuple(self) -> tuple[object, ...]:
        return (self.name,)

    @property
    def group(self) -> str:
        return store.group_of(self.as_tuple())


def _store(key: _Key, *, used_at: int) -> None:
    store.store(key, end=_END, digest="d")
    path = store._cache_dir() / f"fit_corpus_{key.group}_{_END.isoformat()}.pkl"
    os.utime(path, (used_at, used_at))


def _groups() -> set[str]:
    return {p.stem.split("_")[2] for p in store._cache_dir().glob("fit_corpus_*.pkl")}


def test_a_store_past_the_budget_evicts_the_least_recently_used_group(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    keys = [_Key(str(i)) for i in range(store._KEEP_GROUPS + 1)]
    for used_at, key in enumerate(keys, start=1_000_000):
        _store(key, used_at=used_at)
    assert _groups() == {k.group for k in keys[1:]}


def test_a_hit_keeps_its_group_through_the_next_eviction(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(store, "_facts_prefix_digests", lambda _c, ends: dict.fromkeys(ends, "d"))
    keys = [_Key(str(i)) for i in range(store._KEEP_GROUPS + 1)]
    for used_at, key in enumerate(keys[:-1], start=1_000_000):
        _store(key, used_at=used_at)
    with closing(sqlite3.connect(":memory:")) as conn:
        assert store.find_extendable(conn, keys[0], end=_END)[0] is not None
    store.store(keys[-1], end=_END, digest="d")
    assert _groups() == {keys[0].group, *(k.group for k in keys[2:])}
