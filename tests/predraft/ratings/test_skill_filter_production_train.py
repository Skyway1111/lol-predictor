"""The Rating trainer has one fixed CUDA/DEER production path."""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

from league_predictor.predraft.ratings.skill_filter import train


def test_production_fit_refuses_to_fall_back_without_cuda(monkeypatch):
    """A missing GPU is an error, never a silent slow-path selection."""
    monkeypatch.setattr(train, "_deer_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA-capable GPU"):
        train.fit_production_bag(
            SimpleNamespace(days=[]),
            loss_end=date(2026, 1, 1),
            val_window=(date(2026, 1, 1), date(2026, 2, 1)),
        )


def test_production_fit_owns_the_deployed_k5_schedule(monkeypatch):
    """Callers supply dates only; the trainer owns K and every fit constant."""
    captured = {}
    corpus = SimpleNamespace(days=[SimpleNamespace(a_won=[0, 1, 0])])
    monkeypatch.setattr(train, "_deer_available", lambda: True)
    monkeypatch.setattr(
        train, "_gate_deer_vram", lambda games, members: captured.update(gate=(games, members))
    )

    def fake_fit(corpus_arg, **kwargs):
        captured.update(corpus=corpus_arg, kwargs=kwargs)
        return [object()] * 5

    monkeypatch.setattr(train, "_fit_deployed_bag", fake_fit)

    result = train.fit_production_bag(
        corpus,
        loss_end=date(2026, 1, 1),
        val_window=(date(2026, 1, 1), date(2026, 2, 1)),
    )

    assert len(result) == 5
    assert captured["gate"] == (3, 5)
    assert captured["kwargs"] == {
        "loss_end": date(2026, 1, 1),
        "val_window": (date(2026, 1, 1), date(2026, 2, 1)),
    }


def test_production_fit_refuses_insufficient_vram(monkeypatch):
    """The fixed path checks memory before entering the expensive solver."""
    corpus = SimpleNamespace(days=[SimpleNamespace(a_won=[0, 1, 0])])
    monkeypatch.setattr(train, "_deer_available", lambda: True)
    monkeypatch.setattr(train, "_available_vram_gb", lambda: 0.2)

    with pytest.raises(RuntimeError, match="no slower fallback"):
        train.fit_production_bag(
            corpus,
            loss_end=date(2026, 1, 1),
            val_window=(date(2026, 1, 1), date(2026, 2, 1)),
        )


def test_adjoint_skips_members_that_cannot_affect_a_later_epoch():
    done = [False, True, False, True]
    assert train._deer_backward_live([0, 1, 2, 3], done, epoch=7, max_epochs=20) == [0, 2]
    assert train._deer_backward_live([0, 2], done, epoch=20, max_epochs=20) == []
