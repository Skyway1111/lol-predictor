"""The machine-resource guards (`resources.py`): fan-out width bounded by memory."""

from __future__ import annotations

from types import SimpleNamespace

from league_predictor import resources


def test_available_memory_reports_binary_gib(monkeypatch):
    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(available=8 * 1024**3),
    )

    assert resources.available_memory_gb() == 8.0


def _patch_box(monkeypatch, *, available_gb: float, cpus: int) -> None:
    monkeypatch.setattr(resources, "available_memory_gb", lambda: available_gb)
    monkeypatch.setattr(resources.os, "cpu_count", lambda: cpus)


def test_bounded_workers_is_memory_bound_on_a_tight_box(monkeypatch):
    """A heavy parent (little available memory) shrinks the pool below the CPU width."""
    _patch_box(monkeypatch, available_gb=11.0, cpus=16)
    assert resources.bounded_workers(2.0) == 4  # (11 - 3 reserve) // 2


def test_bounded_workers_is_cpu_bound_on_a_roomy_box(monkeypatch):
    """With memory to spare the pool caps at cpu_count - 1 (never oversubscribes cores)."""
    _patch_box(monkeypatch, available_gb=200.0, cpus=16)
    assert resources.bounded_workers(2.0) == 15


def test_bounded_workers_floors_at_serial_when_starved(monkeypatch):
    """A starved box degrades to one worker (the serial footprint), never zero."""
    _patch_box(monkeypatch, available_gb=1.0, cpus=16)
    assert resources.bounded_workers(2.0) == 1


def test_bounded_workers_respects_an_explicit_cap(monkeypatch):
    _patch_box(monkeypatch, available_gb=200.0, cpus=16)
    assert resources.bounded_workers(2.0, cap=6) == 6
