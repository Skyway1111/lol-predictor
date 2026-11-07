"""Machine-resource guards for fan-out work; memory-aware pool sizing lives here."""

from __future__ import annotations

import os

import psutil


def available_memory_gb() -> float:
    """Available physical memory in binary GiB — the headroom every fan-out guard checks."""
    return psutil.virtual_memory().available / 1024**3


def bounded_workers(
    per_worker_gb: float, *, reserve_gb: float = 3.0, cap: int | None = None
) -> int:
    """Pool width the box holds: (available GiB - ``reserve_gb``) / ``per_worker_gb``."""
    # Capped at ``cpu_count - 1`` (or ``cap``) and floored at 1, so a starved box degrades
    # to serial. Available memory excludes the caller's own footprint.
    by_memory = int(max(0.0, available_memory_gb() - reserve_gb) // per_worker_gb)
    by_cpu = max(1, (os.cpu_count() or 2) - 1)
    width = min(by_cpu, max(1, by_memory))
    if cap is not None:
        width = min(width, cap)
    return max(1, width)
