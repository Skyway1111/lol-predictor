"""Artifact paths and device selection for the draft family."""

from pathlib import Path

import torch


def family_meta_path(root: Path) -> Path:
    """The family meta ``{root.name}.json`` — name-APPEND, so a dotted stem keeps its own."""
    root = Path(root)
    return root.with_name(f"{root.name}.json")


def _encoder_path(path: Path) -> Path:
    """The set-family encoder artifact path: ``{path}.encoder.pt``."""
    return path.with_name(f"{path.name}.encoder.pt")


def resolve_device(device: torch.device | str | None) -> torch.device:
    """The device a family loads onto; ``None`` takes CUDA when it is there."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device) if isinstance(device, str) else device
