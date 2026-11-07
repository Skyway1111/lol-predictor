"""Map skill-filter latent strength onto the dashboard's unanchored team display."""

from __future__ import annotations

from league_predictor.probability import ELO_ORIGIN, ELO_PER_LOGODDS


def latent_strength_to_display(strength: float | None) -> float | None:
    """Map latent team strength onto the 1500-centered display scale."""
    if strength is None:
        return None
    return ELO_ORIGIN + strength * ELO_PER_LOGODDS
