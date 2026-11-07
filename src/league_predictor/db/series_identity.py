"""Deterministic ``series_id`` minting — the one home the ingestion path shares."""

# The key is (calendar day, canonical team_a, canonical team_b, sequence-within-day):
# league is deliberately excluded, as it never participates in series grouping.

from __future__ import annotations

import hashlib


def mint_series_id(day: str, team_a: str, team_b: str, seq: int) -> str:
    """``'ser_' + sha1(day|team_a|team_b|seq)[:16]``; teams must already be canonical."""
    # day is 'YYYY-MM-DD'; seq disambiguates same-day series of the same pair (0 = first).
    key = f"{day}|{team_a}|{team_b}|{seq}"
    return "ser_" + hashlib.sha1(key.encode(), usedforsecurity=False).hexdigest()[:16]
