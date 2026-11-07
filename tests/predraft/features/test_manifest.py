"""Served original-family manifest contracts."""

from __future__ import annotations

import pytest

from league_predictor.predraft.features import manifest
from league_predictor.predraft.pipeline.chosen_features import SERVED_FEATURES


def test_universe_covers_diffs_sides_and_context():
    universe = manifest.full_feature_universe()
    for base in manifest.diff_keys():
        assert f"diff_{base}" in universe
    assert "t1_golddiffat15" in universe
    assert "t2_golddiffat15" in universe
    assert "is_playoffs" in universe


def test_served_features_are_covered_by_the_manifest():
    manifest.assert_served_features_covered(SERVED_FEATURES)
    original = [
        name for name in SERVED_FEATURES if not name.startswith(("rating_", "lineup_", "recency_"))
    ]
    assert original
    assert set(original) <= manifest.full_feature_universe()


def test_assert_served_features_covered_raises_on_typo():
    with pytest.raises(ValueError, match="not in the original-family manifest universe"):
        manifest.assert_served_features_covered(["t1_golddiffat15", "t1_stmp_rate_typo"])


def test_assert_served_features_covered_accepts_other_family_prefixes():
    manifest.assert_served_features_covered(
        ["rating_strength_a", "lineup_mean_elo_diff", "recency_momentum_diff"]
    )


def test_fingerprint_is_stable_and_flag_derived():
    # Regenerate after a semantic manifest edit; the new fingerprint invalidates feature caches.
    assert manifest.feature_schema_fingerprint() == "529ba1c2e5c7971e"
    assert len(manifest.feature_schema_fingerprint()) == 16


def test_no_duplicate_bases():
    bases = [spec.base for spec in manifest.MANIFEST]
    assert len(bases) == len(set(bases))


def test_context_specs_emit_only_themselves():
    for spec in manifest.MANIFEST:
        if spec.context:
            assert spec.emitted() == (spec.base,)
            assert not spec.diff
            assert not spec.drop_sides


def test_fingerprint_tracks_semantics_but_not_documentation(monkeypatch):
    from dataclasses import replace

    original = manifest.MANIFEST
    before = manifest.feature_schema_fingerprint()
    monkeypatch.setattr(
        manifest, "MANIFEST", (replace(original[0], note="clarified"), *original[1:])
    )
    assert manifest.feature_schema_fingerprint() == before
    monkeypatch.setattr(
        manifest, "MANIFEST", (replace(original[0], diff=not original[0].diff), *original[1:])
    )
    assert manifest.feature_schema_fingerprint() != before
