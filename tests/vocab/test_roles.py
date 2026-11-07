"""Tests for the canonical five-position role vocabulary."""

from league_predictor.vocab.roles import N_ROLES, ROLE_ORDER


def test_roles_are_the_canonical_five():
    assert ROLE_ORDER == ("top", "jng", "mid", "bot", "sup")
    assert N_ROLES == len(ROLE_ORDER) == 5


def test_consumers_share_the_one_constant():
    # Corpus builders, lineup canonicalization, and draft state share the vocab leaf.
    from league_predictor.draft_grading import state
    from league_predictor.predraft.features.snapshots import stream
    from league_predictor.predraft.ratings import impact_corpus
    from league_predictor.predraft.ratings.skill_filter import corpus_build

    assert corpus_build.ROLE_ORDER is ROLE_ORDER
    assert stream.ROLE_ORDER is ROLE_ORDER
    assert impact_corpus.ROLE_ORDER is ROLE_ORDER
    assert state.ROLE_ORDER is ROLE_ORDER
