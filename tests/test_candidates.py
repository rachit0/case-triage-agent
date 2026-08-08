"""Part 1 is a recall filter, so the tests are recall tests."""
from __future__ import annotations

import pytest

from app.candidates import generate_candidates, select_for_investigation
from app.data_loader import load_cases, normalise_account, tokens

# Pairs read by hand out of the CSV. If a change to blocking drops one of these,
# the agent never gets a chance to see it and the test must fail loudly.
MUST_BE_PROPOSED = [
    ("CS-64254", "CS-29126"),   # reworded: 'empty export' vs 'blank Excel download'
    ("CS-47101", "CS-50425"),   # reworded: 'feed delayed' vs 'missing daily file'
    ("CS-42208", "CS-43728"),   # reworded: 'webhooks not firing' vs 'no event callbacks'
    ("CS-34123", "CS-19742"),   # reworded: 'rate limit' vs 'throttled'
    ("CS-53642", "CS-34906"),   # reworded: 'login failure' vs 'cannot access portal'
    ("CS-60493", "CS-64238"),   # reworded + carries a prompt injection
    ("CS-68292", "CS-27367"),   # lookalike that is NOT a duplicate
    ("CS-13442", "CS-56388"),   # follow-up, NOT a duplicate
    ("CS-12050", "CS-23224"),   # account typo: 'Ostara Energy' / 'Ostara Eergy'
    ("CS-41584", "CS-70689"),   # account typo: 'Marston Gate Hotels' / 'Htoels'
    ("CS-17398", "CS-40692"),   # '  Norvig Textiles ' / 'NORVIG TEXTILES', both no email
]


@pytest.fixture(scope="module")
def candidates():
    return generate_candidates()


def test_all_hard_pairs_survive_blocking(candidates):
    proposed = {frozenset((c.case_a, c.case_b)) for c in candidates}
    missing = [p for p in MUST_BE_PROPOSED if frozenset(p) not in proposed]
    assert not missing, f"blocking lost these pairs: {missing}"


def test_blocking_actually_reduces_the_space(candidates):
    n = len(load_cases())
    assert len(candidates) < n * (n - 1) / 2 * 0.05, "blocking is not filtering enough"


def test_pair_ordering_is_stable():
    """pair_id must not depend on iteration order, or the DB unique constraint
    on pair_id would let the same pair in twice."""
    a = {c.pair_id for c in generate_candidates()}
    b = {c.pair_id for c in generate_candidates()}
    assert a == b


def test_account_normalisation_collapses_noise_but_not_typos():
    assert normalise_account("  Ostara Energy ") == normalise_account("OSTARA ENERGY")
    assert normalise_account("Ostara Eergy") != normalise_account("Ostara Energy")


def test_boilerplate_is_stripped_from_tokens():
    cases = {c.case_id: c for c in load_cases()}
    mobile = cases["CS-51220"]           # ends with 'Sent from my mobile device'
    assert "mobile" not in tokens(mobile.body_clean)
    assert "mobile" in tokens(mobile.description)


def test_selection_is_stratified_not_top_n(candidates):
    selected = select_for_investigation(candidates, limit=12)
    assert len(selected) == 12
    scores = [c.cheap_score for c in selected]
    # A pure top-N skim would be all >= 0.85 on this dataset.
    assert min(scores) < 0.60, "selection collapsed to the easy high-score band"
    assert max(scores) >= 0.85, "selection has no clear-cut pairs at all"
