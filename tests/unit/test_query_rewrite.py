"""Query rewriter tests — exercise the fuzzy correction logic without
loading the real chunks.json. Tests pass an explicit `vocab` so behaviour
is deterministic and decoupled from the gold-tier artefact.
"""

import pytest

from horcrux.retrieval.query_rewrite import correct_query

pytestmark = pytest.mark.unit


# A focused vocab — only the tokens the tests actually need to match
# against. Real corpus has tens of thousands of tokens; using a small
# set keeps the fuzzy-match outcomes deterministic.
_VOCAB = (
    "harry", "potter", "hermione", "granger", "ron", "weasley",
    "voldemort", "cedric", "diggory", "conjunctivitis", "curse",
    "dragon", "dumbledore", "snape",
)


def _correct(query: str, **kwargs) -> tuple[str, list[tuple[str, str]]]:
    """Test helper — pin vocab to the fixed test set."""
    return correct_query(query, vocab=_VOCAB, **kwargs)


# ── In-vocab tokens pass through unchanged ──────────────────────


def test_in_vocab_query_unchanged():
    rewritten, corrections = _correct("harry potter and the dragon")
    assert rewritten == "harry potter and the dragon"
    assert corrections == []


def test_short_oov_tokens_not_corrected():
    """Short OOV tokens are below the correction threshold — too
    risky to fuzz-match short strings."""
    rewritten, corrections = _correct("xy zz")
    assert rewritten == "xy zz"
    assert corrections == []


def test_oov_with_no_close_match_passes_through():
    """Genuinely unrelated long token — no correction applied."""
    rewritten, corrections = _correct("zzzzzzzzz")
    assert rewritten == "zzzzzzzzz"
    assert corrections == []


# ── Real typos get corrected ────────────────────────────────────


def test_typo_corrected_to_nearest_vocab_token():
    rewritten, corrections = _correct("voldermort")
    assert rewritten == "voldemort"
    assert corrections == [("voldermort", "voldemort")]


def test_severe_typo_still_corrected_above_threshold():
    rewritten, corrections = _correct("conjuncatvitus")
    assert rewritten == "conjunctivitis"
    assert ("conjuncatvitus", "conjunctivitis") in corrections


def test_multiple_typos_corrected_independently():
    rewritten, corrections = _correct("cedrik diggry killed by voldermort")
    assert "cedric" in rewritten
    assert "voldemort" in rewritten
    typo_to_fix = dict(corrections)
    assert typo_to_fix["cedrik"] == "cedric"
    assert typo_to_fix["voldermort"] == "voldemort"


def test_mix_of_in_vocab_and_typo():
    """Real-world shape: most tokens correct, one typo."""
    rewritten, corrections = _correct("who killed cedrik diggory")
    assert rewritten == "who killed cedric diggory"
    assert corrections == [("cedrik", "cedric")]


# ── Threshold behaviour ─────────────────────────────────────────


def test_high_threshold_blocks_marginal_corrections():
    """At threshold=99, even close matches don't pass."""
    rewritten, corrections = _correct("voldermort", threshold=99)
    assert rewritten == "voldermort"
    assert corrections == []


def test_low_threshold_lets_loose_matches_through():
    """At threshold=50, even loose matches get applied."""
    rewritten, _ = _correct("herzeione", threshold=50)
    # Won't equal "hermione" without checking, but should differ
    # from the input (i.e. *some* correction was applied).
    assert rewritten != "herzeione"


# ── Tokenization edge cases ────────────────────────────────────


def test_empty_query():
    rewritten, corrections = _correct("")
    assert rewritten == ""
    assert corrections == []


def test_punctuation_dropped_consistently():
    """The tokenizer strips punctuation; the rewritten query is the
    space-joined tokens, no punctuation re-added. Documented behaviour
    so tests can rely on it."""
    rewritten, _ = _correct("harry, potter!")
    assert rewritten == "harry potter"


def test_case_normalised_to_lowercase():
    rewritten, _ = _correct("HARRY Potter")
    assert rewritten == "harry potter"
