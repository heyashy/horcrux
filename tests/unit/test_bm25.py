"""BM25 index tests — built from synthetic chunks, no real corpus."""

import pytest

from horcrux.models import ChapterChunk
from horcrux.retrieval.bm25 import BM25Index, _tokenize

pytestmark = pytest.mark.unit


def _chunk(cid: str, text: str, *, chunk_type: str = "paragraph") -> ChapterChunk:
    return ChapterChunk(
        id=cid,
        book_num=1,
        chapter_num=1,
        chapter_title="t",
        text=text,
        chunk_type=chunk_type,
        characters=[],
        page_start=1,
    )


# ── Tokenizer ───────────────────────────────────────────────────


def test_tokenize_lowercases():
    assert _tokenize("Hello World") == ["hello", "world"]


def test_tokenize_strips_punctuation():
    assert _tokenize("Hello, world! How's it going?") == [
        "hello", "world", "how", "s", "it", "going"
    ]


def test_tokenize_handles_em_dashes_and_smart_quotes():
    """OCR output frequently has em-dashes and smart quotes."""
    assert _tokenize("Harry — alone — said “Accio”.") == [
        "harry", "alone", "said", "accio"
    ]


def test_tokenize_empty():
    assert _tokenize("") == []
    assert _tokenize("   \n\n  ") == []


# ── BM25Index basic ─────────────────────────────────────────────


def test_bm25_build_rejects_empty():
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Index.build([])


def test_bm25_finds_exact_keyword_match():
    """BM25's whole job: a chunk containing a rare query token outranks one
    that doesn't."""
    chunks = [
        _chunk("a", "the cat sat on the mat"),
        _chunk("b", "conjunctivitis affects the eye"),
        _chunk("c", "ron and harry walked the corridor"),
    ]
    index = BM25Index.build(chunks)
    hits = index.search("conjunctivitis", top_k=3, source="paragraph")
    assert hits[0].id == "b"


def test_bm25_skips_zero_score_hits():
    """Hits with no token overlap shouldn't be returned even if there's
    space in top_k."""
    chunks = [
        _chunk("a", "alpha beta"),
        _chunk("b", "gamma delta"),
    ]
    index = BM25Index.build(chunks)
    hits = index.search("epsilon", top_k=10, source="paragraph")
    assert hits == []


def test_bm25_returns_scored_candidates_with_source_tag():
    """Need ≥3 chunks for IDF to be positive — single-doc corpora are an
    edge case BM25Okapi handles by returning negative IDF, which our
    zero-score filter then drops."""
    chunks = [
        _chunk("a", "this is a paragraph chunk", chunk_type="paragraph"),
        _chunk("b", "alpha beta gamma"),
        _chunk("c", "delta epsilon zeta"),
    ]
    index = BM25Index.build(chunks)
    hits = index.search("paragraph", top_k=1, source="paragraph")
    assert hits[0].source == "paragraph"
    assert hits[0].id == "a"


def test_bm25_top_k_truncates():
    chunks = [_chunk(f"id-{i}", f"common word number {i}") for i in range(20)]
    index = BM25Index.build(chunks)
    hits = index.search("word", top_k=5, source="paragraph")
    assert len(hits) == 5


def test_bm25_idf_outweighs_term_frequency():
    """A rare token in chunk A beats a common token repeated in chunk B.
    Padded with filler chunks so the corpus has enough docs for the rare
    token to score positively under BM25's IDF formula."""
    chunks = [
        _chunk("rare-once", "the conjunctivitis"),
        _chunk("common-many", "the the the the the the"),
        _chunk("filler-1", "alpha beta gamma delta"),
        _chunk("filler-2", "epsilon zeta eta theta"),
        _chunk("filler-3", "iota kappa lambda mu"),
    ]
    index = BM25Index.build(chunks)
    hits = index.search("conjunctivitis", top_k=2, source="paragraph")
    assert hits[0].id == "rare-once"


def test_bm25_score_is_positive_float():
    """Scores should be floats and positive for actual matches —
    callers shouldn't have to guess the type. Larger corpus so IDF
    is well-behaved (single-doc corpora produce negative IDF)."""
    chunks = [
        _chunk("a", "the wand chooses the wizard"),
        _chunk("b", "ron and harry walked the corridor"),
        _chunk("c", "potions class with snape was tense"),
    ]
    index = BM25Index.build(chunks)
    hits = index.search("wand", top_k=1, source="paragraph")
    assert isinstance(hits[0].score, float)
    assert hits[0].score > 0
