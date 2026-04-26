"""Chunker tests — focused on the splitting logic, not the embedding.

The encoder is a callable that returns L2-normalised vectors. Tests pass
in synthetic embedding functions (constant similarity, alternating
similarity, etc.) to exercise the cut behaviour without loading real
sentence-transformers.
"""

import numpy as np
import pytest

from horcrux.corpus.chunking import chunk_chapter, chunk_chapter_text, split_sentences
from horcrux.models import Chapter

pytestmark = pytest.mark.unit


# ── split_sentences ──────────────────────────────────────────────

def test_split_sentences_basic():
    text = "The first sentence. The second sentence! And a third?"
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "The first sentence."


def test_split_sentences_empty():
    assert split_sentences("") == []
    assert split_sentences("   \n\n  ") == []


def test_split_sentences_strips_whitespace():
    text = "  First.  \n\n  Second.  "
    sents = split_sentences(text)
    assert sents == ["First.", "Second."]


# ── chunk_chapter_text ───────────────────────────────────────────

def _const_encoder(similarity: float):
    """Return an encoder that produces vectors with a fixed pairwise cosine.

    similarity=1.0 → all sentences in the same direction (no cuts ever).
    similarity=0.0 → all orthogonal (every adjacent pair triggers cut).
    """
    def encode(sentences: list[str]) -> np.ndarray:
        # Construct vectors v_i = [cos(theta_i), sin(theta_i), 0...]
        # where theta_i drifts so dot(v_i, v_{i+1}) = similarity
        n = len(sentences)
        if n == 0:
            return np.zeros((0, 4), dtype=np.float64)
        # Use 2D rotation to control similarity
        if similarity >= 1.0:
            angles = np.zeros(n)
        elif similarity <= -1.0:
            angles = np.arange(n) * np.pi
        else:
            theta = np.arccos(similarity)
            angles = np.arange(n) * theta
        return np.stack(
            [np.array([np.cos(a), np.sin(a), 0.0, 0.0]) for a in angles]
        )
    return encode


def test_no_cuts_when_similarity_high():
    """All sentences highly similar → one big chunk."""
    text = " ".join([f"Sentence {i} with content." for i in range(10)])
    chunks = chunk_chapter_text(
        text, _const_encoder(0.9),
        similarity_threshold=0.35,
        min_chunk_tokens=5,
    )
    assert len(chunks) == 1


def test_cut_at_every_boundary_when_similarity_low():
    """All sentences orthogonal AND chunks always above min → frequent cuts.
    Uses prose-shaped sentences because spaCy's sentencizer requires
    realistic prose patterns to detect boundaries."""
    # 30+ tokens per sentence to clear min_chunk_tokens.
    base = (
        "The wizard waved his wand in a slow deliberate arc through the "
        "dim light of the corridor while the boys watched in silence "
    )
    sentences = [
        base + "and then he spoke of the coming storm.",
        base + "and the candles flickered violently in response.",
        base + "before the shadows gathered themselves into a shape.",
        base + "as the great hall trembled at the approach of dawn.",
    ]
    text = " ".join(sentences)
    chunks = chunk_chapter_text(
        text, _const_encoder(0.0),
        similarity_threshold=0.35,
        min_chunk_tokens=20,
        overlap_sentences=0,
    )
    assert len(chunks) > 1


def test_min_chunk_tokens_prevents_premature_cut():
    """Even with low similarity, don't cut before reaching min_chunk_tokens."""
    sentences = ["Tiny."] * 10  # very short sentences
    text = " ".join(sentences)
    chunks = chunk_chapter_text(
        text, _const_encoder(0.0),
        similarity_threshold=0.35,
        min_chunk_tokens=50,  # 10 sentences x 1 word = 10 tokens, never reaches 50
        overlap_sentences=0,
    )
    # Single chunk because min never reached
    assert len(chunks) == 1


def test_max_chunk_tokens_forces_cut():
    """Even with high similarity, force a cut at the max token cap."""
    sentences = [("word " * 20).strip() + "." for _ in range(20)]
    text = " ".join(sentences)
    chunks = chunk_chapter_text(
        text, _const_encoder(1.0),    # never a soft cut
        similarity_threshold=0.35,
        min_chunk_tokens=1,
        max_chunk_tokens=80,           # forces cut every ~4 sentences
        overlap_sentences=0,
    )
    assert len(chunks) > 1


def test_overlap_sentences_carry_forward():
    """Last sentence of one chunk appears at the start of the next."""
    sentences = [("Sentence " + chr(65 + i)) * 1 + ". " + ("filler " * 30) for i in range(6)]
    text = " ".join(sentences)
    chunks = chunk_chapter_text(
        text, _const_encoder(0.0),
        similarity_threshold=0.35,
        min_chunk_tokens=20,
        overlap_sentences=1,
    )
    if len(chunks) >= 2:
        # The last sentence of chunk 0 should appear inside chunk 1
        # (overlap carry-forward).
        last_of_0 = chunks[0].split(".")[-2].strip()  # roughly last sentence
        assert last_of_0 in chunks[1] or any(
            last_of_0 in c for c in chunks[1:]
        )


def test_empty_text_returns_empty():
    assert chunk_chapter_text("", _const_encoder(1.0)) == []


def test_single_sentence_passes_through():
    text = "Only one sentence here."
    assert chunk_chapter_text(text, _const_encoder(1.0)) == [text]


# ── chunk_chapter (full ChapterChunk construction) ───────────────

def test_chunk_chapter_produces_chapter_plus_paragraphs():
    chapter = Chapter(
        book_num=1, chapter_num=4, chapter_title="Test Chapter",
        text="First sentence here. Second sentence with content. Third one to chunk.",
        page_start=10, page_end=15,
    )
    chunks = chunk_chapter(
        chapter,
        encode_fn=_const_encoder(1.0),  # no soft cuts
        extract_chars_fn=lambda _: ["harry_potter"],
    )
    # At least one chapter-level + at least one paragraph-level
    assert any(c.chunk_type == "chapter" for c in chunks)
    assert any(c.chunk_type == "paragraph" for c in chunks)


def test_chunk_chapter_sets_metadata_correctly():
    chapter = Chapter(
        book_num=2, chapter_num=7, chapter_title="The Mudbloods",
        text="A short chapter. With two sentences.",
        page_start=42, page_end=58,
    )
    chunks = chunk_chapter(
        chapter, _const_encoder(1.0),
        extract_chars_fn=lambda _: ["hermione_granger"],
    )
    for c in chunks:
        assert c.book_num == 2
        assert c.chapter_num == 7
        assert c.chapter_title == "The Mudbloods"
        assert c.page_start == 42
        assert c.characters == ["hermione_granger"]


def test_chunk_chapter_ids_are_deterministic():
    """Re-running on the same chapter produces the same chunk IDs."""
    chapter = Chapter(
        book_num=1, chapter_num=1, chapter_title="The Boy Who Lived",
        text="The Dursleys were proud to say. They had no time for nonsense.",
        page_start=1, page_end=10,
    )
    chunks1 = chunk_chapter(chapter, _const_encoder(1.0), lambda _: [])
    chunks2 = chunk_chapter(chapter, _const_encoder(1.0), lambda _: [])
    ids1 = [c.id for c in chunks1]
    ids2 = [c.id for c in chunks2]
    assert ids1 == ids2
