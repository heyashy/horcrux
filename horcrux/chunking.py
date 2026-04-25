"""Semantic paragraph chunking with sliding-window cosine similarity.

Splits chapter text into chunks where adjacent sentences are semantically
cohesive. Cuts where similarity drops below a threshold (topic shift).
One sentence of overlap carried forward as context bridge — protects
against bad cuts from patchy OCR.

Threshold is a hyperparameter:
  - lower (~0.25) → more cuts → smaller, more topic-precise chunks
  - higher (~0.50) → fewer cuts → larger chunks, more context per chunk
  - default 0.35 is the design-doc value; tune via threshold-sweep tool

Sentence segmentation uses spaCy's `sentencizer` pipe — lightweight, no
parser or NER overhead. Lazily loaded and cached.
"""

from collections.abc import Callable

import numpy as np
import spacy

from horcrux.models import Chapter, ChapterChunk, make_chunk_id

EncodeFn = Callable[[list[str]], np.ndarray]
"""Callable that takes a list of sentences and returns a (N, D) array
of L2-normalised embeddings — so dot(a, b) = cosine similarity."""

ExtractCharsFn = Callable[[str], list[str]]
"""Callable that takes chunk text and returns list of character IDs."""


_NLP = None


def _get_segmenter():
    """Lazy-load a blank English spaCy pipeline with just the sentencizer."""
    global _NLP  # noqa: PLW0603 — module-level cache for the spaCy pipeline
    if _NLP is None:
        _NLP = spacy.blank("en")
        _NLP.add_pipe("sentencizer")
    return _NLP


def split_sentences(text: str) -> list[str]:
    """Sentence-segment text. Returns non-empty whitespace-stripped sentences."""
    if not text or not text.strip():
        return []
    nlp = _get_segmenter()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def chunk_chapter_text(  # noqa: PLR0913 — every arg is a tuned hyperparameter
    text: str,
    encode_fn: EncodeFn,
    *,
    similarity_threshold: float = 0.35,
    min_chunk_tokens: int = 100,
    max_chunk_tokens: int = 400,
    overlap_sentences: int = 1,
) -> list[str]:
    """Split chapter text into semantically-cohesive chunks.

    Sliding-window algorithm:
      1. Embed each sentence (normalised → cosine = dot product).
      2. Walk through; track current chunk's sentences and token count.
      3. At each adjacent-sentence boundary:
         - If similarity < threshold AND current ≥ min_chunk_tokens → cut.
         - If current + next > max_chunk_tokens → force cut.
      4. Carry `overlap_sentences` from end of one chunk to the next.

    Args:
        text: chapter text (already cleansed).
        encode_fn: returns L2-normalised embeddings for a list of sentences.
        similarity_threshold: cosine cut threshold.
        min_chunk_tokens: floor — don't cut shorter than this.
        max_chunk_tokens: ceiling — force cut at or above this.
        overlap_sentences: how many sentences carry forward as context.

    Returns:
        list of chunk texts in chapter order.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    embeddings = encode_fn(sentences)

    chunks: list[str] = []
    current: list[str] = [sentences[0]]
    current_tokens = _count_tokens(sentences[0])

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]))
        next_tokens = _count_tokens(sentences[i])

        force_cut = (current_tokens + next_tokens) > max_chunk_tokens
        soft_cut = sim < similarity_threshold and current_tokens >= min_chunk_tokens

        if force_cut or soft_cut:
            chunks.append(" ".join(current))
            # Carry overlap forward — protects against a bad cut from
            # OCR noise interrupting an otherwise-coherent passage.
            # Note: `current[-0:]` evaluates to the whole list (Python
            # quirk), so guard explicitly when overlap_sentences == 0.
            overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
            current = [*overlap, sentences[i]]
            current_tokens = sum(_count_tokens(s) for s in current)
        else:
            current.append(sentences[i])
            current_tokens += next_tokens

    if current:
        chunks.append(" ".join(current))

    return chunks


def _count_tokens(text: str) -> int:
    """Whitespace-split token count. Cheap proxy for true tokenizer count."""
    return len(text.split())


def chunk_chapter(
    chapter: Chapter,
    encode_fn: EncodeFn,
    extract_chars_fn: ExtractCharsFn,
    *,
    similarity_threshold: float = 0.35,
) -> list[ChapterChunk]:
    """Build all ChapterChunks for a chapter — one chapter-level chunk
    plus N paragraph-level chunks. Each chunk's `characters` payload
    is populated via `extract_chars_fn` (typically using the Tier 3
    alias dictionary).

    The chapter-level chunk is the whole chapter as one unit — for
    analytical queries that want broad narrative context. The
    paragraph-level chunks are semantically cohesive sub-sections —
    for factual / specific-scene queries.
    """
    chunks: list[ChapterChunk] = []

    # 1. Chapter-level chunk (one per chapter, chunk_index = 0).
    chunks.append(
        ChapterChunk(
            id=make_chunk_id(chapter.book_num, chapter.chapter_num, 0, "chapter"),
            book_num=chapter.book_num,
            chapter_num=chapter.chapter_num,
            chapter_title=chapter.chapter_title,
            text=chapter.text,
            chunk_type="chapter",
            characters=extract_chars_fn(chapter.text),
            page_start=chapter.page_start,
        )
    )

    # 2. Paragraph-level chunks via semantic chunking.
    paragraph_texts = chunk_chapter_text(
        chapter.text,
        encode_fn,
        similarity_threshold=similarity_threshold,
    )
    for i, ptext in enumerate(paragraph_texts):
        chunks.append(
            ChapterChunk(
                id=make_chunk_id(chapter.book_num, chapter.chapter_num, i, "paragraph"),
                book_num=chapter.book_num,
                chapter_num=chapter.chapter_num,
                chapter_title=chapter.chapter_title,
                text=ptext,
                chunk_type="paragraph",
                characters=extract_chars_fn(ptext),
                page_start=chapter.page_start,
            )
        )

    return chunks
