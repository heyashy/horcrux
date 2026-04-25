"""Pydantic models — single source of truth for the shapes that flow through
the pipeline. Models grow phase by phase.
"""

import uuid
from typing import Literal

from pydantic import BaseModel, Field

# ── Phase 1 — OCR ────────────────────────────────────────────────

class RawPage(BaseModel):
    """A single OCR'd page from the corpus PDF.

    Carries no chapter or book metadata — that comes from chapter detection
    in Phase 2. Keeping OCR concerns separate from structure-detection
    concerns means each layer can fail independently and be tested in
    isolation.
    """

    page_num: int = Field(ge=1, description="1-indexed page number in the source PDF")
    text: str = Field(description="OCR text for this page, whitespace preserved")


# ── Phase 2 — chapter detection + chunking ───────────────────────

class Chapter(BaseModel):
    """A chapter detected from a contiguous page range.

    `book_num` is inferred from chapter-number resets in the source PDF
    (a multi-book compilation). `chapter_num` resets to 1 at each book
    boundary.
    """

    book_num: int = Field(ge=1)
    chapter_num: int = Field(ge=1)
    chapter_title: str
    text: str = Field(description="cleansed, concatenated text for the chapter")
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class ChapterChunk(BaseModel):
    """A chunk ready for embedding and Qdrant upsert.

    `id` is a deterministic UUID5 (see `make_chunk_id`). Re-running the
    chunker on the same chapter produces the same IDs, which makes Qdrant
    upsert idempotent — re-ingest never duplicates points.

    `chunk_type` distinguishes the two collections we maintain:
      - "chapter"   → one point per chapter, lives in hp_chapters
      - "paragraph" → semantic chunk, lives in hp_paragraphs

    `characters` is populated by character extraction (Phase 2). Filterable
    at query time via Qdrant array-keyword `match` predicates.
    """

    id: str = Field(description="deterministic UUID5 — see make_chunk_id")
    book_num: int = Field(ge=1)
    chapter_num: int = Field(ge=1)
    chapter_title: str
    text: str
    chunk_type: Literal["chapter", "paragraph"]
    characters: list[str] = Field(default_factory=list)
    page_start: int = Field(ge=1)


# ── Phase 4 — retrieval ──────────────────────────────────────────

class ScoredCandidate(BaseModel):
    """A retrieval hit returned by search.

    `score` is whatever the source put in the slot — cosine similarity for
    a single-collection ANN search, RRF score for a fused list. Don't compare
    scores across sources without normalising; do compare *ranks*.

    `source` records which collection produced the hit (for explanations and
    for letting the synthesis agent reason over breadth-vs-precision). The
    rest of the fields are payload pulled straight from Qdrant.
    """

    id: str
    score: float
    source: Literal["chapter", "paragraph"]
    text: str
    book_num: int = Field(ge=1)
    chapter_num: int = Field(ge=1)
    chapter_title: str
    page_start: int = Field(ge=1)
    characters: list[str] = Field(default_factory=list)


# ── Helpers ──────────────────────────────────────────────────────

# Fixed namespace UUID for chunk ID derivation. Arbitrary but stable —
# changing this value changes every chunk ID in every collection, so don't.
_CHUNK_NAMESPACE = uuid.UUID("8d6f5a31-2c4e-4b5f-a1b2-c3d4e5f60718")


def make_chunk_id(
    book_num: int,
    chapter_num: int,
    chunk_index: int,
    chunk_type: str,
) -> str:
    """Derive a deterministic UUID5 for a chunk.

    Same inputs always produce the same ID. This is what makes Qdrant
    upsert idempotent across re-ingest runs — without it, every run would
    create new points and the index would grow indefinitely.

    `chunk_index` is per-chapter (resets at each chapter), not global.
    """
    raw = f"b{book_num}-c{chapter_num}-i{chunk_index}-{chunk_type}"
    return str(uuid.uuid5(_CHUNK_NAMESPACE, raw))
