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


# ── Phase 5 — synthesis ──────────────────────────────────────────


class Finding(BaseModel):
    """The synthesis agent's typed output.

    Strict-RAG enforcement happens at three layers:
      1. System prompt forbids parametric knowledge.
      2. Schema: `source_ids` has min_length=1 — the model can't return
         an answer with no citations and have it parse.
      3. Runtime check (in `agents.synthesise`): every returned source_id
         must exist in the candidate set passed to the agent. Catches
         the case where the model invents a plausible-looking ID.

    `conviction` is a 1-5 scale — see the system prompt for the rubric
    embedded in the agent's instructions. Field descriptions here are
    tool-schema-visible to the model and are part of the prompt surface.
    """

    answer: str = Field(
        description=(
            "Direct, complete answer to the user's question. Plain prose, "
            "no markdown. Must be derivable from the provided passages alone."
        ),
    )
    source_ids: list[str] = Field(
        min_length=1,
        description=(
            "IDs of every passage that supports the answer. Use the exact "
            "ID strings from the numbered context — never invent IDs."
        ),
    )
    conviction: int = Field(
        ge=1,
        le=5,
        description=(
            "Confidence in the answer, 1-5. "
            "5: unambiguous direct evidence in passages. "
            "4: clearly implied by passages, no contradictions. "
            "3: supported by passages but with caveats or partial coverage. "
            "2: plausible from passages but significant gaps remain. "
            "1: passages don't really answer this — say so in `gaps`. "
            "When uncertain between two values, pick the lower number."
        ),
    )
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Any aspects of the question the passages don't cover, or "
            "contradictions between passages. Empty list if no gaps. "
            "Contradictions cap conviction at 3."
        ),
    )


# ── Phase 6 — research mode (multi-step planning + aggregation) ──


class Plan(BaseModel):
    """A research plan produced by the planner agent.

    Decomposes a complex query into focused sub-questions. Each
    sub-question is dispatched to its own retrieve+synthesise branch
    in parallel via LangGraph's `Send`; the aggregator then merges the
    sub-findings into a coherent final answer.

    The cap is intentional. More than ~5 sub-queries dilutes the final
    synthesis (too many overlapping findings to reconcile) and the
    parallel cost grows linearly. The schema allows up to 8 for
    headroom on rare deeply-multifaceted questions.
    """

    sub_questions: list[str] = Field(
        min_length=1,
        max_length=8,
        description=(
            "Focused sub-questions that decompose the original query. "
            "Each should be answerable from a localised set of passages. "
            "Aim for 3-5 sub-questions; use fewer if the question is "
            "already focused."
        ),
    )
    rationale: str = Field(
        default="",
        description=(
            "Brief reasoning for why these sub-questions cover the original "
            "query. Shown in the trace, not in the final answer."
        ),
    )


class SubFinding(BaseModel):
    """One sub-question's worth of work.

    Captures the question, the synthesis output for it, and the candidate
    set that was retrieved. Persisted in `ResearchReport.sub_findings` so
    the trace command can re-render the per-sub-query reasoning long after
    the live run is over.
    """

    sub_question: str
    finding: Finding
    candidates: list[ScoredCandidate]


class ResearchReport(BaseModel):
    """The output of a research-mode (multi-step planning) query.

    Carries the entire reasoning chain — the plan, every sub-finding, and
    the aggregator's final answer — as one typed value. The CLI's `/trace`
    command reads this directly to re-render how the system arrived at
    the answer; persistence (when added) just means writing this dataclass
    to disk.

    `source_ids` aggregates citations across all sub-findings that the
    final answer references. The aggregator is responsible for keeping
    this list non-empty (strict-RAG also applies at the aggregate layer).
    """

    query: str = Field(description="The query as the retrievers saw it (post-rewrite).")
    original_query: str = Field(description="What the user typed.")
    plan: Plan
    sub_findings: list[SubFinding]
    answer: str = Field(
        description=(
            "Final coherent answer, synthesised across all sub-findings. "
            "Plain prose, no markdown."
        ),
    )
    source_ids: list[str] = Field(
        min_length=1,
        description="Unique citations referenced in the final answer.",
    )
    conviction: int = Field(ge=1, le=5)
    gaps: list[str] = Field(default_factory=list)


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
