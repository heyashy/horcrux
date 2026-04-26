"""LangGraph state machine for hybrid retrieval.

A `rewrite` node runs first to fuzzy-correct typos in the query against
the corpus vocabulary. Then four parallel retrievers (paragraph-dense,
chapter-dense, paragraph-bm25, chapter-bm25) fan out, run concurrently,
and converge on a `fuse` node that RRF-merges their ranked lists.

    START → rewrite ──┬── paragraph_dense ──┐
                      ├── chapter_dense    ─┤
                      ├── paragraph_bm25   ─┤
                      └── chapter_bm25     ─┘
                                             ↓
                                           fuse  →  END

This shape suits LangGraph because:

  - Each branch is independent (no shared state until fuse).
  - Each node writes to a *distinct* state field, so no reducer is
    needed — the default TypedDict overlay merges cleanly.
  - The fan-in to `fuse` makes LangGraph wait for all four branches
    before running fusion. We get the parallel-await pattern for free,
    server-side from the graph runtime.
  - Nodes are async-friendly. Sync work (Qdrant client, BM25 score loop)
    is wrapped in `asyncio.to_thread` so the four branches actually
    overlap on the event loop, not just *look* parallel.

Resources (Qdrant client) are closed over at graph-build time. Building
once per process and invoking many times keeps the compile cost out of
every query. The BM25 indexes are loaded lazily via `lru_cache` inside
`horcrux.bm25` — same one-time cost, no plumbing through the graph.

`hybrid_search()` at the bottom of this module is the public API: a thin
async wrapper that builds (or reuses) the compiled graph and invokes it
with the user's query. Existing callers (`scripts/search.py`,
`scripts/answer.py`, `horcrux/agents.py`) keep their current import path.
"""

import asyncio
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from qdrant_client import QdrantClient

from horcrux.ingest import get_client
from horcrux.models import ScoredCandidate
from horcrux.query_rewrite import correct_query
from horcrux.retrieval import (
    reciprocal_rank_fusion,
    search_chapters,
    search_chapters_bm25,
    search_paragraphs,
    search_paragraphs_bm25,
)


class RetrievalState(TypedDict, total=False):
    """State carried through the retrieval graph.

    `total=False` so individual nodes can return partial state without
    declaring every key. Each retrieval node writes exactly one key
    (paragraph_dense / chapter_dense / paragraph_bm25 / chapter_bm25);
    fuse reads the four and writes `fused`.

    `query` is the (possibly-rewritten) text the retrievers see;
    `original_query` is what the user typed verbatim. `corrections`
    captures the (typo, fix) pairs the rewriter applied — surfaced in
    the CLI as a "did you mean…" note.

    The k-knobs are all in state rather than closure — keeps the graph
    invocations explicit and lets callers tune per-query without
    rebuilding the graph.
    """

    query: str
    original_query: str
    corrections: list[tuple[str, str]]
    character_filter: list[str] | None
    paragraph_k: int
    chapter_k: int
    top_k: int
    paragraph_dense: list[ScoredCandidate]
    chapter_dense: list[ScoredCandidate]
    paragraph_bm25: list[ScoredCandidate]
    chapter_bm25: list[ScoredCandidate]
    fused: list[ScoredCandidate]


# ── Node factories ───────────────────────────────────────────────
#
# Nodes are produced by closure factories so the Qdrant client is bound
# once at graph-build time. Module-level functions would have to either
# rebuild the client per call or pull from a global — both worse.


def _paragraph_dense_node(client: QdrantClient):
    async def node(state: RetrievalState) -> dict:
        hits = await asyncio.to_thread(
            search_paragraphs,
            client,
            state["query"],
            top_k=state["paragraph_k"],
            character_filter=state.get("character_filter"),
        )
        return {"paragraph_dense": hits}

    return node


def _chapter_dense_node(client: QdrantClient):
    async def node(state: RetrievalState) -> dict:
        hits = await asyncio.to_thread(
            search_chapters,
            client,
            state["query"],
            top_k=state["chapter_k"],
            character_filter=state.get("character_filter"),
        )
        return {"chapter_dense": hits}

    return node


def _rewrite_node(state: RetrievalState) -> dict:
    """Fuzzy-correct typos in the query against the corpus vocabulary.

    Pure, deterministic, ~milliseconds. Updates `query` to the corrected
    form (so all downstream retrievers see the fixed version) and stores
    the original alongside for display.
    """
    original = state["query"]
    corrected, corrections = correct_query(original)
    return {
        "original_query": original,
        "query": corrected,
        "corrections": corrections,
    }


async def _paragraph_bm25_node(state: RetrievalState) -> dict:
    hits = await asyncio.to_thread(
        search_paragraphs_bm25,
        state["query"],
        top_k=state["paragraph_k"],
        character_filter=state.get("character_filter"),
    )
    return {"paragraph_bm25": hits}


async def _chapter_bm25_node(state: RetrievalState) -> dict:
    hits = await asyncio.to_thread(
        search_chapters_bm25,
        state["query"],
        top_k=state["chapter_k"],
        character_filter=state.get("character_filter"),
    )
    return {"chapter_bm25": hits}


def _fuse_node(state: RetrievalState) -> dict:
    """RRF-merge the four ranked lists and truncate to top_k."""
    ranked_lists = [
        state.get("paragraph_dense", []),
        state.get("chapter_dense", []),
        state.get("paragraph_bm25", []),
        state.get("chapter_bm25", []),
    ]
    fused = reciprocal_rank_fusion(ranked_lists, top_k=state["top_k"])
    return {"fused": fused}


# ── Graph construction ───────────────────────────────────────────


def build_retrieval_graph(client: QdrantClient) -> CompiledStateGraph:
    """Wire and compile the retrieval graph for a given Qdrant client.

    The compiled graph is reusable across queries; only the state
    (query / filters / k-knobs) changes. See `_compiled_graph` for the
    process-level cache used by `hybrid_search`.
    """
    graph = StateGraph(RetrievalState)

    graph.add_node("rewrite", _rewrite_node)
    graph.add_node("paragraph_dense", _paragraph_dense_node(client))
    graph.add_node("chapter_dense", _chapter_dense_node(client))
    graph.add_node("paragraph_bm25", _paragraph_bm25_node)
    graph.add_node("chapter_bm25", _chapter_bm25_node)
    graph.add_node("fuse", _fuse_node)

    # rewrite runs first; fan-out from there to all four retrievers.
    graph.add_edge(START, "rewrite")
    for name in ("paragraph_dense", "chapter_dense", "paragraph_bm25", "chapter_bm25"):
        graph.add_edge("rewrite", name)
        graph.add_edge(name, "fuse")
    graph.add_edge("fuse", END)

    return graph.compile()


@lru_cache(maxsize=1)
def _compiled_graph() -> CompiledStateGraph:
    """Build-once cache. Uses `get_client()` so settings.qdrant flows
    through. Tests that need a different client should call
    `build_retrieval_graph(client)` directly.
    """
    return build_retrieval_graph(get_client())


# ── Public API ───────────────────────────────────────────────────


@dataclass(slots=True)
class RetrievalResult:
    """Output of `hybrid_search`. Carries the fused candidates plus the
    rewriter's view of what happened to the query.

    Why a dataclass and not a bare list: the rewriter's corrections are
    user-facing context (CLI shows "did you mean…"). Returning a list
    would force every caller to either re-run the rewriter or peek into
    graph state. A small typed result keeps the API self-describing.
    """

    candidates: list[ScoredCandidate]
    query: str
    """The query as the retrievers actually saw it (post-rewrite)."""
    original_query: str
    """The query as the user typed it (pre-rewrite)."""
    corrections: list[tuple[str, str]] = field(default_factory=list)
    """`(typo, replacement)` pairs the rewriter applied. Empty if no
    corrections were needed."""

    def __iter__(self):
        """Iterate over candidates so existing callers that did
        `for c in hybrid_search(...)` keep working."""
        return iter(self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def __bool__(self) -> bool:
        return bool(self.candidates)


async def hybrid_search(  # noqa: PLR0913 — query + three k-knobs + filter are all distinct concerns
    client: QdrantClient | None = None,
    query: str = "",
    *,
    paragraph_k: int = 20,
    chapter_k: int = 5,
    top_k: int = 10,
    character_filter: list[str] | None = None,
) -> RetrievalResult:
    """Retrieve from all four sources in parallel and fuse via RRF.

    `client` is accepted for tests and runtime override; if omitted, the
    cached process-level compiled graph is used.

    Defaults are skewed toward paragraphs (paragraph_k=20, chapter_k=5)
    because paragraphs carry the precise evidence; chapters contribute
    breadth via the RRF bonus when they overlap on topic.
    """
    if not query:
        raise ValueError("hybrid_search requires a non-empty query")

    graph = build_retrieval_graph(client) if client is not None else _compiled_graph()
    state: RetrievalState = {
        "query": query,
        "character_filter": character_filter,
        "paragraph_k": paragraph_k,
        "chapter_k": chapter_k,
        "top_k": top_k,
    }
    final = await graph.ainvoke(state)
    return RetrievalResult(
        candidates=final["fused"],
        query=final.get("query", query),
        original_query=final.get("original_query", query),
        corrections=final.get("corrections", []),
    )
