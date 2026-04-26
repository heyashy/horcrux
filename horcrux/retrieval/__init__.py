"""Retrieval — finding relevant chunks for a given query.

The subpackage layers up:

    store.py         Qdrant client + collection setup + upsert.
    bm25.py          In-memory BM25 index over chunks.json.
    query_rewrite.py Fuzzy-correct typos against the corpus vocabulary.
    search.py        search_paragraphs / search_chapters (+ BM25 variants);
                     reciprocal_rank_fusion.
    graph.py         LangGraph state machine: rewrite → 4-way fan-out → fuse.
                     hybrid_search() is the public API for everything
                     downstream of retrieval.

Top-level re-exports below cover the surface most callers need; pull
from specific modules for less-common helpers.
"""

from horcrux.retrieval.bm25 import BM25Index, get_chapter_index, get_paragraph_index
from horcrux.retrieval.graph import (
    RetrievalResult,
    build_retrieval_graph,
    hybrid_search,
)
from horcrux.retrieval.query_rewrite import correct_query
from horcrux.retrieval.search import (
    reciprocal_rank_fusion,
    search_chapters,
    search_chapters_bm25,
    search_paragraphs,
    search_paragraphs_bm25,
)
from horcrux.retrieval.store import (
    ensure_collection,
    get_client,
    upsert_chunks,
)

__all__ = [
    "BM25Index",
    "RetrievalResult",
    "build_retrieval_graph",
    "correct_query",
    "ensure_collection",
    "get_chapter_index",
    "get_client",
    "get_paragraph_index",
    "hybrid_search",
    "reciprocal_rank_fusion",
    "search_chapters",
    "search_chapters_bm25",
    "search_paragraphs",
    "search_paragraphs_bm25",
    "upsert_chunks",
]
