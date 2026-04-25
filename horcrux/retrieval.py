"""Retrieval primitives — dense ANN over Qdrant, BM25 over an in-memory
index, and Reciprocal Rank Fusion over their outputs.

Composition (the `hybrid_search` four-way fusion) lives in
`retrieval_graph.py` as a LangGraph state machine. This module holds the
leaf functions: each one produces a ranked list of `ScoredCandidate`
from a single retriever (one collection x one modality).

Design choices worth flagging:

1. **Query encoding goes through `encode_query`.** bge-large-en-v1.5 is
   asymmetric — passages were embedded with no prefix, queries get the
   "Represent this sentence…" prefix. Mixing the two paths silently
   degrades retrieval quality without any error, so the prefix lives
   behind a typed wrapper that callers can't bypass.

2. **`character_filter` is explicit, not auto-derived.** The eventual
   query pipeline (next phase) will run NER on the query and resolve
   to slug IDs. For now retrieval stays pure: take a list of slugs,
   apply, return.

3. **BM25 character filter is a post-pass, not a pre-pass.** Pre-filtering
   would mean rebuilding the BM25 index per query (because IDF depends
   on the corpus). Post-filtering means we pull a generous top_k and
   drop hits that don't carry any of the requested characters. Cheap;
   the only cost is over-fetching.

4. **Returns `ScoredCandidate`, not raw retriever outputs.** The model
   is the contract between retrieval and downstream consumers
   (synthesis agent, Rich renderer, RRF). Everything gets validated on
   the way out.
"""

from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from horcrux.bm25 import BM25Index, get_chapter_index, get_paragraph_index
from horcrux.config import settings
from horcrux.embedding import encode_query
from horcrux.models import ScoredCandidate

_Source = Literal["chapter", "paragraph"]

# When BM25 character-filtering is requested, pull this multiple of top_k
# from the index so the post-filter still has enough hits left after
# dropping non-matching candidates. Tuned empirically for our corpus
# (5,377 paragraphs, character tags on 99% of chunks). If a queried
# character is rare, raise this — but don't make it unbounded; we
# don't want to scan the whole index for every query.
_BM25_OVERFETCH_FACTOR = 4

# RRF constant. 60 is the original Cormack-et-al value and remains the
# defensible default — small k weights top ranks more heavily, large k
# flattens the contribution. Don't tune unless you have a real eval set.
_RRF_K = 60


def _build_character_filter(character_ids: list[str] | None) -> Filter | None:
    """Build a Qdrant `match_any` filter on the indexed `characters` field.

    Returns None for empty / missing input — Qdrant treats `Filter()` and
    `None` differently; passing `None` skips filtering entirely.
    """
    if not character_ids:
        return None
    return Filter(
        must=[
            FieldCondition(
                key="characters",
                match=MatchAny(any=character_ids),
            )
        ]
    )


def _ann_search(  # noqa: PLR0913 — collection / source / k / filter are all distinct concerns
    client: QdrantClient,
    collection: str,
    query: str,
    source: _Source,
    *,
    top_k: int,
    character_filter: list[str] | None,
) -> list[ScoredCandidate]:
    """Shared ANN-search core: encode → query_points → ScoredCandidate."""
    vector = encode_query(query)
    response = client.query_points(
        collection_name=collection,
        query=vector.tolist(),
        query_filter=_build_character_filter(character_filter),
        limit=top_k,
        with_payload=True,
    )
    return [
        ScoredCandidate(
            id=str(hit.id),
            score=float(hit.score),
            source=source,
            text=hit.payload["text"],
            book_num=hit.payload["book_num"],
            chapter_num=hit.payload["chapter_num"],
            chapter_title=hit.payload["chapter_title"],
            page_start=hit.payload["page_start"],
            characters=hit.payload.get("characters", []),
        )
        for hit in response.points
    ]


def search_paragraphs(
    client: QdrantClient,
    query: str,
    *,
    top_k: int = 10,
    character_filter: list[str] | None = None,
) -> list[ScoredCandidate]:
    """ANN search over `hp_paragraphs` for factual / scene-level queries.

    Args:
        client: live Qdrant client (use `horcrux.ingest.get_client`).
        query: natural-language query. Encoded with the bge-large query
            prefix before search.
        top_k: number of hits to return.
        character_filter: optional list of character slug IDs. If set,
            only paragraphs tagged with at least one of these are
            returned — this is the Tier-3 filter that turns
            "everything Hermione said about house-elves" into a
            payload predicate.

    Returns:
        list of ScoredCandidate, ordered by descending cosine similarity.
        `source` is always "paragraph".
    """
    return _ann_search(
        client,
        settings.qdrant.paragraphs_collection,
        query,
        "paragraph",
        top_k=top_k,
        character_filter=character_filter,
    )


def search_chapters(
    client: QdrantClient,
    query: str,
    *,
    top_k: int = 5,
    character_filter: list[str] | None = None,
) -> list[ScoredCandidate]:
    """ANN search over `hp_chapters` for breadth / analytical queries.

    Caveat (Finding 17): chapter chunks are bge-large-embedded, which
    truncates input to 512 tokens. The point in `hp_chapters` represents
    the *opening* of the chapter, not the whole text. This is reasonable
    for routing-style queries (HP openings establish setting + characters)
    but not for fine-grained content lookups — that's what `search_paragraphs`
    is for.

    Top_k defaults to 5 because there are only 198 points; pulling 10
    consistently dilutes the result set with marginal hits.
    """
    return _ann_search(
        client,
        settings.qdrant.chapters_collection,
        query,
        "chapter",
        top_k=top_k,
        character_filter=character_filter,
    )


# ── BM25 search ───────────────────────────────────────────────────


def _filter_by_characters(
    candidates: list[ScoredCandidate],
    character_ids: list[str] | None,
    *,
    top_k: int,
) -> list[ScoredCandidate]:
    """Post-pass character filter for BM25 results.

    Dense search filters server-side via Qdrant's payload index; BM25
    runs against the full in-memory corpus and filters here instead.
    Same observable semantics: a hit is kept iff its payload `characters`
    set intersects `character_ids`.
    """
    if not character_ids:
        return candidates[:top_k]
    requested = set(character_ids)
    return [c for c in candidates if requested.intersection(c.characters)][:top_k]


def _bm25_search(
    index: BM25Index,
    query: str,
    source: _Source,
    *,
    top_k: int,
    character_filter: list[str] | None,
) -> list[ScoredCandidate]:
    """Shared BM25 entry: over-fetch when filtering, then post-filter."""
    fetch_k = top_k * _BM25_OVERFETCH_FACTOR if character_filter else top_k
    raw = index.search(query, top_k=fetch_k, source=source)
    return _filter_by_characters(raw, character_filter, top_k=top_k)


def search_paragraphs_bm25(
    query: str,
    *,
    top_k: int = 10,
    character_filter: list[str] | None = None,
) -> list[ScoredCandidate]:
    """BM25 search over the in-memory paragraph index.

    Counterpart to `search_paragraphs` — same return shape, different
    retriever. Solves the rare-keyword case dense embeddings miss
    (Finding 21).
    """
    return _bm25_search(
        get_paragraph_index(),
        query,
        "paragraph",
        top_k=top_k,
        character_filter=character_filter,
    )


def search_chapters_bm25(
    query: str,
    *,
    top_k: int = 5,
    character_filter: list[str] | None = None,
) -> list[ScoredCandidate]:
    """BM25 search over the in-memory chapter index. See
    `search_paragraphs_bm25` for the rationale.
    """
    return _bm25_search(
        get_chapter_index(),
        query,
        "chapter",
        top_k=top_k,
        character_filter=character_filter,
    )


# ── Fusion ────────────────────────────────────────────────────────


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredCandidate]],
    *,
    k: int = _RRF_K,
    top_k: int = 10,
) -> list[ScoredCandidate]:
    """Merge ranked candidate lists by Reciprocal Rank Fusion.

    For each candidate, RRF score = Σ 1/(k + rank_in_list_i). Items
    appearing in multiple lists accumulate; items at top ranks contribute
    more. Crucially **rank-based, not score-based** — it doesn't matter
    that paragraph cosines and chapter cosines come from different
    distributions, only that each list is ordered.

    De-duplication is by `id`. When a candidate appears in two lists we
    keep the first instance's payload (chapter and paragraph payloads
    have the same shape so this is safe), and the *first* `source` value
    we saw — the source label loses meaning post-fusion, but we preserve
    something for debugging.

    Returns the top_k merged candidates with `score` overwritten by the
    RRF score.
    """
    rrf_scores: dict[str, float] = {}
    representatives: dict[str, ScoredCandidate] = {}

    for ranked in ranked_lists:
        for rank, cand in enumerate(ranked, start=1):
            rrf_scores[cand.id] = rrf_scores.get(cand.id, 0.0) + 1.0 / (k + rank)
            representatives.setdefault(cand.id, cand)

    fused: list[ScoredCandidate] = []
    for cid, rrf in sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True):
        rep = representatives[cid]
        fused.append(rep.model_copy(update={"score": rrf}))

    return fused[:top_k]
