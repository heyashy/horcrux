"""Retrieval over the two Qdrant collections.

Built up incrementally — at this stage we only have paragraph search.
Chapter search and RRF fusion land in the next steps; keeping the surface
small lets us verify each layer against live Qdrant before composing.

Design choices worth flagging:

1. **Query encoding goes through `encode_query`.** bge-large-en-v1.5 is
   asymmetric — passages were embedded with no prefix, queries get the
   "Represent this sentence…" prefix. Mixing the two paths silently
   degrades retrieval quality without any error, so the prefix lives
   behind a typed wrapper that callers can't bypass.

2. **`character_filter` is explicit, not auto-derived.** The eventual
   query pipeline (Phase 5) will run NER on the query and resolve to
   slug IDs. For now we keep the retrieval layer pure: take a list of
   slugs, filter on the indexed `characters` payload field, return.
   Easier to test, easier to debug.

3. **Returns `ScoredCandidate`, not raw Qdrant points.** The model is
   the contract between retrieval and downstream consumers (synthesis
   agent, Rich renderer). Everything pulled out of Qdrant gets validated
   on the way out.
"""

import asyncio
from typing import Literal

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny

from horcrux.config import settings
from horcrux.embedding import encode_query
from horcrux.models import ScoredCandidate

_Source = Literal["chapter", "paragraph"]

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


async def hybrid_search(  # noqa: PLR0913 — three independent k-knobs + filter
    client: QdrantClient,
    query: str,
    *,
    paragraph_k: int = 20,
    chapter_k: int = 5,
    top_k: int = 10,
    character_filter: list[str] | None = None,
) -> list[ScoredCandidate]:
    """Retrieve from both collections in parallel and fuse via RRF.

    Defaults are skewed: pull more paragraphs (20) than chapters (5)
    because paragraphs are where the precise evidence lives. Chapters
    contribute breadth — they get into the fused result via the RRF
    bonus when they happen to share characters / topic with paragraphs
    that already ranked.

    Runs as a coroutine because the long-term plan is to dispatch other
    parallel work alongside it (e.g. BM25 sparse retrieval, query
    rewriting). For now it's two awaits in `asyncio.gather`; the shape
    is what matters.
    """
    paragraphs, chapters = await asyncio.gather(
        asyncio.to_thread(
            search_paragraphs,
            client,
            query,
            top_k=paragraph_k,
            character_filter=character_filter,
        ),
        asyncio.to_thread(
            search_chapters,
            client,
            query,
            top_k=chapter_k,
            character_filter=character_filter,
        ),
    )
    return reciprocal_rank_fusion([paragraphs, chapters], top_k=top_k)
