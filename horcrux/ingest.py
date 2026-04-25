"""Qdrant collection setup and chunk upsert.

Two collections, one per chunk_type:
  - hp_chapters    — one point per chapter, for broad analytical queries
  - hp_paragraphs  — semantic sub-sections, for factual / scene-level queries

Both use cosine distance over 1024-dim bge-large vectors. Splitting by
collection (rather than mixing and filtering) keeps the working set small
on the chapter side (~200 points) and lets the two be tuned independently
later.

Idempotent by construction: chunk IDs are deterministic UUID5s
(`models.make_chunk_id`), so re-running the embed → upsert pipeline
overwrites in place rather than duplicating.
"""

from collections.abc import Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from horcrux.config import settings
from horcrux.models import ChapterChunk

# Payload fields we filter on at query time. Without these indexes Qdrant
# falls back to a full payload scan — fine for 5k points, slow at 5M.
_INDEXED_FIELDS: dict[str, PayloadSchemaType] = {
    "book_num": PayloadSchemaType.INTEGER,
    "chapter_num": PayloadSchemaType.INTEGER,
    "characters": PayloadSchemaType.KEYWORD,
}


def get_client() -> QdrantClient:
    """Connect to Qdrant via gRPC (port 6334) — REST is for the dashboard."""
    return QdrantClient(
        host=settings.qdrant.host,
        port=settings.qdrant.port,
        prefer_grpc=True,
    )


def ensure_collection(client: QdrantClient, name: str, *, dim: int) -> None:
    """Create the collection if missing, plus payload indexes. Idempotent."""
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    for field, schema in _INDEXED_FIELDS.items():
        # create_payload_index is idempotent — re-creating an existing index
        # is a no-op on the server side.
        client.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=schema,
        )


def _to_point(chunk: ChapterChunk, vector: np.ndarray) -> PointStruct:
    return PointStruct(
        id=chunk.id,
        vector=vector.tolist(),
        payload={
            "book_num": chunk.book_num,
            "chapter_num": chunk.chapter_num,
            "chapter_title": chunk.chapter_title,
            "text": chunk.text,
            "chunk_type": chunk.chunk_type,
            "characters": chunk.characters,
            "page_start": chunk.page_start,
        },
    )


def upsert_chunks(
    client: QdrantClient,
    collection: str,
    chunks: Iterable[ChapterChunk],
    vectors: np.ndarray,
    *,
    batch_size: int = 128,
) -> int:
    """Upsert chunks + vectors into the named collection in batches.

    Returns the count of points written. Caller is responsible for slicing
    chunks/vectors to a single chunk_type before calling.
    """
    chunks = list(chunks)
    if len(chunks) != len(vectors):
        raise ValueError(
            f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
        )

    written = 0
    for start in range(0, len(chunks), batch_size):
        end = start + batch_size
        points = [
            _to_point(c, v)
            for c, v in zip(chunks[start:end], vectors[start:end], strict=True)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        written += len(points)
    return written
