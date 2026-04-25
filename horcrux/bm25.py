"""In-memory BM25 retrieval over the chunk corpus.

The corpus is small enough (~17MB raw text, ~5,500 chunks) that an
in-memory BM25 index fits comfortably in RAM and answers queries in
sub-millisecond time. This avoids the operational overhead of
provisioning sparse vectors in Qdrant — and is genuinely the right
call at this scale (see ADR-0008 for the production-equivalent
discussion).

Two indexes, one per chunk type, mirroring the dense Qdrant collections:

  - paragraphs: ~5,400 points → fine-grained keyword retrieval.
  - chapters:   ~200 points   → broad keyword retrieval.

Both are loaded from `data/processed/chunks.json` lazily on first call
and cached for the process lifetime via `lru_cache`. The caches are
keyed on the file path so a different chunks file (e.g. for tests) gets
its own index.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from rank_bm25 import BM25Okapi

from horcrux.models import ChapterChunk, ScoredCandidate

_CHUNKS_PATH = Path("data/processed/chunks.json")

# Tokenizer must produce identical output at index time and query time.
# Lowercased word characters; everything else is a separator. Drops
# punctuation, OCR artefacts, em-dashes — any token-defining ambiguity.
_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    """Lowercase + word-character tokenize. The shared corpus + query path."""
    return _TOKEN_RE.findall(text.lower())


@dataclass(slots=True)
class BM25Index:
    """A built BM25 index over a slice of chunks (one chunk_type).

    `chunks` is held alongside the BM25 model so we can reconstruct
    `ScoredCandidate`s from the rank output. Score is the raw BM25 value;
    callers should treat it as ordinal (rank-only) rather than as a
    normalised probability — see `reciprocal_rank_fusion` in retrieval.py
    for why ranks compose better than scores across heterogeneous lists.
    """

    chunks: list[ChapterChunk]
    bm25: BM25Okapi

    @classmethod
    def build(cls, chunks: list[ChapterChunk]) -> "BM25Index":
        if not chunks:
            raise ValueError("BM25Index needs at least one chunk")
        tokenized = [_tokenize(c.text) for c in chunks]
        return cls(chunks=chunks, bm25=BM25Okapi(tokenized))

    def search(
        self,
        query: str,
        *,
        top_k: int,
        source: Literal["chapter", "paragraph"],
    ) -> list[ScoredCandidate]:
        """Return the top_k chunks for a tokenized query, as ScoredCandidate.

        `source` is plumbed through onto each candidate so the synthesis
        layer (and RRF) can tell where the hit came from.
        """
        scores = self.bm25.get_scores(_tokenize(query))
        # Argsort descending; take top_k. numpy would be marginally faster
        # but rank-bm25 already returns numpy under the hood — the bottleneck
        # is `get_scores`, not the sort.
        ordered = sorted(
            range(len(self.chunks)),
            key=lambda i: -scores[i],
        )[:top_k]
        return [
            ScoredCandidate(
                id=self.chunks[i].id,
                score=float(scores[i]),
                source=source,
                text=self.chunks[i].text,
                book_num=self.chunks[i].book_num,
                chapter_num=self.chunks[i].chapter_num,
                chapter_title=self.chunks[i].chapter_title,
                page_start=self.chunks[i].page_start,
                characters=self.chunks[i].characters,
            )
            for i in ordered
            if scores[i] > 0  # zero score = no token overlap; skip
        ]


@lru_cache(maxsize=4)
def _load_chunks(path_str: str) -> list[ChapterChunk]:
    """Read chunks.json once per path and reuse. Paths are stringified for
    `lru_cache` (Path is hashable but pathstrs are clearer in tracebacks).
    """
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `make chunks` to build the gold tier"
        )
    return [ChapterChunk(**c) for c in json.loads(path.read_text())]


@lru_cache(maxsize=2)
def get_paragraph_index(chunks_path: str = str(_CHUNKS_PATH)) -> BM25Index:
    """Lazy-loaded paragraph BM25 index. Cached for the process lifetime."""
    chunks = [c for c in _load_chunks(chunks_path) if c.chunk_type == "paragraph"]
    return BM25Index.build(chunks)


@lru_cache(maxsize=2)
def get_chapter_index(chunks_path: str = str(_CHUNKS_PATH)) -> BM25Index:
    """Lazy-loaded chapter BM25 index. Cached for the process lifetime."""
    chunks = [c for c in _load_chunks(chunks_path) if c.chunk_type == "chapter"]
    return BM25Index.build(chunks)
