"""Sentence-transformer embedding for chunks.

Wraps bge-large-en-v1.5 from sentence-transformers. Bge-large is
asymmetric: passages (chunks) get NO prefix; queries get the prefix
configured in `settings.embedding.query_prefix`. We're embedding
passages here.

Lazily loaded — model load is ~1.3GB and several seconds; defer until
first call so import-time is fast.
"""

from collections.abc import Iterable

import numpy as np

from horcrux.config import settings
from horcrux.models import ChapterChunk

_MODEL = None


def _get_model():
    """Lazy-load bge-large; cache."""
    global _MODEL  # noqa: PLW0603 — module-level cache for the 1.3GB model
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415 — defer heavy import

        _MODEL = SentenceTransformer(settings.embedding.model_name)
    return _MODEL


def encode_passages(
    texts: Iterable[str],
    *,
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """Embed a sequence of passage texts. Returns L2-normalised (N, 1024) array.

    Passages get NO prefix (asymmetric model — query side adds prefix).
    """
    model = _get_model()
    return model.encode(
        list(texts),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )


def encode_query(query: str) -> np.ndarray:
    """Embed a query string. Returns L2-normalised (1024,) vector.

    Bge-large is asymmetric — queries are prefixed with the
    "Represent this sentence for searching relevant passages: " phrase.
    """
    model = _get_model()
    prefixed = settings.embedding.query_prefix + query
    return model.encode(
        prefixed,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def encode_chunks(
    chunks: list[ChapterChunk],
    *,
    batch_size: int = 32,
    show_progress: bool = True,
) -> np.ndarray:
    """Embed every chunk's `.text` and return the (N, 1024) matrix in
    chunk order. Convenience wrapper that pulls texts and dispatches to
    `encode_passages`.
    """
    texts = [c.text for c in chunks]
    return encode_passages(texts, batch_size=batch_size, show_progress=show_progress)
