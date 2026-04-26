"""Query rewriter — fuzzy keyword correction against the corpus vocabulary.

Solves the typo case (Finding 22 candidate): a query like *"what is the
name of the conjuncatvitus spell?"* contains a typo'd token that exists
nowhere in the corpus. BM25 needs exact-token match; dense embeddings
recover partially via subword tokenisation but not reliably for
significant misspellings. Both fail.

Fix at the input layer: tokenise the query, identify any token that's
not in the corpus vocabulary, fuzzy-match it against the vocab, and
substitute the closest in-vocab token if the match is good enough.

Done before retrieval, deterministic, no LLM call. The rewriter doesn't
*remove* tokens (no stopword stripping) — leaves the natural-language
shape intact for the dense path while making rare-keyword tokens
recoverable for BM25.

Vocab is built once per process from `chunks.json` and cached. ~50k
unique tokens for our corpus; rapidfuzz scans them in milliseconds per
query token.
"""

from collections.abc import Iterable
from functools import lru_cache

from rapidfuzz import fuzz, process

from horcrux.retrieval.bm25 import _CHUNKS_PATH, _load_chunks, _tokenize

# Tokens shorter than this aren't fuzzy-corrected. Short tokens have too
# many false-positive neighbours in vocab ("the" / "they" / "them" all
# fuzz-match closely); the risk of misdirection is greater than the
# benefit. Real-world typos that matter for retrieval are on content
# words, which are >= 5 chars.
_MIN_CORRECTABLE_LENGTH = 5

# Fuzzy match score threshold (0-100). Below this the candidate is too
# distant and we leave the original token alone — better to under-correct
# than misdirect retrieval onto an unrelated word.
_FUZZY_THRESHOLD = 80


@lru_cache(maxsize=2)
def _corpus_vocab(chunks_path: str = str(_CHUNKS_PATH)) -> tuple[str, ...]:
    """All distinct tokens in the corpus, lowercased, ordered by first
    appearance. Tuple so `lru_cache` can hold it; rapidfuzz accepts any
    iterable.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk in _load_chunks(chunks_path):
        for tok in _tokenize(chunk.text):
            if tok not in seen:
                seen.add(tok)
                ordered.append(tok)
    return tuple(ordered)


def correct_query(
    query: str,
    *,
    threshold: int = _FUZZY_THRESHOLD,
    min_length: int = _MIN_CORRECTABLE_LENGTH,
    vocab: Iterable[str] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Rewrite `query` by replacing OOV tokens with their nearest in-vocab
    fuzzy matches.

    Returns `(rewritten_query, corrections)` where `corrections` is a list
    of `(original_token, replacement_token)` pairs in encounter order —
    suitable for a "did you mean…" note in the UI.

    `vocab` defaults to the cached corpus vocabulary built from
    `chunks.json`; tests can pass a smaller fixed vocab to keep the
    behaviour deterministic and not coupled to the gold-tier artefact.
    """
    pool: tuple[str, ...] = tuple(vocab) if vocab is not None else _corpus_vocab()
    pool_set = set(pool)
    tokens = _tokenize(query)

    corrections: list[tuple[str, str]] = []
    rewritten: list[str] = []

    for tok in tokens:
        if tok in pool_set or len(tok) < min_length:
            rewritten.append(tok)
            continue
        match = process.extractOne(
            tok, pool, scorer=fuzz.ratio, score_cutoff=threshold
        )
        if match is None:
            # No close enough vocab neighbour — leave the token alone.
            # Retrieval will then naturally produce zero hits and the
            # synthesis layer will return conviction 1/5 with a gap, which
            # is the right user-facing failure mode.
            rewritten.append(tok)
            continue
        replacement = match[0]
        rewritten.append(replacement)
        corrections.append((tok, replacement))

    return " ".join(rewritten), corrections
