"""Retrieval-graph tests — exercise the fuse node + bm25 character
post-filter without standing up Qdrant or BM25 indexes.
"""

import pytest

from horcrux.models import ScoredCandidate
from horcrux.retrieval import _filter_by_characters
from horcrux.retrieval_graph import _fuse_node

pytestmark = pytest.mark.unit


def _cand(
    cid: str,
    *,
    source: str = "paragraph",
    score: float = 0.0,
    characters: list[str] | None = None,
) -> ScoredCandidate:
    return ScoredCandidate(
        id=cid,
        score=score,
        source=source,
        text=f"text-{cid}",
        book_num=1,
        chapter_num=1,
        chapter_title="t",
        page_start=1,
        characters=characters or [],
    )


# ── _filter_by_characters (BM25 post-pass) ──────────────────────


def test_filter_by_characters_none_filter_passes_all():
    cands = [_cand("a"), _cand("b")]
    out = _filter_by_characters(cands, None, top_k=10)
    assert [c.id for c in out] == ["a", "b"]


def test_filter_by_characters_empty_filter_passes_all():
    cands = [_cand("a"), _cand("b")]
    out = _filter_by_characters(cands, [], top_k=10)
    assert [c.id for c in out] == ["a", "b"]


def test_filter_by_characters_drops_non_matching():
    cands = [
        _cand("a", characters=["harry_potter"]),
        _cand("b", characters=["voldemort"]),
        _cand("c", characters=["harry_potter", "hermione_granger"]),
    ]
    out = _filter_by_characters(cands, ["harry_potter"], top_k=10)
    assert [c.id for c in out] == ["a", "c"]


def test_filter_by_characters_truncates_to_top_k():
    cands = [_cand(str(i), characters=["x"]) for i in range(10)]
    out = _filter_by_characters(cands, ["x"], top_k=3)
    assert len(out) == 3


def test_filter_by_characters_returns_empty_when_no_matches():
    cands = [_cand("a", characters=["voldemort"])]
    out = _filter_by_characters(cands, ["harry_potter"], top_k=10)
    assert out == []


# ── _fuse_node ──────────────────────────────────────────────────


def test_fuse_node_merges_four_lists():
    state = {
        "paragraph_dense": [_cand("a"), _cand("b")],
        "chapter_dense": [_cand("c"), _cand("a")],
        "paragraph_bm25": [_cand("d")],
        "chapter_bm25": [_cand("a")],
        "top_k": 10,
    }
    out = _fuse_node(state)["fused"]
    # `a` appears in 3 of 4 lists at varying ranks; should rank first.
    assert out[0].id == "a"
    # All four IDs should appear in the merged result.
    assert {c.id for c in out} == {"a", "b", "c", "d"}


def test_fuse_node_handles_missing_lists():
    """If a retriever node hasn't run / returned nothing, fuse should still
    produce a valid result from the lists that did populate."""
    state = {
        "paragraph_dense": [_cand("a"), _cand("b")],
        "top_k": 5,
        # chapter_dense, paragraph_bm25, chapter_bm25 absent
    }
    out = _fuse_node(state)["fused"]
    assert [c.id for c in out] == ["a", "b"]


def test_fuse_node_respects_top_k():
    state = {
        "paragraph_dense": [_cand(f"p{i}") for i in range(20)],
        "chapter_dense": [],
        "paragraph_bm25": [],
        "chapter_bm25": [],
        "top_k": 3,
    }
    out = _fuse_node(state)["fused"]
    assert len(out) == 3


def test_fuse_node_overwrites_score_with_rrf():
    """Original retriever scores (cosine, BM25) should not survive the
    fusion — `score` becomes the RRF score for downstream consumers."""
    state = {
        "paragraph_dense": [_cand("a", score=0.99)],
        "chapter_dense": [],
        "paragraph_bm25": [],
        "chapter_bm25": [],
        "top_k": 1,
    }
    out = _fuse_node(state)["fused"]
    # 1 / (60 + 1)
    assert out[0].score == pytest.approx(1.0 / 61)
