"""Retrieval-module tests. RRF and filter-builder are pure — testable
without Qdrant. The search functions themselves are integration-tested
elsewhere.
"""

import pytest
from qdrant_client.models import Filter

from horcrux.models import ScoredCandidate
from horcrux.retrieval import _build_character_filter, reciprocal_rank_fusion

pytestmark = pytest.mark.unit


def _cand(cid: str, source: str = "paragraph", score: float = 0.0) -> ScoredCandidate:
    """Minimal ScoredCandidate factory for fusion tests."""
    return ScoredCandidate(
        id=cid,
        score=score,
        source=source,
        text=f"text-{cid}",
        book_num=1,
        chapter_num=1,
        chapter_title="t",
        page_start=1,
        characters=[],
    )


# ── _build_character_filter ────────────────────────────────────────


def test_build_character_filter_none_for_empty():
    assert _build_character_filter(None) is None
    assert _build_character_filter([]) is None


def test_build_character_filter_returns_filter_for_ids():
    f = _build_character_filter(["hermione_granger", "ron_weasley"])
    assert isinstance(f, Filter)


# ── reciprocal_rank_fusion ─────────────────────────────────────────


def test_rrf_single_list_preserves_order():
    """One list in, ranks in same order out."""
    ranked = [_cand("a"), _cand("b"), _cand("c")]
    fused = reciprocal_rank_fusion([ranked])
    assert [c.id for c in fused] == ["a", "b", "c"]


def test_rrf_overlap_promotes_shared_items():
    """Items in both lists should rank above items in only one."""
    list_1 = [_cand("a"), _cand("b"), _cand("c")]
    list_2 = [_cand("c"), _cand("a"), _cand("d")]
    fused = reciprocal_rank_fusion([list_1, list_2])
    # `a` is rank 1 + rank 2 → 1/61 + 1/62, very strong
    # `c` is rank 3 + rank 1 → 1/63 + 1/61, also strong
    # `b`, `d` are single-list contributions → weakest
    fused_ids = [c.id for c in fused]
    assert fused_ids[0] in ("a", "c")
    assert set(fused_ids[:2]) == {"a", "c"}


def test_rrf_dedups_by_id():
    list_1 = [_cand("a"), _cand("a")]
    list_2 = [_cand("a")]
    fused = reciprocal_rank_fusion([list_1, list_2])
    assert len(fused) == 1
    assert fused[0].id == "a"


def test_rrf_score_overwritten_with_rrf_value():
    """The cosine score on the input is replaced by the RRF score."""
    ranked = [_cand("a", score=0.99)]
    fused = reciprocal_rank_fusion([ranked])
    # 1 / (60 + 1)
    assert fused[0].score == pytest.approx(1.0 / 61)


def test_rrf_top_k_truncates():
    list_1 = [_cand(str(i)) for i in range(20)]
    fused = reciprocal_rank_fusion([list_1], top_k=5)
    assert len(fused) == 5


def test_rrf_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []
