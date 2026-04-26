"""Research-mode tests — graph nodes, candidate merging, citation
resolution. The agents themselves (planner, aggregator) are mocked via
PydanticAI's TestModel; we don't hit the proxy.
"""

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from horcrux import aggregator as aggregator_mod
from horcrux import planner as planner_mod
from horcrux.aggregator import _merge_candidates, aggregate_subfindings
from horcrux.models import (
    Finding,
    Plan,
    ResearchReport,
    ScoredCandidate,
    SubFinding,
)
from horcrux.planner import plan_query

pytestmark = pytest.mark.unit


def _candidate(cid: str = "id-a") -> ScoredCandidate:
    return ScoredCandidate(
        id=cid,
        score=0.5,
        source="paragraph",
        text=f"text-{cid}",
        book_num=1,
        chapter_num=1,
        chapter_title="t",
        page_start=1,
        characters=[],
    )


def _finding(answer: str = "x", source_ids: list[str] | None = None) -> Finding:
    return Finding(answer=answer, source_ids=source_ids or ["id-a"], conviction=4)


def _sub(
    sub_question: str = "q",
    candidates: list[ScoredCandidate] | None = None,
    source_ids: list[str] | None = None,
) -> SubFinding:
    cands = candidates if candidates is not None else [_candidate()]
    return SubFinding(
        sub_question=sub_question,
        finding=_finding(source_ids=source_ids),
        candidates=cands,
    )


# ── Plan / ResearchReport schema ─────────────────────────────────


def test_plan_rejects_empty_sub_questions():
    with pytest.raises(ValidationError):
        Plan(sub_questions=[])


def test_plan_rejects_too_many_sub_questions():
    """Cap at 8. Real usage targets 3-5; the schema enforces a hard
    upper bound to prevent the planner exploding the workload."""
    with pytest.raises(ValidationError):
        Plan(sub_questions=[f"q{i}" for i in range(9)])


def test_research_report_rejects_empty_source_ids():
    plan = Plan(sub_questions=["q"])
    with pytest.raises(ValidationError):
        ResearchReport(
            query="q",
            original_query="q",
            plan=plan,
            sub_findings=[_sub()],
            answer="x",
            source_ids=[],
            conviction=3,
        )


# ── _merge_candidates ────────────────────────────────────────────


def test_merge_candidates_dedupes_by_id():
    """Two sub-findings retrieved the same chunk — appears once in the
    aggregator's view."""
    shared = _candidate("shared")
    sf1 = _sub("q1", candidates=[shared, _candidate("a")])
    sf2 = _sub("q2", candidates=[_candidate("b"), shared])
    merged = _merge_candidates([sf1, sf2])
    assert [c.id for c in merged] == ["shared", "a", "b"]


def test_merge_candidates_preserves_order():
    """First-appearance order — passage numbers are stable across
    aggregations."""
    sf1 = _sub("q1", candidates=[_candidate("z"), _candidate("y")])
    sf2 = _sub("q2", candidates=[_candidate("x")])
    merged = _merge_candidates([sf1, sf2])
    assert [c.id for c in merged] == ["z", "y", "x"]


def test_merge_candidates_handles_empty_subfinding():
    sf1 = _sub("q1", candidates=[])
    sf2 = _sub("q2", candidates=[_candidate("a")])
    merged = _merge_candidates([sf1, sf2])
    assert [c.id for c in merged] == ["a"]


# ── plan_query ───────────────────────────────────────────────────


async def test_plan_query_rejects_empty():
    with pytest.raises(ValueError, match="non-empty query"):
        await plan_query("")


async def test_plan_query_returns_typed_plan():
    """Mock the planner agent — verify the wrapper produces a Plan."""
    fake_plan = Plan(
        sub_questions=["q1", "q2"], rationale="because"
    )
    test_model = TestModel(custom_output_args=fake_plan)
    agent = planner_mod._planner_agent()
    with agent.override(model=test_model):
        plan = await plan_query("any research question")
    assert plan.sub_questions == ["q1", "q2"]


# ── aggregate_subfindings ────────────────────────────────────────


async def test_aggregate_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one sub-finding"):
        await aggregate_subfindings("q", [])


async def test_aggregate_rejects_no_candidates():
    """Sub-findings that all came up empty — aggregator raises rather
    than calling the LLM with no passages to ground from."""
    sf = _sub("q", candidates=[])
    with pytest.raises(ValueError, match="no candidates"):
        await aggregate_subfindings("q", [sf])


async def test_aggregate_translates_passage_numbers_to_ids():
    """Aggregator returns numbered citations; wrapper resolves them
    against the merged candidate set. Same runtime check as the
    single-shot synthesis layer."""
    sub_findings = [
        _sub("q1", candidates=[_candidate("id-1"), _candidate("id-2")]),
        _sub("q2", candidates=[_candidate("id-3")]),
    ]
    fake_finding = Finding(
        answer="merged answer",
        source_ids=["1", "3"],  # numbered citations
        conviction=4,
    )
    test_model = TestModel(custom_output_args=fake_finding)

    agent = aggregator_mod._aggregator_agent()
    with agent.override(model=test_model):
        out = await aggregate_subfindings("q", sub_findings)

    # 1 → id-1, 3 → id-3 in the merged candidate order.
    assert out.source_ids == ["id-1", "id-3"]


async def test_aggregate_rejects_out_of_range_citation():
    """Runtime check: aggregator can't cite a passage number that
    doesn't exist in the merged set. Strict-RAG at the aggregate layer."""
    sf = _sub("q", candidates=[_candidate("id-1")])
    fake_finding = Finding(
        answer="x",
        source_ids=["99"],  # only 1 candidate in merged set
        conviction=4,
    )
    test_model = TestModel(custom_output_args=fake_finding)

    agent = aggregator_mod._aggregator_agent()
    with agent.override(model=test_model):  # noqa: SIM117
        with pytest.raises(ValueError, match="not in"):
            await aggregate_subfindings("q", [sf])
