"""Research-mode LangGraph — multi-step planning + parallel sub-query
execution + aggregation.

    START
      ↓
    plan ──Send──┬──> subquery (sub-q 1) ──┐
                 ├──> subquery (sub-q 2) ──┤
                 ├──> subquery (sub-q 3) ──┼──> aggregate ──> END
                 ├──> subquery (sub-q 4) ──┤
                 └──> subquery (sub-q 5) ──┘

`Send` (LangGraph's graph-level fan-out primitive) dispatches one
`subquery` invocation per sub-question. Each runs the existing
`hybrid_search` + `synthesise` pipeline independently. The fan-in to
`aggregate` waits for all branches to complete before reducing.

Why `Send` rather than `asyncio.gather` inside one node:
  - Each sub-query becomes a *separate* LangGraph node invocation, so
    `astream` yields a per-sub-query event the renderer can surface.
    With `asyncio.gather` inside one node, the renderer would only see
    "subquery_node started" / "subquery_node finished" — no per-branch
    visibility, no progress UX.
  - Send is a documented LangGraph primitive for exactly this shape.
  - The `Annotated[list, operator.add]` reducer on `sub_findings` lets
    the parallel branches each append a single SubFinding without
    races — the framework handles the merge.

This is also the first place in the codebase that actually exercises
`Send`. Every prior parallel point used `asyncio.gather` inside a single
node (per ADR-0001 / CLAUDE.md guidance). Research mode earns the
graph-level fan-out because each branch needs to be independently
visible in the streaming output.
"""

from __future__ import annotations

import operator
from functools import lru_cache
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from horcrux.agents import synthesise_with_history
from horcrux.agents.aggregator import aggregate_subfindings
from horcrux.agents.planner import plan_query
from horcrux.models import (
    Finding,
    Plan,
    ResearchReport,
    ScoredCandidate,
    SubFinding,
)
from horcrux.retrieval.graph import hybrid_search


class ResearchState(TypedDict, total=False):
    """Top-level research-graph state.

    `total=False` so individual nodes can return partial state.

    The reducer on `sub_findings` is the load-bearing piece: parallel
    `subquery` nodes each return `{"sub_findings": [one_finding]}` and
    `operator.add` concatenates them into the final list. Without the
    reducer, the second branch's write would clobber the first.
    """

    query: str
    original_query: str
    plan: Plan
    sub_findings: Annotated[list[SubFinding], operator.add]
    report: ResearchReport


class _SubQueryState(TypedDict):
    """Per-branch state when the planner dispatches via `Send`."""

    sub_question: str
    original_query: str


# ── Nodes ────────────────────────────────────────────────────────


async def _plan_node(state: ResearchState) -> dict:
    """Run the planner. Carries `original_query` through unchanged so
    the trace can show the user-typed query alongside the plan."""
    original = state["query"]
    plan = await plan_query(original)
    return {"plan": plan, "original_query": original}


async def _subquery_node(state: _SubQueryState) -> dict:
    """Run retrieve + synthesise for one sub-question.

    Reuses the existing retrieval graph (`hybrid_search`) and the
    existing synthesis agent. Each call is independent — no shared
    state across sibling branches, no message_history (each sub-query
    is its own conversational context).

    Returns a `sub_findings` delta that `operator.add` merges into the
    parent state's list.
    """
    sub_q = state["sub_question"]
    result = await hybrid_search(query=sub_q, top_k=10)
    if not result.candidates:
        # No retrievable evidence — record a sub-finding with a
        # synthetic conviction-1 Finding rather than raising. The
        # aggregator can decide how to handle it; raising would kill
        # the whole research run on a single empty branch.
        empty_finding = Finding(
            answer=f"No passages retrieved for: {sub_q}",
            source_ids=["__no_evidence__"],
            conviction=1,
            gaps=[f"No candidates retrieved for sub-question: {sub_q}"],
        )
        sub = SubFinding(
            sub_question=sub_q, finding=empty_finding, candidates=[]
        )
        return {"sub_findings": [sub]}

    finding, _ = await synthesise_with_history(
        result.query,
        list(result.candidates),
        message_history=None,
    )
    sub = SubFinding(
        sub_question=sub_q,
        finding=finding,
        candidates=list(result.candidates),
    )
    return {"sub_findings": [sub]}


async def _aggregate_node(state: ResearchState) -> dict:
    """Reduce sub-findings into a `ResearchReport`.

    Filters out the `__no_evidence__` sentinel sub-findings before
    aggregating — they're tracked in the report's sub_findings list (so
    the trace shows the empty branch) but the aggregator shouldn't try
    to cite them.
    """
    sub_findings = list(state["sub_findings"])
    aggregable = [
        sf
        for sf in sub_findings
        if sf.finding.source_ids != ["__no_evidence__"]
    ]
    if not aggregable:
        # Every branch came up empty — produce a graceful failure report
        # rather than crash. Mirrors the synthesis layer's behaviour
        # when retrieval returns nothing.
        report = ResearchReport(
            query=state["query"],
            original_query=state.get("original_query", state["query"]),
            plan=state["plan"],
            sub_findings=sub_findings,
            answer=(
                "No passages were found for any of the planned sub-questions. "
                "Strict-RAG can't ground an answer without evidence."
            ),
            source_ids=["__no_evidence__"],
            conviction=1,
            gaps=[f"sub-question {i}: no passages" for i in range(len(sub_findings))],
        )
        return {"report": report}

    finding = await aggregate_subfindings(state["query"], aggregable)
    report = ResearchReport(
        query=state["query"],
        original_query=state.get("original_query", state["query"]),
        plan=state["plan"],
        sub_findings=sub_findings,
        answer=finding.answer,
        source_ids=finding.source_ids,
        conviction=finding.conviction,
        gaps=finding.gaps,
    )
    return {"report": report}


# ── Conditional edge: plan → Send-fan-out per sub-question ──────


def _dispatch_subqueries(state: ResearchState) -> list[Send]:
    """LangGraph conditional-edge function. Returns one `Send` per
    sub-question; each Send dispatches to the `subquery` node with a
    custom payload.

    This is the first place in the codebase that uses `Send` (vs the
    `asyncio.gather`-inside-a-node pattern we've been using). The
    research case earns it: each sub-query needs to be independently
    visible in `astream` output.
    """
    return [
        Send(
            "subquery",
            {"sub_question": q, "original_query": state["query"]},
        )
        for q in state["plan"].sub_questions
    ]


# ── Graph construction ───────────────────────────────────────────


def build_research_graph() -> CompiledStateGraph:
    graph: StateGraph = StateGraph(ResearchState)

    graph.add_node("plan", _plan_node)
    graph.add_node("subquery", _subquery_node)
    graph.add_node("aggregate", _aggregate_node)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _dispatch_subqueries, ["subquery"])
    graph.add_edge("subquery", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


@lru_cache(maxsize=1)
def _compiled_graph() -> CompiledStateGraph:
    return build_research_graph()


# ── Public API ───────────────────────────────────────────────────


async def research(query: str) -> ResearchReport:
    """One-shot research-mode invocation. Returns the final
    ResearchReport once the graph completes.

    Use this when you don't need streaming visibility — e.g. tests,
    or programmatic callers that want the whole report at once.
    Streaming UI uses `_compiled_graph().astream()` directly.
    """
    if not query:
        raise ValueError("research requires a non-empty query")
    final = await _compiled_graph().ainvoke({"query": query})
    return final["report"]


# ScoredCandidate is exported via SubFinding payload; this re-export
# makes the typing clear at the module surface for tests that need to
# construct one without importing models directly.
__all__ = [
    "ResearchState",
    "ScoredCandidate",
    "build_research_graph",
    "research",
]
