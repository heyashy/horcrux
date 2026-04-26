"""Research-mode planner — decomposes a query into focused sub-questions.

This module currently exposes a deterministic stub. The real Haiku-driven
planner agent replaces `_stub_plan` once the streaming UX is validated
against the stub.

Why stub-first: the LangGraph topology (plan → fan-out → aggregate) and
the streaming renderer are independent of *what* the planner produces.
Validating the graph + UX with a deterministic stub means the real LLM
work is only debugging one layer at a time.
"""

from horcrux.models import Plan


def _stub_plan(query: str) -> Plan:
    """Templated decomposition — produces three sub-questions covering
    direct content, surrounding context, and related material.

    Useful enough for UX validation: runs three parallel branches so the
    fan-out is visible, returns a deterministic shape so tests can pin
    behaviour, and the templated wording is recognisably 'stubby' so
    nobody mistakes it for the real planner output.
    """
    return Plan(
        sub_questions=[
            query,
            f"What surrounding context helps answer: {query}",
            f"What related events or characters are relevant to: {query}",
        ],
        rationale=(
            "(stub planner — produces three template sub-questions: direct, "
            "context, related. Replaced by Haiku planner once streaming UX "
            "is validated.)"
        ),
    )


async def plan_query(query: str) -> Plan:
    """Public API — produce a research plan for the given query.

    Async-shaped from day one (the real Haiku planner will be async); the
    stub doesn't need it but matches the eventual signature so downstream
    callers don't change when we swap in the LLM agent.
    """
    return _stub_plan(query)
