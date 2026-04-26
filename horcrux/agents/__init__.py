"""LLM-driven agents.

Three agents, three roles:

    synthesis.py   Single-shot synthesis. Reads a candidate set,
                   produces a typed Finding under strict-RAG rules.
                   Used by `make answer` and `make chat`.
    planner.py     Haiku-driven decomposition. Splits a research
                   question into focused sub-questions.
    aggregator.py  Sonnet-driven cross-finding synthesis. Reads
                   per-sub-question Findings + a flattened candidate
                   set, produces a coherent final answer.

All three are PydanticAI agents pointed at the LiteLLM proxy via
OpenAI-compatible interface (per ADR-0002). Strict-RAG (system prompt
+ schema invariant + runtime in-range citation check) applies at the
synthesis and aggregator layers; the planner is parametric by design.
See ADR-0009 for the limits of that design.
"""

from horcrux.agents.aggregator import aggregate_subfindings
from horcrux.agents.planner import plan_query
from horcrux.agents.synthesis import synthesise, synthesise_with_history

__all__ = [
    "aggregate_subfindings",
    "plan_query",
    "synthesise",
    "synthesise_with_history",
]
