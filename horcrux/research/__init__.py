"""Research mode — multi-step planning + parallel sub-queries + aggregation.

    graph.py     LangGraph state machine: plan → Send-fan-out per
                 sub-question → subquery × N → aggregate.
    renderer.py  Streaming Rich UX consuming graph.astream events,
                 plus static post-hoc rendering for /trace.

Public API:

    from horcrux.research import research            # one-shot run
    from horcrux.research import build_research_graph
    from horcrux.research import StreamingRenderer   # for live streaming
    from horcrux.research import render_report       # for /trace re-render
"""

from horcrux.research.graph import build_research_graph, research
from horcrux.research.renderer import StreamingRenderer, render_report

__all__ = [
    "StreamingRenderer",
    "build_research_graph",
    "render_report",
    "research",
]
