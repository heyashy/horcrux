"""Standalone research-mode CLI — show the agent thinking as it works.

End-to-end research run: planner decomposes the query into sub-questions,
each runs retrieve+synthesise in parallel, an aggregator produces the
final report. Every node-lifecycle event surfaces in the terminal so the
user can watch the agent reason.

    make local && make proxy   # in two other terminals
    make research Q="tell me about Snape's story arc"
"""

import argparse
import asyncio
import warnings

from rich.console import Console
from rich.panel import Panel

from horcrux.research.graph import _compiled_graph
from horcrux.research.renderer import StreamingRenderer

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-step research mode")
    parser.add_argument("query", help="natural-language research question")
    return parser.parse_args()


async def _amain() -> None:
    args = _parse_args()
    console = Console()

    console.print(
        Panel.fit(
            f"[bold]research[/]\n{args.query}",
            border_style="blue",
        )
    )

    renderer = StreamingRenderer(console)
    graph = _compiled_graph()
    async for event in graph.astream({"query": args.query}, stream_mode="debug"):
        renderer.handle(event)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
