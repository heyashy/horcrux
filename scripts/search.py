"""Hybrid-search smoke CLI.

End-to-end retrieval against live Qdrant — encode the query, hit both
collections, RRF-merge, render top-N with Rich. Exists to (a) sanity-
check the retrieval layer before Phase 5 wires it into PydanticAI agents,
and (b) give an interactive debugging surface — pull the alias dict, try
character filters, eyeball the ranking.

    make search Q="Voldemort returns to power"
    make search Q="charm" CHARS="hermione_granger"
    make search Q="quidditch" K=20

The K knob controls top_k of the fused result; paragraph_k and chapter_k
are left at module defaults to avoid burying tunables in shell.
"""

import argparse
import asyncio
import warnings

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from horcrux.ingest import get_client
from horcrux.retrieval_graph import hybrid_search

# Suppress the qdrant-client / server version-mismatch warning.
# Lab is locked to 1.12.4 server; client moves faster — known divergence.
warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid retrieval smoke test")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument(
        "--chars",
        default="",
        help="comma-separated character slug IDs to filter on",
    )
    parser.add_argument("-k", "--top-k", type=int, default=10, help="fused top_k")
    return parser.parse_args()


def _render(console: Console, query: str, hits: list) -> None:
    if not hits:
        console.print("[yellow]no hits[/]")
        return

    console.print(Rule(f"top {len(hits)} hits for {query!r}", style="bold"))

    for rank, h in enumerate(hits, start=1):
        snippet = " ".join(h.text.split())[:280]
        if len(h.text) > 280:
            snippet += "…"
        chars = ", ".join(h.characters[:5])
        if len(h.characters) > 5:
            chars += f"  (+{len(h.characters) - 5} more)"

        header = (
            f"[bold]#{rank}[/]  "
            f"[cyan]rrf={h.score:.4f}[/]  "
            f"[magenta]{h.source}[/]  "
            f"[green]B{h.book_num}.C{h.chapter_num}[/] "
            f"[dim]{h.chapter_title}[/] "
            f"[dim](p.{h.page_start})[/]"
        )
        body = f"[dim]characters:[/] {chars or '[dim](none)[/]'}\n\n{snippet}"
        console.print(Panel(body, title=header, title_align="left", border_style="dim"))


async def _amain() -> None:
    args = _parse_args()
    console = Console()

    character_filter = [c.strip() for c in args.chars.split(",") if c.strip()]
    if character_filter:
        console.print(
            Panel.fit(
                f"query: [bold]{args.query}[/]\n"
                f"filter: {character_filter}\n"
                f"top_k: {args.top_k}",
                title="hybrid_search",
                border_style="blue",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"query: [bold]{args.query}[/]\ntop_k: {args.top_k}",
                title="hybrid_search",
                border_style="blue",
            )
        )

    client = get_client()
    result = await hybrid_search(
        client,
        args.query,
        top_k=args.top_k,
        character_filter=character_filter or None,
    )
    if result.corrections:
        pairs = ", ".join(f"{o!r}→{n!r}" for o, n in result.corrections)
        console.print(f"[yellow]did you mean:[/] {pairs}")
    _render(console, result.query, result.candidates)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
