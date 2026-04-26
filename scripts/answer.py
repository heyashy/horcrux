"""End-to-end answer pipeline smoke CLI.

Retrieve → synthesise → render. The full strict-RAG path: hybrid_search
pulls candidates from Qdrant, the synthesis agent reads them and produces
a typed Finding, the renderer cross-references source_ids back to their
passages so you can audit every citation.

    make local && make proxy   # in two other terminals
    make answer Q="who killed Cedric Diggory"
    make answer Q="how does Polyjuice Potion work" CHARS="hermione_granger"

If the proxy isn't up, you'll get a clear connection error from PydanticAI.
"""

import argparse
import asyncio
import warnings

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from horcrux.agents import synthesise
from horcrux.ingest import get_client
from horcrux.models import Finding, ScoredCandidate
from horcrux.retrieval_graph import hybrid_search

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

# Shown next to a conviction integer so the rendered panel is self-explanatory.
_CONVICTION_LABEL = {
    5: "very high",
    4: "high",
    3: "moderate",
    2: "low",
    1: "very low",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end answer smoke")
    parser.add_argument("query", help="natural-language question")
    parser.add_argument(
        "--chars", default="", help="comma-separated character slug filter"
    )
    parser.add_argument(
        "-k", "--top-k", type=int, default=10, help="candidate set size"
    )
    return parser.parse_args()


def _render(
    console: Console,
    query: str,
    finding: Finding,
    candidates: list[ScoredCandidate],
) -> None:
    console.print(Rule("answer", style="bold"))
    label = _CONVICTION_LABEL.get(finding.conviction, "?")
    header = (
        f"[bold]{query}[/]\n\n"
        f"[white]{finding.answer}[/]\n\n"
        f"[dim]conviction:[/] {finding.conviction}/5 ({label})  "
        f"[dim]citations:[/] {len(finding.source_ids)}  "
        f"[dim]gaps:[/] {len(finding.gaps)}"
    )
    console.print(Panel(header, border_style="green"))

    if finding.gaps:
        console.print(Rule("gaps", style="yellow"))
        for g in finding.gaps:
            console.print(f"  • {g}")

    console.print(Rule("cited passages", style="dim"))
    by_id = {c.id: c for c in candidates}
    for sid in finding.source_ids:
        cand = by_id.get(sid)
        if cand is None:
            # synthesise() guarantees this is unreachable, but defend anyway —
            # better a confusing message than a KeyError.
            console.print(
                f"[red]missing candidate for source_id={sid}[/] (this should be impossible)"
            )
            continue
        snippet = " ".join(cand.text.split())[:240]
        if len(cand.text) > 240:
            snippet += "…"
        console.print(
            Panel(
                f"[dim]characters:[/] "
                f"{', '.join(cand.characters[:5]) or '(none)'}\n\n{snippet}",
                title=(
                    f"[green]B{cand.book_num}.C{cand.chapter_num}[/] "
                    f"[dim]{cand.chapter_title} (p.{cand.page_start})[/]  "
                    f"[magenta]{cand.source}[/]  "
                    f"[cyan]rrf={cand.score:.4f}[/]"
                ),
                title_align="left",
                border_style="dim",
            )
        )


async def _amain() -> None:
    args = _parse_args()
    console = Console()
    character_filter = [c.strip() for c in args.chars.split(",") if c.strip()]

    console.print(
        Panel.fit(
            f"query: [bold]{args.query}[/]\n"
            + (f"filter: {character_filter}\n" if character_filter else "")
            + f"top_k: {args.top_k}",
            title="answer pipeline",
            border_style="blue",
        )
    )

    client = get_client()

    console.print("[dim]retrieving…[/]")
    result = await hybrid_search(
        client,
        args.query,
        top_k=args.top_k,
        character_filter=character_filter or None,
    )
    if not result.candidates:
        console.print("[yellow]no candidates retrieved — nothing to synthesise[/]")
        return

    if result.corrections:
        pairs = ", ".join(f"{o!r}→{n!r}" for o, n in result.corrections)
        console.print(f"[yellow]did you mean:[/] {pairs}")

    console.print(
        f"[dim]retrieved {len(result.candidates)} candidates; synthesising…[/]"
    )
    finding = await synthesise(result.query, result.candidates)
    _render(console, result.query, finding, result.candidates)


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
