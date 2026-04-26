"""Conversational REPL for Horcrux.

Stays alive between queries so follow-up questions can resolve against
prior turns ("what about his brother?" works after a turn about Sirius).
PydanticAI threads the message history into the synthesis call; the
retriever still runs per turn against the post-rewrite query.

    make local && make proxy   # in two other terminals
    make chat

Slash commands inside the REPL:

    /exit, /quit, /q   leave
    /clear             reset conversation history (keeps thread_id)
    /history           list past turns
    /help              show commands
"""

import asyncio
import warnings

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from horcrux.agents import synthesise_with_history
from horcrux.conversation import ChatSession, Turn
from horcrux.retrieval_graph import hybrid_search

warnings.filterwarnings("ignore", category=UserWarning, module="qdrant_client")

_CONVICTION_LABEL = {
    5: "very high",
    4: "high",
    3: "moderate",
    2: "low",
    1: "very low",
}

_HELP_TEXT = """\
Commands:
  [bold]/exit, /quit, /q[/]   Leave the chat.
  [bold]/clear[/]             Reset conversation history; keep thread_id.
  [bold]/history[/]           Print past turns in this session.
  [bold]/help[/]              Show this help.
"""


def _render_corrections(console: Console, corrections: list[tuple[str, str]]) -> None:
    if not corrections:
        return
    pairs = ", ".join(f"{o!r} → {n!r}" for o, n in corrections)
    console.print(f"[yellow]did you mean:[/] {pairs}")


def _render_finding(console: Console, turn: Turn) -> None:
    f = turn.finding
    label = _CONVICTION_LABEL.get(f.conviction, "?")
    body = (
        f"{f.answer}\n\n"
        f"[dim]conviction:[/] {f.conviction}/5 ({label})  "
        f"[dim]citations:[/] {len(f.source_ids)}  "
        f"[dim]gaps:[/] {len(f.gaps)}"
    )
    console.print(Panel(body, border_style="green", title="answer", title_align="left"))

    if f.gaps:
        console.print("[yellow]gaps:[/]")
        for g in f.gaps:
            console.print(f"  • {g}")

    by_id = {c.id: c for c in turn.candidates}
    for sid in f.source_ids:
        cand = by_id.get(sid)
        if cand is None:
            console.print(f"[red]missing candidate for source_id={sid}[/]")
            continue
        snippet = " ".join(cand.text.split())[:200]
        if len(cand.text) > 200:
            snippet += "…"
        console.print(
            Panel(
                snippet,
                title=(
                    f"[green]B{cand.book_num}.C{cand.chapter_num}[/] "
                    f"[dim]{cand.chapter_title} (p.{cand.page_start})[/]  "
                    f"[magenta]{cand.source}[/]"
                ),
                title_align="left",
                border_style="dim",
            )
        )


def _render_history(console: Console, session: ChatSession) -> None:
    if not session.history:
        console.print("[dim](no turns yet)[/]")
        return
    console.print(Rule(f"history ({len(session.history)} turns)"))
    for i, turn in enumerate(session.history, start=1):
        console.print(
            f"[cyan]#{i}[/]  [bold]{turn.query}[/]\n"
            f"     [dim]{turn.finding.answer[:120]}"
            f"{'…' if len(turn.finding.answer) > 120 else ''}[/]\n"
        )


async def _ask(
    console: Console,
    session: ChatSession,
    query: str,
    message_history: list,
) -> list:
    """Run one turn: retrieve, synthesise, append to session, render.

    Returns the updated message_history to thread into the next turn.
    """
    console.print("[dim]retrieving…[/]")
    result = await hybrid_search(query=query, top_k=10)
    _render_corrections(console, result.corrections)

    if not result.candidates:
        console.print("[yellow]no candidates retrieved — nothing to synthesise[/]")
        return message_history

    console.print(
        f"[dim]retrieved {len(result.candidates)} candidates; synthesising…[/]"
    )
    finding, new_messages = await synthesise_with_history(
        result.query,
        list(result.candidates),
        message_history=message_history,
    )
    turn = session.record(
        query=query,
        rewritten_query=result.query,
        finding=finding,
        candidates=list(result.candidates),
    )
    _render_finding(console, turn)
    return new_messages


async def _amain() -> None:
    console = Console()
    session = ChatSession()
    message_history: list = []

    console.print(
        Panel.fit(
            f"[bold]Horcrux chat[/] — multi-turn RAG over the corpus.\n"
            f"thread_id: [dim]{session.thread_id}[/]\n"
            f"Type [bold]/help[/] for commands, [bold]/exit[/] to leave.",
            border_style="blue",
        )
    )

    while True:
        try:
            raw = Prompt.ask("[bold cyan]you[/]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        query = raw.strip()
        if not query:
            continue

        if query in ("/exit", "/quit", "/q"):
            break
        if query == "/help":
            console.print(Panel(Markdown(_HELP_TEXT), border_style="dim"))
            continue
        if query == "/clear":
            session.reset()
            message_history = []
            console.print("[dim]history cleared[/]")
            continue
        if query == "/history":
            _render_history(console, session)
            continue
        if query.startswith("/"):
            console.print(f"[red]unknown command:[/] {query}  (try /help)")
            continue

        try:
            message_history = await _ask(console, session, query, message_history)
        except Exception as e:
            # Catch-all so a single bad turn (rate limit, network blip,
            # malformed model output) doesn't kill the REPL.
            console.print(f"[red]error:[/] {e}")

    console.print(f"[dim]bye — {len(session)} turn(s) this session.[/]")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
