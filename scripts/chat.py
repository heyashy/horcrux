"""Conversational REPL for Horcrux.

Stays alive between queries so follow-up questions can resolve against
prior turns ("what about his brother?" works after a turn about Sirius).
PydanticAI threads the message history into the synthesis call; the
retriever still runs per turn against the post-rewrite query.

    make local && make proxy   # in two other terminals
    make chat

Slash commands inside the REPL:

    /exit, /quit, /q       leave
    /clear                 reset conversation history (keeps thread_id)
    /history               list past turns
    /research <question>   multi-step research with planner + parallel
                           sub-queries + aggregator. Visible reasoning.
    /trace                 re-render the most recent /research run.
    /help                  show commands
"""

import asyncio
import warnings
from dataclasses import dataclass, field

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.rule import Rule

from horcrux.agents import synthesise_with_history
from horcrux.conversation import ChatSession, Turn
from horcrux.models import ResearchReport
from horcrux.research_graph import _compiled_graph as _research_graph
from horcrux.research_renderer import StreamingRenderer, render_report
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
  [bold]/exit, /quit, /q[/]       Leave the chat.
  [bold]/clear[/]                 Reset conversation history; keep thread_id.
  [bold]/history[/]               Print past turns in this session.
  [bold]/research <question>[/]   Multi-step research mode: planner decomposes,
                            sub-queries run in parallel, aggregator
                            synthesises. Reasoning is visible as it runs.
  [bold]/trace[/]                 Re-render the most recent /research run.
  [bold]/help[/]                  Show this help.

Anything not starting with `/` is a normal single-shot question.
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


async def _research(
    console: Console, query: str
) -> ResearchReport | None:
    """Run multi-step research mode with the streaming renderer.

    Returns the final ResearchReport so the caller can stash it for
    /trace re-rendering. None if no useful state was produced (rare —
    the graph emits a graceful failure report rather than returning
    nothing, but we defend anyway).
    """
    renderer = StreamingRenderer(console)
    graph = _research_graph()
    final_report: ResearchReport | None = None
    async for event in graph.astream({"query": query}, stream_mode="debug"):
        renderer.handle(event)
        # Capture the final report from the aggregate event so /trace
        # has something to re-render. The renderer also pulls it but
        # doesn't expose it; cheaper to re-extract here than to thread
        # it through.
        payload = event.get("payload") or {}
        if (
            event.get("type") == "task_result"
            and payload.get("name") == "aggregate"
        ):
            result = payload.get("result") or {}
            final_report = result.get("report") if isinstance(result, dict) else None
    return final_report


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


@dataclass(slots=True)
class _ReplState:
    """REPL-loop state. Bundled so the slash-command dispatcher can
    mutate any of these fields without long parameter lists."""

    console: Console
    session: ChatSession
    message_history: list = field(default_factory=list)
    last_report: ResearchReport | None = None
    should_exit: bool = False


async def _dispatch_command(  # noqa: C901, PLR0911 — slash-command dispatch is intrinsically branchy
    state: _ReplState, query: str
) -> bool:
    """Handle slash commands. Returns True if `query` was a slash command
    (handled or rejected), False if it should fall through to a normal
    Q&A turn."""
    if query in ("/exit", "/quit", "/q"):
        state.should_exit = True
        return True
    if query == "/help":
        state.console.print(Panel(Markdown(_HELP_TEXT), border_style="dim"))
        return True
    if query == "/clear":
        state.session.reset()
        state.message_history = []
        state.console.print("[dim]history cleared[/]")
        return True
    if query == "/history":
        _render_history(state.console, state.session)
        return True
    if query == "/trace":
        if state.last_report is None:
            state.console.print(
                "[dim](no /research run yet — type "
                "[bold]/research <question>[/] first)[/]"
            )
        else:
            render_report(state.console, state.last_report, show_plan=True)
        return True
    if query.startswith("/research"):
        research_query = query[len("/research") :].strip()
        if not research_query:
            state.console.print(
                "[red]usage:[/] /research <your research question>"
            )
            return True
        try:
            state.last_report = await _research(state.console, research_query)
        except Exception as e:
            state.console.print(f"[red]research error:[/] {e}")
        return True
    if query.startswith("/"):
        state.console.print(f"[red]unknown command:[/] {query}  (try /help)")
        return True
    return False


async def _amain() -> None:
    state = _ReplState(console=Console(), session=ChatSession())

    state.console.print(
        Panel.fit(
            f"[bold]Horcrux chat[/] — multi-turn RAG over the corpus.\n"
            f"thread_id: [dim]{state.session.thread_id}[/]\n"
            f"Type [bold]/help[/] for commands, [bold]/exit[/] to leave.",
            border_style="blue",
        )
    )

    while not state.should_exit:
        try:
            raw = Prompt.ask("[bold cyan]you[/]")
        except (EOFError, KeyboardInterrupt):
            state.console.print()
            break
        query = raw.strip()
        if not query:
            continue

        if await _dispatch_command(state, query):
            continue

        try:
            state.message_history = await _ask(
                state.console, state.session, query, state.message_history
            )
        except Exception as e:
            # Catch-all so a single bad turn (rate limit, network blip,
            # malformed model output) doesn't kill the REPL.
            state.console.print(f"[red]error:[/] {e}")

    state.console.print(f"[dim]bye — {len(state.session)} turn(s) this session.[/]")


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
