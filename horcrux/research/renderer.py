"""Streamed renderer for research-mode runs — the visible reasoning.

Drives a Rich console from LangGraph's `astream(stream_mode="debug")`
events. Each node lifecycle event (`task` start, `task_result` end)
maps to a status update so the user sees the agent reasoning in
real time:

    ▶ Planning…
       ↳ <sub-question 1>
       ↳ <sub-question 2>
       ↳ <sub-question 3>
    ▶ Sub-queries (3 in parallel)
       ⠋ <sub-question 1>            running…
       ✓ <sub-question 2>            8 hits, conviction 4/5  3.2s
       ⠋ <sub-question 3>            running…
    ▶ Synthesising final report…
    [final answer panel]
    [per-citation panels]

Also exposes `render_report()` for static post-hoc rendering — `/trace`
in the chat REPL re-renders saved `ResearchReport`s through the same
visual language.
"""

from __future__ import annotations

import time
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from horcrux.models import ResearchReport, ScoredCandidate, SubFinding

_CONVICTION_LABEL = {
    5: "very high",
    4: "high",
    3: "moderate",
    2: "low",
    1: "very low",
}


def _conviction_str(conviction: int) -> str:
    label = _CONVICTION_LABEL.get(conviction, "?")
    return f"{conviction}/5 ({label})"


def _snippet(text: str, limit: int = 220) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit] + "…"


# ── Live streaming during execution ──────────────────────────────


class StreamingRenderer:
    """Consumes `graph.astream(stream_mode='debug')` events and prints
    progressive status to a Rich console.

    Stateful: tracks per-sub-query start times so each completion line
    can show wall-clock elapsed. Sub-question text is captured at task
    start and recalled at task_result time (the result event's payload
    doesn't include the input).
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._subquery_started: dict[str, float] = {}
        self._subquery_text: dict[str, str] = {}
        self._subquery_count = 0

    def handle(self, event: dict[str, Any]) -> None:
        """Dispatch a single astream debug event to the right handler."""
        kind = event.get("type")
        payload = event.get("payload", {}) or {}
        name = payload.get("name")

        if kind == "task" and name == "plan":
            self._on_plan_start()
        elif kind == "task_result" and name == "plan":
            self._on_plan_end(payload)
        elif kind == "task" and name == "subquery":
            self._on_subquery_start(payload)
        elif kind == "task_result" and name == "subquery":
            self._on_subquery_end(payload)
        elif kind == "task" and name == "aggregate":
            self._on_aggregate_start()
        elif kind == "task_result" and name == "aggregate":
            self._on_aggregate_end(payload)

    # ── plan ────────────────────────────────────────────────────

    def _on_plan_start(self) -> None:
        self.console.print("\n[bold cyan]▶ Planning…[/]")

    def _on_plan_end(self, payload: dict) -> None:
        result = payload.get("result") or {}
        plan = result.get("plan") if isinstance(result, dict) else None
        if plan is None:
            return
        for sq in plan.sub_questions:
            self.console.print(f"   [dim]↳[/] {sq}")
        if plan.rationale:
            self.console.print(f"   [dim italic]rationale:[/] [dim]{plan.rationale}[/]")

    # ── subquery ────────────────────────────────────────────────

    def _on_subquery_start(self, payload: dict) -> None:
        task_id = payload.get("id", "")
        sub_q = payload.get("input", {}).get("sub_question", "?")
        if self._subquery_count == 0:
            self.console.print("\n[bold cyan]▶ Sub-queries (parallel)[/]")
        self._subquery_count += 1
        self._subquery_started[task_id] = time.monotonic()
        self._subquery_text[task_id] = sub_q
        self.console.print(f"   [yellow]⠋[/] {sub_q}  [dim]…running[/]")

    def _on_subquery_end(self, payload: dict) -> None:
        task_id = payload.get("id", "")
        sub_q = self._subquery_text.get(task_id, "?")
        elapsed = time.monotonic() - self._subquery_started.get(task_id, time.monotonic())

        # Pull the SubFinding out of the result dict.
        result = payload.get("result") or {}
        sub_list = result.get("sub_findings") if isinstance(result, dict) else None
        sub: SubFinding | None = sub_list[0] if sub_list else None
        if sub is None:
            self.console.print(f"   [red]✗[/] {sub_q}  [dim]no result[/]")
            return

        marker = "[green]✓[/]" if sub.candidates else "[yellow]∅[/]"
        n_hits = len(sub.candidates)
        n_cites = (
            len(sub.finding.source_ids)
            if sub.finding.source_ids != ["__no_evidence__"]
            else 0
        )
        self.console.print(
            f"   {marker} {sub_q}  "
            f"[dim]{n_hits} hits, {n_cites} cited, "
            f"conviction {sub.finding.conviction}/5, {elapsed:.1f}s[/]"
        )

    # ── aggregate ───────────────────────────────────────────────

    def _on_aggregate_start(self) -> None:
        self.console.print("\n[bold cyan]▶ Synthesising final report…[/]")

    def _on_aggregate_end(self, payload: dict) -> None:
        result = payload.get("result") or {}
        report = result.get("report") if isinstance(result, dict) else None
        if report is None:
            return
        self.console.print()
        render_report(self.console, report, show_plan=False)


# ── Static post-hoc rendering (for /trace and standalone) ────────


def render_report(  # noqa: C901 — sequential rendering of plan/findings/answer/citations is one cohesive operation
    console: Console, report: ResearchReport, *, show_plan: bool = True
) -> None:
    """Render a finished `ResearchReport`. Used by /trace and by the
    streaming renderer as the post-aggregate output. Set `show_plan` to
    True for /trace (where the user wants the full reasoning chain) and
    False for live output (where the plan was already shown progressively).
    """
    if show_plan:
        console.print(Rule(f"research trace — {report.original_query!r}"))
        console.print(f"\n[bold]plan[/] ({len(report.plan.sub_questions)} sub-questions)")
        for sq in report.plan.sub_questions:
            console.print(f"   [dim]↳[/] {sq}")
        if report.plan.rationale:
            console.print(f"   [dim italic]rationale:[/] [dim]{report.plan.rationale}[/]")

        console.print(f"\n[bold]sub-findings[/] ({len(report.sub_findings)})")
        for sf in report.sub_findings:
            n_cites = (
                len(sf.finding.source_ids)
                if sf.finding.source_ids != ["__no_evidence__"]
                else 0
            )
            marker = "[green]✓[/]" if sf.candidates else "[yellow]∅[/]"
            console.print(
                f"   {marker} {sf.sub_question}  "
                f"[dim]{len(sf.candidates)} hits, {n_cites} cited, "
                f"conviction {sf.finding.conviction}/5[/]"
            )
            console.print(f"     [dim]{_snippet(sf.finding.answer, 200)}[/]\n")

    console.print(Rule("final answer", style="bold green"))
    body = (
        f"{report.answer}\n\n"
        f"[dim]conviction:[/] {_conviction_str(report.conviction)}  "
        f"[dim]citations:[/] {len(report.source_ids)}  "
        f"[dim]gaps:[/] {len(report.gaps)}"
    )
    console.print(Panel(body, border_style="green"))

    if report.gaps:
        console.print("\n[yellow]gaps:[/]")
        for g in report.gaps:
            console.print(f"  • {g}")

    # Citation drill-down: the report's source_ids reference candidates
    # spread across sub-findings. Build a lookup so each cited passage
    # gets its own panel.
    by_id: dict[str, ScoredCandidate] = {}
    for sf in report.sub_findings:
        for c in sf.candidates:
            by_id.setdefault(c.id, c)

    if report.source_ids and report.source_ids != ["__no_evidence__"]:
        console.print(Rule("cited passages", style="dim"))
        for sid in report.source_ids:
            cand = by_id.get(sid)
            if cand is None:
                console.print(f"[red]missing candidate for source_id={sid}[/]")
                continue
            console.print(
                Panel(
                    _snippet(cand.text, 240),
                    title=(
                        f"[green]B{cand.book_num}.C{cand.chapter_num}[/] "
                        f"[dim]{cand.chapter_title} (p.{cand.page_start})[/]  "
                        f"[magenta]{cand.source}[/]"
                    ),
                    title_align="left",
                    border_style="dim",
                )
            )
