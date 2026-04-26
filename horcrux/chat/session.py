"""Multi-turn chat session — pure data structures, no I/O.

A `ChatSession` owns the conversation between user and assistant: a
stable `thread_id` for traceability and a list of `Turn` records, each
holding one user query and the typed `Finding` produced for it.

Why a separate module rather than dropping this in `agents.py`:
- Agents are about LLM mechanics (prompts, retries, schema enforcement).
- A chat session is about conversation state (turn ordering, history
  windowing, what counts as context for follow-up questions).
- Keeping them apart lets the agent stay stateless — every `synthesise`
  call is self-contained and idempotent. State lives here.

The session does *not* persist itself to disk. Persistent threads
(survive across CLI restarts) are out of scope for this phase; the
`thread_id` is allocated in-memory at session start and lost on exit.
LangGraph's SQLite checkpointer is reserved for clarification interrupts
in a later phase, where mid-graph resume actually matters.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from horcrux.models import Finding, ScoredCandidate


@dataclass(slots=True)
class Turn:
    """One round-trip: user query and the assistant's typed answer.

    `candidates` is captured so the chat UI can re-render the citations
    later (e.g. for a `/history` command). Held by reference, not copied
    — `ScoredCandidate`s are immutable Pydantic models so aliasing is
    safe.
    """

    query: str
    """The user's query as typed (pre-rewrite)."""
    rewritten_query: str
    """The query the retrievers actually saw. Equal to `query` when no
    rewriter corrections were applied."""
    finding: Finding
    candidates: list[ScoredCandidate]


@dataclass(slots=True)
class ChatSession:
    """Multi-turn conversation state.

    `thread_id` is a UUID per session — used for traceability against
    LangSmith / Temporal (which the lab's observability layer threads
    on) and as the eventual key for the SQLite checkpointer.

    `history` is append-only. The session doesn't trim or summarise old
    turns — that's a future concern when context windows start
    pressuring. Empirically a few turns of HP Q&A fit comfortably under
    Sonnet's input budget.
    """

    thread_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    history: list[Turn] = field(default_factory=list)

    def record(
        self,
        query: str,
        rewritten_query: str,
        finding: Finding,
        candidates: list[ScoredCandidate],
    ) -> Turn:
        """Append a turn and return it. Returning the Turn lets callers
        chain rendering off the same value without reaching back into
        the list."""
        turn = Turn(
            query=query,
            rewritten_query=rewritten_query,
            finding=finding,
            candidates=candidates,
        )
        self.history.append(turn)
        return turn

    def reset(self) -> None:
        """Clear history but keep the same thread_id — the session is
        still the same logical conversation, just with no carryover
        context. Use a new ChatSession() instead if you want a fresh
        thread_id too."""
        self.history.clear()

    def __len__(self) -> int:
        return len(self.history)
