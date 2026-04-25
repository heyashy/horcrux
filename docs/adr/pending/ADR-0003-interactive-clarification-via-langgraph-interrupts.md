# ADR-0003: Interactive clarification via LangGraph interrupts

**Date:** 2026-04-25
**Status:** pending
**Pattern:** Human-in-the-loop via durable graph interrupt + checkpointer (state machine pause/resume).

## Context

Vague queries ("tell me about Snape") waste the most expensive parts of the pipeline: retrieval, candidate scoring, synthesis. The intent classifier (Haiku) can detect when a query lacks the specificity needed to retrieve usefully. The lab needs a mechanism to ask the user a clarifying question and resume with the enriched query.

There are two clean ways to do this:

1. **CLI-only loop.** The graph terminates with a clarification question; the CLI captures a reply and re-invokes the graph with a combined query. Graph itself is unaware.
2. **LangGraph interrupt.** A node calls `interrupt(question)`, which pauses the graph's execution, persists state via a checkpointer, and surfaces the question to the runtime. The runtime captures a reply and resumes the graph with `Command(resume=reply)`. The graph itself orchestrates the pause.

## Decision

Use **LangGraph interrupt** with a SQLite checkpointer (`langgraph-checkpoint-sqlite`). Specifically:

- Extend `QueryIntent` with `quality: Literal["good", "needs_clarification"]` and `clarification_question: str | None`.
- Intent agent's system prompt instructs it to set `quality="needs_clarification"` and write a single specific question when the query lacks named entities, ambiguous scope, or no concrete claim.
- New graph node `ask_clarification` calls `interrupt(intent.clarification_question)`.
- New graph node `merge_query` combines the original query with the user's reply and routes back to `classify_intent`.
- Cap clarification at 2 rounds via a `clarification_count` field in state — prevents infinite loops on truly underspecified queries.
- Checkpointer: SQLite file (`horcrux.db`), gitignored.
- CLI loop catches the interrupt, renders the question via `rich.panel.Panel`, captures the reply via `rich.prompt.Prompt.ask`, resumes with `Command(resume=reply)` and the same `thread_id`.

## Alternatives Considered

**CLI-only loop (option 1 above).** Simpler — no checkpointer, no thread management. Rejected because:
- The interrupt pattern is one of LangGraph's signature features, and demonstrating it is an explicit lab goal.
- The state at the moment of clarification (intent, retry count, partial plan) would have to be reconstructed by the CLI on the second invocation, leaking graph internals into the runtime.
- The interrupt pattern generalises to other future use cases (approval gates, slot filling, disambiguation prompts) that the CLI loop does not.

**Synchronous question via tool call.** Build the clarification as a tool the synthesis agent can call. Rejected because clarification belongs *before* retrieval, not at synthesis time — the entire pipeline is wasted if the query is vague. Synthesis-time clarification is a different, smaller use case.

**Pre-validate query with rules.** Reject queries that don't match a regex or pass a hand-coded vagueness heuristic. Rejected because the heuristic is exactly what the LLM is good at, and brittle in the dimensions a model is most graceful in (idioms, named entity recognition, intent inference).

## Consequences

**Positive**
- The graph is the orchestrator. Pause/resume state is owned by LangGraph, not the CLI. The CLI is dumb — it forwards user input.
- The pattern generalises. Future work (approval gates before destructive actions, "did you mean X or Y?", multi-step slot filling) reuses the same machinery.
- Observability: the interrupt and the resume both appear as nodes in the LangSmith trace tree, so debugging routing is straightforward.

**Negative / risks**
- Adds a checkpointer dependency and a database file (`horcrux.db`) to the lab footprint. Small, but real.
- A `thread_id` must be allocated per query and passed through the CLI invocation. Mistakes (re-using a thread, losing it on crash) cause confusing replays. Mitigation: thread IDs are UUIDs minted at CLI invocation, never persisted across runs.
- The 2-round cap on clarifications is arbitrary. If users hit it often, retrieval quality is bounded by initial query specificity. Worth measuring.

**Follow-ups**
- Track how often clarification fires. If common, the intent classifier's threshold may be too aggressive; if rare, queries are well-formed and the feature is mostly defensive.
- Consider rendering the conversation history in the second clarification round (not just the latest question) so the user can see context.

## Rollback

If the interrupt pattern proves more friction than it earns:

1. Remove the `ask_clarification` and `merge_query` nodes from the graph.
2. Replace the routing condition that selects them with `END` — vague queries terminate with a `gaps`-only `ResearchReport`.
3. Drop the SQLite checkpointer dependency.

The CLI then becomes a single-shot invocation again. ~30 minutes of work. The `quality` and `clarification_question` fields on `QueryIntent` can stay as informational metadata even if not used to drive a graph node.
