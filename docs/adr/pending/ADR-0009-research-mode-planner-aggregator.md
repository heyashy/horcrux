# ADR-0009: Research mode — planner + parallel sub-queries + aggregator

**Date:** 2026-04-26
**Status:** pending
**Pattern:** Multi-step LLM pipeline (Plan-Execute-Aggregate); first-class
LangGraph fan-out via `Send`.

## Context

Single-shot RAG works well for focused questions ("who killed Cedric
Diggory") but fails for multi-faceted ones ("tell me about Snape's
story arc"). The failure mode isn't retrieval recall — it's that
top_k=10 candidates retrieved against a vague query semantically
resemble *none* of the passages that would actually answer the
multi-part question.

Two paths to better answers on this kind of question:

1. **Long-context** — give the model 100k+ tokens of context (whole
   chapters or whole books) and let it pick what's relevant. Brute
   force; works at the cost of latency, $, and signal-to-noise.
2. **Query planning** — decompose the question into focused
   sub-questions, retrieve and synthesise per sub-question
   independently, then aggregate. The system runs N+2 LLM calls (1
   plan + N synthesis + 1 aggregate) but each call has tightly-scoped
   context.

The lab picks (2). The differentiator from the long-context approach
is **visible reasoning** — the planner's sub-questions and the
per-sub-query progress are surfaced to the user as the system runs.
This converts "the agent is doing something" into "the agent is doing
*this*". Trust grows with visibility; opacity reads as magic at best
and as broken at worst.

## Decision

**Research mode** is a separate LangGraph (`horcrux/research_graph.py`)
that orchestrates a Plan-Execute-Aggregate pattern:

```
START → plan → Send-fan-out per sub-question → subquery × N → aggregate → END
```

- **Plan node**: Haiku-driven PydanticAI agent (`horcrux/planner.py`).
  Output: typed `Plan` with `min_length=1`, `max_length=8`
  sub-questions plus a rationale. The schema bounds prevent
  pathological plans (no decomposition; over-decomposition that burns
  the rate limit).
- **Send-fan-out**: LangGraph's `Send` primitive dispatches one
  parallel branch per sub-question. Each branch is its own node
  invocation, so `astream` yields a per-sub-query event that the
  streaming renderer surfaces.
- **Subquery nodes**: each runs the existing single-shot pipeline —
  `hybrid_search` (the four-way dense + BM25 fusion from Phase 4.5)
  followed by `synthesise` (Sonnet, strict-RAG-bounded). The
  sub-finding's candidates and source_ids are preserved.
- **Aggregate node**: Sonnet-driven PydanticAI agent
  (`horcrux/aggregator.py`). Reads all sub-findings, plus a flattened
  deduplicated candidate set, and produces a final coherent `Finding`
  citing across the merged set. Strict-RAG continues at this layer:
  `source_ids` schema invariant + runtime check that every cited
  passage number is in `[1..N]`.
- **Output**: typed `ResearchReport` carrying the original query, the
  plan, every sub-finding (preserved for `/trace`), and the final
  answer. The whole reasoning chain is one Pydantic value.

**Streaming visibility is load-bearing.** The renderer
(`horcrux/research_renderer.py`) consumes
`graph.astream(stream_mode="debug")` and prints node-lifecycle events
as they happen — the plan as soon as the planner produces it,
per-sub-query "running…" markers, completion lines with
hits/citations/conviction/elapsed, and the final answer. This is the
demo surface: anyone watching can see the agent reason rather than
stare at a frozen prompt.

**Surfaces:**
- `make research Q="..."` — standalone one-shot research mode.
- `/research <q>` slash command in `make chat`.
- `/trace` slash command — re-renders the most recent `ResearchReport`
  (plan + per-sub-query findings + final answer + citations) for
  post-hoc inspection.

## Alternatives Considered

### Long-context single-shot (large model, large prompt)

Hand the whole corpus (or large slices) to a long-context model
(Gemini 1.5/2.5, Claude with 1M context) and let it pick relevant
passages itself. Rejected for this lab:

- Token cost grows linearly with corpus size; not zero even at
  Gemini's price point.
- Loses the visible-reasoning UX. The model picks passages
  internally; we never see the plan.
- Doesn't compose with strict-RAG citations — the model can cite
  whatever it wants from the input, with no chunk-level
  identification.
- Doesn't transfer to corpora larger than the context window.

This is the fast path for very small corpora and single-shot
"summarise this document" use cases. Not what the lab is exploring.

### `asyncio.gather` inside a single subquery node

Run all sub-queries inside one node's body via `asyncio.gather`
instead of fanning out via `Send`. Simpler graph topology — only
plan, subquery, aggregate — but only one node-event for the whole
sub-query phase.

Rejected because **streaming visibility per sub-query is the load-
bearing UX**. With a single node we'd see "subquery_node started" /
"subquery_node finished" — no per-branch progress, no individual
"running…" markers. The whole rationale for query planning over
long-context is the visible reasoning, and `Send` is the LangGraph
mechanism that lets each branch be visible.

This is also the first place in the codebase that genuinely earns
`Send`. CLAUDE.md previously preferred `asyncio.gather` (Phase 4 four-
way fusion uses gather inside a single node, because the four
retrievers fuse together immediately without needing per-retriever
visibility). Research mode flips the rationale.

### Recursive planning (sub-questions can spawn their own
sub-questions)

A sub-question that itself looks complex could trigger another planner
call. Rejected: predictability matters more than power for the lab.
A linear `plan → execute → aggregate` is comprehensible; a recursive
agent loop adds tail-cost variance, retry logic, and
"runaway-budget" failure modes that don't fit a weekend lab. The
straightforward shape is sufficient for HP-corpus questions; richer
agents are a future-work item.

### Same agent for plan + synthesis + aggregate (Sonnet
everywhere)

One model handles all three roles. Rejected: the planner is a
structural / decomposition task that Haiku does cheaply and well.
Sonnet's capability margin doesn't help at the plan layer; latency
and cost do. The split keeps research-mode latency bounded and lets
the budget stay focused on synthesis (where reasoning quality
genuinely matters).

### LangGraph SQLite checkpointer for research-mode resume

Persist mid-run state so a research run can resume after a crash or a
clarification interrupt. Out of scope this phase: research runs are
tens of seconds long and idempotent (rerun gets the same plan
roughly, the same hits exactly). Worth re-visiting alongside
clarification interrupts (deferred to a later phase).

## Consequences

### What this lab now demonstrates

- Multi-step LLM orchestration via LangGraph that's actually
  *visible* in the terminal. The system is no longer a black box
  between prompt and answer.
- A genuine production-style research-agent shape: cheap planner,
  parallel workers, capable aggregator, all bounded by typed schemas.
- That **F20 (conviction calibration anchoring high) is organically
  resolved by the architecture**, not by prompt-tuning. Each sub-
  finding has its own conviction; the aggregator is bounded by the
  weakest sub-finding via the rubric's "pick the lower number" rule.
  Empirically: the Snape arc question returns conviction 4/5 because
  one sub-finding is 3/5, even though others are 5/5. The single-shot
  path on the same question would have returned 5/5 (anchoring high).
- That **strict-RAG composes**. Each sub-synthesis enforces source_ids
  non-empty + runtime ID check; the aggregator does the same. Three
  defensive layers per LLM call, applied independently at each layer.

### Empirical results on the motivating query

`make research Q="tell me about Snape's story arc"` (2026-04-26):

- Plan: 4 sub-questions (early-book role, Prince's Tale, Dumbledore's
  killing, Harry's view changing). Planner output time: ~2s.
- Sub-queries ran in parallel: 29-36s wall clock for all four,
  10 hits each, conviction 3-5/5 honestly distributed.
- Aggregator: ~12s. Final conviction 4/5 (correctly bounded by the
  weakest sub-finding).
- 19 unique citations across all 7 books in the final answer.
- `gaps` field listed three honest absences: Prince's Tale memory
  contents not directly retrieved, Marvolo Gaunt ring detail
  partially covered, Harry's emotional aftermath not in passages.

Compare to single-shot `make answer` on the same question: returned a
shallower answer, conviction 3/5 with 9 citations, missed Lily-love
thread and the planned-killing reveal.

### Tunable knobs and their defaults

- `Plan.sub_questions` length: schema-bounded `[1, 8]`; planner system
  prompt aims for 3-5.
- Sub-query `top_k`: 10 (matches single-shot default).
- Aggregator passage budget: chapter chunks truncated to 200 words
  (same `_truncate_for_synthesis` as the single-shot path) so the
  flattened candidate set fits under Sonnet's per-call rate limit.
- Planner model: `settings.litellm.haiku_alias`.
- Aggregator model: `settings.litellm.sonnet_alias`.

### Latency profile

Wall-clock is roughly:

- Plan: ~2s (Haiku, short output).
- Sub-queries (parallel): max(per-query latency) ≈ 25-40s. Each
  sub-query is a full retrieve + Sonnet synthesise.
- Aggregate: ~10-15s (Sonnet over flattened context).

Total: ~40-60s for a 4-sub-question run. Streamed visibility
absorbs the perceived latency — users watch progress rather than
wait on a blank screen.

### Failure modes and graceful degradation

- A sub-query that retrieves zero candidates emits a sentinel
  `__no_evidence__` SubFinding rather than raising. Aggregator
  filters these before the LLM call; the trace still shows the
  empty branch.
- All sub-queries empty → aggregator returns a graceful
  ResearchReport with `__no_evidence__` source_ids and
  conviction 1/5 instead of crashing the run.
- Aggregator runtime citation check fails → `ValueError` propagated
  through `astream`; chat REPL catches and displays it without
  killing the session.

### What this *doesn't* solve

- **Latency** — research mode is genuinely 30-60s. Acceptable for
  exploratory research questions, not for snappy conversational use.
  Single-shot `synthesise` remains the default for the chat REPL;
  research mode is opt-in via `/research`.
- **Iterative refinement** — sub-questions don't get to spawn their
  own sub-questions. Linear plan-execute-aggregate.
- **Cross-turn research history** — each `/research` call is its own
  thread; the session-level `ChatSession` history doesn't thread into
  the research planner. Could be added later (planner sees prior
  Q&A for context), out of scope this phase.

## Rollback

Research mode is purely additive:

- `horcrux/research_graph.py`, `horcrux/planner.py`,
  `horcrux/aggregator.py`, `horcrux/research_renderer.py`, and
  `scripts/research.py` can be deleted in entirety.
- The `Plan` / `SubFinding` / `ResearchReport` Pydantic models in
  `horcrux/models.py` are unused outside research mode and can be
  deleted alongside.
- The `_research` / `_dispatch_command` block in `scripts/chat.py`
  reverts to the previous slash-command structure.
- The `make research` target reverts.

No data migration. Single-shot `make answer` and `make chat` are
unaffected.

The double-checked-locking guard in `horcrux/embedding.py` (added
during research-mode integration to prevent parallel
`SentenceTransformer` constructors from racing on the GPU) stays
either way — it's correct under any concurrency model.
