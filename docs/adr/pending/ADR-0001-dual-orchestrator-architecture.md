# ADR-0001: Dual-orchestrator architecture with strict-RAG grounding

**Date:** 2026-04-24
**Status:** pending
**Pattern:** Layered orchestration — durable workflow engine (ingest) + graph state machine (query), sharing a typed contract layer (PydanticAI). Strict-RAG grounding policy enforced at three layers.

## Context

Horcrux has two fundamentally different runtime shapes:

1. **Ingest** — OCR 3623 pages (~48 minutes), detect chapters, chunk, embed, upsert to Qdrant. Long-running, side-effectful, must survive worker crashes without re-running completed work.
2. **Query** — seven-node pipeline: classify intent → plan → parallel retrieve → score candidates → synthesise report. Short-lived (seconds), conditional routing, parallel LLM fan-out, typed outputs at every step.

These have incompatible operational requirements. Ingest is measured in minutes and needs durable execution. Query is measured in seconds and needs graph-shaped control flow.

We also need typed, validated outputs from every LLM call — `conviction: "high"` must fail fast rather than propagate downstream.

## Decision

**Orchestration:** split by timescale and shape.

- **Temporal** orchestrates the ingest workflow. Activities handle OCR, chapter detection, chunking, and Qdrant upserts. Event history guarantees crashed work resumes at the last completed activity.
- **LangGraph** orchestrates the query pipeline. `HorcruxState` TypedDict is reduced across nodes. Conditional edges express routing (confidence gates, retry loops). Parallel retrieval fans out via `asyncio.gather` inside a single node.
- **PydanticAI** is the common typed boundary at every LLM call in both subsystems. Agent outputs are declared as Pydantic models; validation retry is automatic.

Data models (`horcrux/models.py`) are the shared contract between ingest and query — ingest produces `ChapterChunk`, query consumes via `raw_candidates` → `scored_candidates` → `ResearchReport`.

**Grounding policy: strict RAG.** The synthesis agent may only make claims supported by the retrieved chunks. Model bleed — answering from Claude's training knowledge rather than the corpus — is explicitly disallowed. Enforced at three layers:
1. System prompt instruction.
2. Schema invariant: `Finding.source_ids` has `min_length=1` — PydanticAI retries on violation.
3. Runtime check: every `source_id` must exist in the candidates passed to the agent.

**Chunking:** two Qdrant collections (`hp_chapters`, `hp_paragraphs`). A `characters: list[str]` payload populated at ingest time serves relational queries via filter, replacing the originally-planned `hp_entity_mentions` collection.

**Conviction calibration:** a 1-5 rubric baked into the synthesis system prompt, with a contradiction cap (if findings contradict, conviction is capped at 3 and the contradiction is surfaced in `gaps`). Combined with a `Field(description=...)` instructing the model to pick the lower value when uncertain.

**Observability:** LangSmith tracing enabled from day one via two env vars. No code instrumentation — LangGraph and PydanticAI auto-report. Console logging via `rich.logging.RichHandler`.

**Infrastructure:** local only. `temporal server start-dev` + `qdrant/qdrant` Docker container. `ANTHROPIC_API_KEY` and `LANGCHAIN_API_KEY` are the only secrets.

## Alternatives Considered

**LangGraph-only.** LangGraph has checkpointing (SQLite/Postgres persistence between nodes) which superficially resembles Temporal durability. Rejected because LangGraph checkpointing is designed for interactive pipelines with human-in-the-loop pauses, not 48-minute batch jobs with per-activity retry policies and visibility via a dedicated UI. Shoehorning ingest into LangGraph would lose the crash-test exercise that makes Temporal's value concrete.

**Temporal-only.** Expressing the query pipeline as Temporal activities works but is awkward. Conditional routing becomes nested activity calls; parallel LLM fan-out becomes child workflows; the state-reducer model LangGraph offers disappears. The query pipeline is the *wrong shape* for Temporal — it's a graph, not a DAG of durable steps.

**Prefect / Airflow.** Both are scheduled batch pipeline tools. Neither offers interactive query latency nor first-class support for LLM-driven conditional control flow. Wrong category.

**Raw asyncio + a custom state machine.** Loses replay (Temporal), loses graph visibility (LangGraph), loses typed boundaries (PydanticAI). Would require rebuilding all three primitives badly.

**Hybrid grounding (RAG + parametric fallback).** Allow the synthesis agent to fall back on Claude's training knowledge when retrieval is insufficient. Rejected: conviction scoring becomes meaningless (what does a score mean when sources are empty?), and more importantly, we lose the ability to tell whether retrieval quality is adequate. If the LLM can paper over retrieval gaps, every chunking and routing decision becomes unmeasurable. Strict RAG preserves the feedback loop that makes the lab worth running.

**Three-collection chunking with `hp_entity_mentions`.** Dedicated sentence-level collection with one chunk per `(sentence, character-mentioned)`. Rejected on complexity and chunk-count grounds (estimated ~120k chunks for the HP corpus vs ~30k for paragraphs). A `characters: list[str]` payload on paragraph chunks delivers the same relational-query capability through filtering at a fraction of the storage and embedding cost — with richer context per chunk (a paragraph vs a sentence triple).

## Consequences

**Positive**
- Each tool does one thing well; concerns are cleanly separated.
- Evaluation value: the three tools are orthogonal, so each one's role is observable in isolation rather than conflated — a clean comparative read for the lab.
- The crash test (kill worker at batch 15, watch it resume) is a natural exercise and works exactly as Temporal documents advertise.
- LangSmith traces (when enabled) visualise the query graph live — debugging routing decisions becomes a UI task, not a log-grep task.

**Negative / risks**
- Two orchestrators to operate. Justified for a bounded lab evaluating both tools side-by-side; would need consolidation review before production.
- `pydantic-ai` is pre-1.0; API churn is expected. Tracked with an unpinned `>=0.0.14` requirement — acceptable for a time-boxed lab, would need pinning in production.
- Determinism rules in Temporal workflows (`no datetime.now()`, `no random`, `no file I/O` outside activities) are easy to violate accidentally. Mitigation: all non-deterministic operations are wrapped in activities; reviewers check workflow code specifically for determinism violations.
- Character extraction at ingest depends on a maintained canonical-name list. Drift between the canonical list and the corpus (new aliases introduced over time) silently degrades relational queries. Mitigation: a smoke test that asserts known canonical pairs (e.g. "Snape" + "Half-Blood Prince") co-occur in expected chunks.

**Follow-ups**
- If the paragraph-level `characters` filter proves insufficient for relational queries (e.g. too coarse), revisit sentence-level chunking in a follow-up ADR with measured results.
- Conviction rubric will be validated empirically — run a fixed set of queries, inspect whether the distribution actually spreads across 1-5. If bunching persists, move to deterministic derivation (Option B in the system design discussion).
- The low-candidate / zero-candidate graph paths need explicit tests to confirm graceful degradation, not error.

## Rollback

Not an infrastructure change — no state migrations required. If the dual-orchestrator split proves wrong:
- All ingest code can be ported to LangGraph checkpointed workflows within a day. Qdrant collection state is unaffected.
- All query-pipeline code can be inlined as async functions. `HorcruxState` reduces to a dataclass.
- PydanticAI agents are independent of either orchestrator and survive any rework.

No data is lost in a rollback; only orchestration code changes.
