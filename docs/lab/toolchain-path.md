# Toolchain Path

> A 10-phase weekend lab evaluating PydanticAI + LangGraph + Temporal + Qdrant
> + LiteLLM + LangSmith on a non-trivial RAG problem. Each phase introduces one
> tool or concept and tests it in isolation before composition.

This is the working journal for the lab. Read it linearly the first time. Each phase has the same shape:

- **Claim** — what we're testing in this layer.
- **Setup** — files added / services started.
- **Walkthrough** — the path from empty to working.
- **Exercise** — the test that proves the layer works (often a deliberate failure).
- **Gotchas** — the surprises worth flagging.
- **Got it** — the signal that the concept is internalised.

---

## Phase 0 — Repo skeleton + infrastructure

**Claim:** the lab is reproducible. A reader clones the repo, runs three commands, and is ready for Phase 1 in five minutes.

**Setup**
- `pyproject.toml` (Python 3.12, ruff config, deps).
- `Makefile` with `preflight`, `run`, `local`, `test`, `lint`, etc.
- `.env.example` (no real secrets) → copy to `.env` and add a real `ANTHROPIC_API_KEY`.
- `docker-compose.yaml` for Qdrant (named volume — no host filesystem state).
- `scripts/temporal-dev.sh` — wraps `temporal server start-dev`.
- `litellm_config.yaml` — model aliases.
- `horcrux/__init__.py` and `horcrux/config.py` — the typed config singleton.
- `tests/conftest.py` + `tests/unit/test_config.py` — smoke tests for the singleton.

**Pre-flight (host-side tools)**
Five tools must exist on PATH before `uv sync` is useful: `uv`, `docker`, `tesseract`, `temporal` (the CLI binary, separate from the `temporalio` Python SDK), and `litellm` (covered by `uv sync` once the Python deps install).

```bash
make preflight
```

prints a tick or cross for each. Fix any crosses, then:

**Walkthrough**
1. `cp .env.example .env` and put a real (or stub) `ANTHROPIC_API_KEY` in.
2. `uv sync` — installs Python deps.
3. `make test` — 5 unit tests on the config singleton; should be green.
4. `make local` — brings up Qdrant via docker-compose.
5. `make temporal` (in another terminal) — Temporal dev server.
6. `make proxy` (in a third terminal) — LiteLLM proxy.
7. Open `localhost:6333/dashboard` (Qdrant), `localhost:8233` (Temporal), `localhost:4000/ui` (LiteLLM). All three empty — that's correct.

**Exercise**
- `make lint` and `make test` both green.
- `make doctor` prints the resolved config with `anthropic_api_key` masked as `"**********"`.
- Click around all three UIs. They're empty — that's correct. Note where collections will land in Qdrant, where workflows will land in Temporal, where calls will land in LiteLLM.

**Gotchas**
- Temporal dev server is a single binary; production Temporal needs a Postgres + Cassandra cluster. The dev server is in-memory — restart it and event history is gone. Fine for the lab.
- Qdrant's docker-compose uses a *named volume* (`qdrant-data`) rather than a host bind mount. Docker manages the storage; we never see it on the host filesystem. Wipe via `docker compose down -v`.

**Got it** — the environment is reproducible, and you know where each service's UI lives before any code points at it.

---

## Phase 1 — Data cleansing (OCR)

**Claim:** the corpus is image-based PDF; raw text quality drives every downstream layer.

**Setup**
- `horcrux/models.py` — plain Pydantic models: `RawPage`, `Chapter`. *No agents yet.*
- `horcrux/ocr.py` — pymupdf + tesseract pipeline at 2× zoom.
- `scripts/test_pdf.py` — runs OCR on book 1 only (~520 pages, ~7 minutes on a laptop), dumps text to `data/raw/book_01.txt`.

**Walkthrough**
1. Drop your legally-obtained PDF into `data_lake/corpus.pdf`. `.gitignore` already excludes it; `data_lake/README.md` documents the convention.
2. Run `scripts/test_pdf.py`.
3. Watch a Rich progress bar tick through pages.

**Exercise**
- Open `data/raw/book_01.txt`. Skim a few pages. Find OCR errors — Hagrid's dialect mangled, italics dropped, page-numbers bleeding into body text, em-dashes turning into `—` or just `-` inconsistently.
- Decide what to clean. Strip page numbers? Yes — they confuse chunking. Normalise whitespace? Yes. Fix Hagrid's dialect? No — that's not OCR error, that's authorial style.

**Gotchas**
- `pymupdf` extracts embedded text first; only falls back to OCR when none exists. HP PDFs are image-based, so OCR runs every page.
- Tesseract at 1× zoom drops accuracy badly. 2× is the sweet spot — 4× doubles runtime for marginal gain.
- Memory: rendering a page at 2× zoom can hit 50MB. Don't fan all 3623 pages out at once — batch.

**Got it** — you've seen the raw material. You know the downstream chunker has to tolerate imperfect text.

---

## Phase 2 — Data modelling (chapters + chunking + character extraction)

**Claim:** structure emerges from raw text via three deterministic, non-LLM steps. Each step is independently testable.

**Setup**
- `horcrux/chapters.py` — regex on `CHAPTER ONE/TWO/...` markers; book boundaries from title pages.
- `horcrux/chunking.py` — semantic paragraph chunker. Sliding-window cosine similarity on sentence embeddings; cut at threshold 0.35; one-sentence overlap.
- `horcrux/characters.py` — canonical character list + alias dictionary. Substring match per chunk; populates `chunk.characters`.
- `tests/unit/` — fixtures with synthetic fantasy text (no copyright).

**Walkthrough**
1. Feed cleansed book 1 text into `detect_chapters` → list of `Chapter`.
2. For each chapter, `chunk_chapter(chapter)` → list of `ChapterChunk` (one per paragraph + one for the whole chapter).
3. For each chunk, `extract_characters(chunk.text)` → populates `chunk.characters: list[str]`.

**Exercise**
- Inspect chunk boundaries. Find ones that cut mid-thought — these are the threshold's failures.
- Tweak threshold to 0.25 (more aggressive cuts, smaller chunks). Then 0.50 (fewer cuts, larger chunks). Note how character density per chunk changes.
- Inspect the `characters` payload. Who got picked up consistently? Who got missed because their alias isn't in the canonical list?

**Gotchas**
- Sentence segmentation breaks on dialogue with internal punctuation (`"Don't!" he cried.`). Use `nltk.sent_tokenize` or `pysbd`, not naive `.split('.')`.
- Character aliases overlap: `"Tom"` matches both Tom Riddle and the bartender at the Leaky Cauldron. Disambiguation is hard; in this lab we accept some noise rather than building NER.
- Embedding sentences for similarity is *expensive* if done repeatedly. Cache embeddings per chapter on disk.

**Got it** — chunking is a tunable with measurable consequences. You've felt how threshold affects chunk shape and how alias-list quality affects relational query precision.

---

## Phase 3 — Embedding + Qdrant ingestion

**Claim:** the vector store is the retriever. Once data is in, query quality is bounded by what's been indexed.

**Setup**
- `horcrux/ingest.py` — bge-large-en-v1.5 model loader (sentence-transformers); Qdrant client; collection setup with payload indexes; batch upsert.
- Two collections: `hp_chapters`, `hp_paragraphs`. Same embedding model, different chunk granularity.

**Walkthrough**
1. Create collections with `vectors_config={size: 1024, distance: COSINE}` and payload indexes on `book_num` (INTEGER), `chapter_num` (INTEGER), `characters` (KEYWORD), `chunk_type` (KEYWORD).
2. Embed each chunk's text with the bge-large model (asymmetric embedding — passages get no prefix; queries get the `"Represent this sentence for searching relevant passages: "` prefix).
3. Upsert in batches of 64.

**Exercise**
- Open Qdrant dashboard. Verify both collections exist with correct point counts (book 1: ~17 chapters, ~600-1000 paragraph chunks).
- Click into individual points. Confirm payloads are populated (especially `characters`).
- Run a few searches by hand from a Python REPL using `httpx` against the Qdrant REST API. Same query against `hp_chapters` vs `hp_paragraphs`. Then add a `characters` filter (`must contain "Snape"`). See the result shapes differ.

**Gotchas**
- bge-large is *asymmetric*. Forgetting the query prefix at search time degrades precision noticeably — and silently. Wrap embedding so query and passage paths cannot be mixed up.
- Payload indexes must be created *before* upsert for them to apply efficiently. Adding an index post-hoc on a populated collection is slow.
- Cosine distance + normalised vectors → use `COSINE` in Qdrant *and* `normalize_embeddings=True` in sentence-transformers. Mismatch is a silent precision hit.

**Got it** — the retriever has personality. You can predict what each collection returns for a given query before running it.

---

## Phase 4 — Temporal (durability arrives)

**Claim:** durable execution earns its place by surviving a crash mid-pipeline without re-running completed work.

**Pre-phase ritual.** Before introducing Temporal, run your Phase 1-3 pipeline as one bare Python script on the **full** corpus. Halfway through (~minute 20), `Ctrl+C`. Now restart from scratch. Feel the pain. *That* is the problem Temporal solves.

**Setup**
- `horcrux/workflows.py` — `IngestWorkflow` (the orchestrator).
- `horcrux/activities.py` — `ocr_batch_activity`, `detect_chapters_activity`, `chunk_and_ingest_activity`.
- `horcrux/worker.py` — registers workflows + activities, connects to dev server.

**Walkthrough**
1. Refactor existing functions into Temporal activities — same logic, decorated with `@activity.defn`.
2. Workflow calls activities via `workflow.execute_activity(...)` with an explicit retry policy and `start_to_close_timeout`.
3. Run the worker (`uv run python -m horcrux.worker`).
4. Trigger the workflow (`uv run python -m horcrux.main ingest`).

**Exercise — the crash test**
1. Start a fresh ingest run. Let it complete batches 0-3.
2. `Ctrl+C` the worker.
3. Confirm in Temporal UI that the workflow is "running, no worker."
4. Restart the worker.
5. Watch the workflow resume — batches 0-3 do **not** re-execute. Batch 4 starts fresh.

**Gotchas**
- **Determinism in workflows.** No `datetime.now()`, `random`, file I/O, or `await asyncio.sleep` *in workflow code*. Use `workflow.now()`, `workflow.uuid4()`, `workflow.sleep()`. All non-deterministic work goes into activities.
- Activity timeouts must accommodate worst-case OCR pages (~8 seconds for hard pages); 60s is a safe bound.
- Retry policy: `non_retryable_error_types=["FileNotFoundError"]` — don't retry config errors.
- The Python SDK's worker uses sandbox mode for workflows by default. Some imports (anything with side-effects at import time) trip the sandbox. Solution: `passthrough_modules=[...]` for known-safe modules.

**Got it** — durable execution is no longer a phrase. You understand workflow vs activity, replay vs execute, why determinism rules exist.

---

## Phase 5 — PydanticAI primer (typed agents)

**Claim:** every LLM call is a boundary. PydanticAI makes that boundary type-safe at the cost of one validation retry on bad model output.

**Setup**
- Extend `horcrux/models.py` with `QueryIntent`, `ResearchPlan`, `CandidateScore`, `Finding`, `ResearchReport`.
- `horcrux/agents.py` — `intent_agent`, `planner_agent`, `relevance_agent`, `synthesis_agent`. Each declares a `result_type` and a system prompt.

**Walkthrough**
1. Wire `intent_agent` first. `Agent("haiku", result_type=QueryIntent, system_prompt=...)` — model name is the LiteLLM alias, configured in Phase 7.
2. Run on a sample query. The result is a typed `QueryIntent`, never a string.
3. Add the other three agents in turn. Each tested in isolation before moving on.

**Exercise**
- Deliberately weaken the `Field(description=...)` on `confidence: float` — remove the "must be between 0 and 1" guidance. Run again. Watch the model occasionally return `confidence: 1.4` or string `"high"` → Pydantic raises `ValidationError` → PydanticAI feeds the error back to the model → it self-corrects.
- Restore the description. Validation retries drop to near zero.

**Gotchas**
- `Field` descriptions are part of the *tool schema sent to the model*. Better descriptions = fewer retries. This is the highest-leverage prompt engineering you'll do.
- PydanticAI uses Anthropic's tool-call mechanism under the hood, not "please return JSON." This is structural — the model can't fail to return *some* JSON; it can only fail to return *valid* JSON for your schema.
- `result_type=Foo` does not make Foo the *only* output the model can produce. It makes Foo the *expected shape*. The model can still produce text alongside (which PydanticAI ignores in `result.data`).

**Got it** — every LLM call has a typed contract. The retry mechanism is automatic and cheap.

---

## Phase 6 — LangGraph one node at a time

**Claim:** the query pipeline is a state machine. Conditional edges and parallel fan-out are first-class. The graph is dry-runnable without LLMs.

**Setup**
- `horcrux/graph.py` — `HorcruxState` TypedDict, node functions, edges, compilation.
- `horcrux/retrieval.py` — wraps Qdrant + RRF merge as a callable.

**Walkthrough — incremental**
1. Register `classify_intent` only. Add a conditional edge to `END` based on a confidence stub.
2. Compile. Call `graph.invoke({"query": "..."})`. Watch it run one node and stop.
3. Add `plan_research`, `retrieve`, `score_candidates`, `synthesise` one at a time, each with their conditional edges. Re-compile and re-test after each.
4. Final graph matches the design doc diagram.

**Exercise**
- **Dry-run routing.** Don't call the graph end-to-end yet. Build state manually:
  ```python
  fake_state = {"query": "x", "intent": QueryIntent(..., confidence=0.6), "retry_count": 0}
  next_node = route_after_intent(fake_state)
  assert next_node == "classify_intent"   # confidence too low → retry
  ```
- Conditional routing tests need *zero* LLM calls. Build a dozen of these to lock in the graph's behaviour before you trust it on real queries.

**Gotchas**
- LangGraph nodes return *partial* state, not full state. `return {"intent": some_intent}` merges that key into existing state. Returning a full state dict overwrites everything — common mistake.
- Parallel fan-out happens within a single node via `asyncio.gather`. LangGraph itself processes nodes serially unless you use `Send` for true parallel branches. For this lab, in-node `asyncio.gather` is enough.
- `END` is `langgraph.graph.END`, not the string `"END"`. Conditional functions must return `END` or a node name *string* — anything else is a runtime error.

**Got it** — the graph is the control flow. You can reason about routing in isolation from agent behaviour.

---

## Phase 7 — LiteLLM proxy + LangSmith

**Claim:** routing and observability are passive layers — they wrap the existing system without code changes.

**Setup**
- `litellm_config.yaml` — model aliases (`haiku`, `sonnet`) mapping to Anthropic model IDs, with response caching enabled.
- `.env` adds `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY=...`, `LANGCHAIN_PROJECT=horcrux`.

**Walkthrough**
1. Start LiteLLM proxy: `litellm --config litellm_config.yaml --port 4000`.
2. Open `localhost:4000/ui`. Empty traces, empty spend.
3. Update PydanticAI agent definitions to point at the LiteLLM proxy via `OPENAI_BASE_URL=http://localhost:4000`. Model strings become aliases (`"haiku"`, `"sonnet"`).
4. Set the `LANGCHAIN_*` env vars.
5. Run a full query end-to-end.

**Exercise**
- Watch the same query in **two** UIs simultaneously:
  - LiteLLM (`localhost:4000/ui`) — flat list of LLM calls, per-call cost and latency.
  - LangSmith (`smith.langchain.com`) — graph trace tree showing nodes, edges, and LLM calls nested inside.
- Run the *same* query twice. The second run hits LiteLLM's response cache for any LLM call with identical inputs. Note the latency drop.

**Gotchas**
- LiteLLM's Anthropic adapter sometimes lags new Anthropic features (cache_control, extended thinking). Verify caching headers actually flow through end-to-end if you rely on them.
- LangSmith picks up PydanticAI calls *and* LangGraph node executions automatically when env vars are set. No code change. If traces don't appear, check the env vars are loaded *before* the first import of `langgraph` or `pydantic_ai`.
- `LANGCHAIN_PROJECT` defaults to `"default"`. Always set it explicitly so traces don't pollute other projects.

**Got it** — you understand exactly what each tool sees. LiteLLM sees calls; LangSmith sees the graph; PydanticAI sees the typed contracts. Three views, one execution.

---

## Phase 8 — Clarification interrupt (human-in-the-loop)

**Claim:** LangGraph nodes can pause the graph, surface state to a human, and resume on response. The graph itself becomes interactive.

**Setup**
- Extend `QueryIntent` with `quality: Literal["good", "needs_clarification"]` and `clarification_question: str | None`.
- New node `ask_clarification` that calls `interrupt(question)`.
- New node `merge_query` that combines the original query with the user's reply and routes back to `classify_intent`.
- SQLite checkpointer: `SqliteSaver.from_conn_string("horcrux.db")`.

**Walkthrough**
1. Update intent agent's system prompt: "If the query is too vague to retrieve usefully (no named entities, ambiguous scope, no concrete claim), set `quality='needs_clarification'` and write one specific follow-up question."
2. Compile graph with the checkpointer.
3. CLI loop:
   ```python
   thread_id = str(uuid4())
   config = {"configurable": {"thread_id": thread_id}}
   result = graph.invoke({"query": q}, config=config)
   while interrupt := result.get("__interrupt__"):
       reply = Prompt.ask(f"[yellow]> {interrupt[0].value}[/yellow]")
       result = graph.invoke(Command(resume=reply), config=config)
   ```

**Exercise**
- Ask a vague query: `"tell me about Snape"`.
- Graph routes through `classify_intent` → `ask_clarification` → pauses.
- A Rich-styled panel appears: `> Which aspect of Snape — his loyalty, his teaching style, his backstory?`
- Reply: `"his loyalty across the series"`.
- Graph resumes, merges the reply, re-classifies as analytical, fans out to chapter retrieval, synthesises.
- Cap clarification at 2 rounds — prevents infinite loops on truly ambiguous queries.

**Gotchas**
- Without a checkpointer, `interrupt()` fails — the graph has nowhere to persist the paused state.
- Each query must use a unique `thread_id`. Re-using one across different queries replays the wrong checkpoint.
- The interrupted state is *the entire `HorcruxState`* at the moment of pause, not just the question. Resume merges only what `Command(resume=value)` provides.

**Got it** — interrupts as a primitive. You can imagine using this for approval gates, slot filling, "did you mean X or Y?" disambiguation.

---

## Phase 9 — Stretch experiments

Once the system runs end-to-end, deliberately break it to learn:

1. **Drop `hp_chapters`.** Re-run an analytical query (`"trace Snape's loyalty across the series"`). The synthesis quality drops noticeably — paragraphs alone don't capture chapter-scale narrative arcs. You've measured *why* the chapter collection earned its keep.

2. **Threshold sweep.** Re-chunk paragraphs at 0.25 vs 0.50. Re-ingest. Run the same query against each. Diff the answers. Visceral evidence that chunking is a hyperparameter.

3. **Model swap, no code change.** In `litellm_config.yaml`, change `sonnet`'s underlying model from `claude-sonnet-4-6` to `claude-opus-4-7`. Restart the proxy. Re-run. Did conviction calibration shift? (Larger models tend to spread more across the rubric.)

4. **Disable strict RAG.** Remove the `min_length=1` constraint on `Finding.source_ids`. Re-run a query you know your retriever struggles with. Watch the synthesis agent quietly fill gaps from training knowledge. Compare the report to the strict-RAG version. The difference is the *learning loop you preserved* by holding the line on grounding.

5. **Swap LangSmith for self-hosted Langfuse.** Swap two env vars. Stand up Langfuse via docker-compose. The trace tree looks slightly different but the data is the same. You've now used both observability stacks.

Each of these is a 30-minute experiment. They turn the lab from "I built it" into "I understand which decisions mattered."

---

## Closing notes

The lab is bounded. It is not production. Specifically out of scope:
- Multi-tenant isolation, auth, rate limiting at the application layer.
- Production-grade Temporal cluster (we use the dev server).
- Cost guardrails beyond the soft caps the LiteLLM proxy provides.
- Retraining or fine-tuning the embedding model.
- Web UI of any kind.

In scope and demonstrated:
- End-to-end RAG over a literary corpus, ~3600 pages.
- Strict-RAG grounding enforced at three layers (prompt / schema / runtime).
- Crash-resilient ingest via Temporal, demonstrated by a deliberate kill-restart test.
- Typed LLM I/O at every boundary, validated by PydanticAI.
- Conditional graph routing with human-in-the-loop interrupts via LangGraph.
- Provider-agnostic routing via LiteLLM (single-provider in this build, swap-ready by config).
- Live observability via LangSmith.

Time-box: one weekend. Scope: deliberate. Ship state: runnable end-to-end.
