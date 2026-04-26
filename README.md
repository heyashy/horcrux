# Horcrux

> A deep-research RAG agent over a literary corpus. Ask the corpus a
> question; watch a planner decompose it into focused sub-questions, see
> sub-queries run in parallel, get back a coherent answer with conviction-
> scored citations grounded only in what the corpus actually contains.

Weekend lab evaluating six tools side-by-side on a non-trivial RAG problem:
**PydanticAI** · **LangGraph** · **Temporal** · **Qdrant** · **LiteLLM** · **LangSmith**.

Built collaboratively with AI coding agents (Claude Code) under explicit
engineering discipline — see [How it was built](#how-it-was-built).

---

## What it does

Three modes, all running against the same indexed corpus:

| Mode | Command | Use when |
|---|---|---|
| **Answer** — single-shot, fast. | `make answer Q="..."` | Focused factual questions. |
| **Chat** — multi-turn, history-aware. | `make chat` | Follow-ups that reference prior turns ("what about his motive?"). |
| **Research** — planned, parallel, with visible reasoning. | `make research Q="..."` | Multi-faceted questions whose answer spans many books. |

The research mode is the one to look at first. A demo transcript
(*"tell me about Snape's story arc"*):

```
▶ Planning…
   ↳ What is Snape's role in the early books and his relationship with Harry?
   ↳ What did Snape's memories in The Prince's Tale reveal about his past
     and his loyalty to Dumbledore?
   ↳ Why did Snape kill Dumbledore and what were his motivations?
   ↳ How did Harry's perception of Snape change after learning the truth?

▶ Sub-queries (parallel)
   ⠋ What is Snape's role in the early books…              …running
   ⠋ What did Snape's memories in The Prince's Tale…       …running
   ⠋ Why did Snape kill Dumbledore…                        …running
   ⠋ How did Harry's perception of Snape change…           …running
   ✓ Snape's role in early books          10 hits, 7 cited, conviction 5/5, 29.0s
   ✓ Why Snape killed Dumbledore           10 hits, 4 cited, conviction 4/5, 30.1s
   ✓ Memories in The Prince's Tale         10 hits, 5 cited, conviction 3/5, 31.8s
   ✓ Harry's perception change             10 hits, 5 cited, conviction 3/5, 36.1s

▶ Synthesising final report…

  Severus Snape's story arc spans all seven Harry Potter novels, moving
  from apparent villainy to a posthumous revelation of hidden heroism…
  [coherent multi-paragraph answer drawing on all four sub-findings]

  conviction: 4/5 (high)  citations: 19  gaps: 3
```

19 citations spanning all 7 books. Final conviction is **4/5, not 5/5** —
bounded by the weakest sub-finding, exactly the calibration pattern good
research agents need. Three honest gaps surfaced (the parts the corpus
didn't cover), not papered over with parametric guesses.

The same question via single-shot returns a shallower answer with 9
citations and misses the Lily-love and planned-killing reveals. Multi-step
planning isn't ceremony — it's the only way to get arc-shaped questions
right without throwing the whole corpus into a million-token context.

---

## Why this lab exists

Most RAG demos hand-wave the parts that actually matter — chunking
strategy, retrieval routing, grounding discipline, durable ingest,
multi-step planning, conversational memory, calibration. This lab takes
one bounded problem (a 3,600-page literary corpus, OCR'd from a personal
PDF) and forces each of those concerns into its own observable layer.

The output isn't a product. It's an opinion on the toolchain — what
each tool earned, what was friction, where the boundaries land, and which
production patterns *generalise* vs which only worked because the corpus
was famous.

What makes it genuinely interesting as a portfolio artefact:

- **It works.** All six tools wired up; three CLI surfaces; 203 unit
  tests passing; ~22 documented findings; 9 ADRs.
- **The agent's reasoning is visible.** Research mode streams every
  step (plan, parallel sub-queries, fusion, synthesis) so you can see
  the system *think*. That's the demo — opacity reads as magic at
  best, broken at worst.
- **It surfaces its own failure modes.** [`docs/lab/findings.md`](docs/lab/findings.md)
  catalogues 22+ empirical findings, including the ones where my
  hypotheses were wrong. The catalog is a portfolio in itself.
- **Honest about what it isn't.** [ADR-0008](docs/adr/pending/ADR-0008-qdrant-for-the-lab-opensearch-for-production.md)
  explicitly says "Qdrant for the lab, OpenSearch for production
  where OS is already in the stack." Not a sales pitch.

---

## Architecture

Three graphs and one durable workflow, composing into one system.

### Ingest pipeline (Temporal-driven, durable)

```mermaid
flowchart TD
    PDF[/PDF corpus/] --> W["IngestOCRWorkflow<br/><i>Temporal — durable</i>"]
    W --> O["ocr_batch_activity × N"]
    O --> M[merge_batches]
    M --> CD[chapter detection<br/>via PDF outline]
    CD --> CH[semantic chunking<br/>+ character extraction]
    CH --> EM[bge-large embedding]
    EM --> QC[("hp_chapters")]
    EM --> QP[("hp_paragraphs<br/>+ characters payload")]
```

OCR earned Temporal because the failure profile demanded it: 30+ minutes
of slow IO-bound work, partial-progress recovery matters, kill-and-resume
behaviour is testable end-to-end. Embedding doesn't earn Temporal
(deterministic + idempotent + single-process); ran raw.

### Retrieval graph (LangGraph, 4-way modality + granularity hybrid)

```mermaid
flowchart TD
    Q([Query]) --> RW["rewrite<br/>fuzzy-correct typos vs<br/>corpus vocabulary"]
    RW --> PD[paragraph_dense<br/>Qdrant ANN]
    RW --> CD[chapter_dense<br/>Qdrant ANN]
    RW --> PB[paragraph_bm25<br/>in-memory rank-bm25]
    RW --> CB[chapter_bm25<br/>in-memory rank-bm25]
    PD --> F[fuse<br/>Reciprocal Rank Fusion]
    CD --> F
    PB --> F
    CB --> F
    F --> R([RetrievalResult])
```

Two axes of hybridity: **granularity** (chapter + paragraph chunks) and
**modality** (dense semantic + sparse keyword). Catches both paraphrase
matches *and* rare-keyword matches. Discovered the modality gap when a
real query ("name the conjunctivitis spell") returned nothing under
dense-only retrieval — a rare keyword the embedder compressed away.
See [Finding 21](docs/lab/findings.md) (`Ctrl+F` for "Finding 21").

### Research graph (LangGraph with `Send` fan-out)

```mermaid
flowchart TD
    Q([Research question]) --> P["plan<br/>Haiku — decompose<br/>into 1-5 sub-questions"]
    P -.Send.-> S1["subquery 1<br/>retrieve + synthesise"]
    P -.Send.-> S2["subquery 2<br/>retrieve + synthesise"]
    P -.Send.-> S3["subquery 3<br/>retrieve + synthesise"]
    S1 --> A["aggregate<br/>Sonnet — coherent answer<br/>across sub-findings"]
    S2 --> A
    S3 --> A
    A --> RR([ResearchReport])
```

`Send`-based fan-out so each sub-query is its own node event in
`graph.astream` — that's what gives research mode its visible-reasoning
UX. Strict-RAG enforces at every synthesis layer (sub-syntheses + the
aggregator); planner is parametric by design, with the limits called
out explicitly in [ADR-0009](docs/adr/pending/ADR-0009-research-mode-planner-aggregator.md).

### Strict-RAG enforcement

Three layers per LLM call, every call:

1. **System prompt** forbids parametric knowledge.
2. **Schema** — `Finding.source_ids` has `min_length=1`. Pydantic
   validation error → PydanticAI auto-retry.
3. **Runtime check** — every cited passage number must be in
   `[1..N]`. Out-of-range → `ValueError`. Caught a real failure live:
   Sonnet dropped two characters from a UUID; the runtime check rejected
   the fabricated citation; the next session switched to numbered
   citations to remove the failure mode at its root.

Full system design: [`docs/horcrux_system_design.md`](docs/horcrux_system_design.md).
Phase walkthrough: [`docs/lab/toolchain-path.md`](docs/lab/toolchain-path.md).
Architecture decisions: [`docs/adr/`](docs/adr/).

---

## How it was built

Honest framing: this lab was built collaboratively with AI coding agents
(Claude Code, Opus 4.7) over a series of focused sessions. The agents
didn't replace engineering rigor — **the rigor is what made the agent
collaboration productive**.

### What that looked like in practice

- **Phase-by-phase, with verification at each layer.** Phase 4
  (retrieval) was built and verified standalone before Phase 5
  (synthesis) wired LLMs on top. Within Phase 5, the dense path was
  verified before BM25 was added. Within research mode, a stub planner
  + stub aggregator validated the streaming UX before the real Haiku /
  Sonnet agents replaced them. Every layer is debuggable in isolation.

- **ADRs for non-trivial decisions.** Nine ADRs in
  [`docs/adr/pending/`](docs/adr/pending/), each captured in the same
  structured form (context / decision / alternatives considered /
  consequences / rollback). Examples: why Temporal *and* LangGraph
  (different failure profiles), why LiteLLM proxy (one-line model
  swaps), why Qdrant for this lab but OpenSearch for production, why
  `Send` for research mode but `asyncio.gather` for retrieval.

- **A findings catalog as a living artefact.**
  [`docs/lab/findings.md`](docs/lab/findings.md) — 22+ entries, each
  in the same Symptom / Root cause / Fix / Lesson template. Documents
  the cases where my hypothesis was *wrong* and the agent or the live
  test corrected me, alongside the cases where the agent's first
  attempt was wrong and human review caught it. The catalog is itself
  evidence of how the collaboration worked: not "AI wrote this," but
  "AI and human iterated, with empirical evidence settling
  disagreements."

- **TDD where it adds signal.** Pure functions (chunking thresholds,
  RRF fusion, fuzzy matching, schema invariants) got tests-first.
  Glue code (wiring an agent into a graph) skipped TDD where tests
  would just mirror the wiring. Test count: 203, broken across unit
  / integration / smoke per the standard pyramid.

- **Layer-by-layer commit cadence.** 21 commits over the lab's
  lifetime; each is one coherent step (a phase, a hotfix, a refactor).
  `git log --oneline` reads like a development diary. No giant
  "initial commit." [Change log](docs/log/) records what shipped each
  day.

- **Specialised AI agents for specific roles.** Code-review agents for
  reviewing diffs, exploration agents for navigating large codebases,
  general-purpose agents for end-to-end implementation. The
  multi-agent setup was used most in early phases (research +
  exploration); later phases ran direct because the codebase shape
  was clear.

### Where the agent helped most

- **Mechanical refactors** (the subpackage reorganisation) — fast,
  consistent, with full test coverage check at the end.
- **Boilerplate-heavy modules** (Pydantic models, LangGraph state
  machines, Rich rendering) — the patterns are stable; agents fill
  them in correctly given clear specs.
- **End-to-end verification** — agents are good at running smoke
  tests, parsing output, surfacing problems.
- **Documentation discipline** — once the ADR/findings template was
  set, the agents kept it consistent.

### Where the human still drove

- **Architectural decisions.** Every ADR's "Decision" section is the
  human's call after considering the alternatives the agent surfaced.
  When the agent proposed using Qdrant native sparse vectors for the
  BM25 layer, the human looked at corpus size and said "in-memory
  rank-bm25 is the right call here" — and was correct.
- **Catching parametric leakage.** The observation that the planner
  was using HP knowledge to write sub-questions ("The Prince's Tale"
  appearing as a sub-question name proves the planner *knew*
  parametrically that this chapter exists) came from the human reading
  the live output. The agent would have shipped without flagging it.
  Documented as a known-good limitation in ADR-0009.
- **Calibration judgement.** Whether the answer was "really" 5/5 or
  "actually" 4/5 — the agent biased high; the human looked at evidence
  and pushed for proper calibration. F20 documents the calibration
  drift; the planner-aggregator architecture organically resolved it.
- **Deciding when to stop.** Saying "this is complete; the next step
  is weeks of optimisation" is a judgement call about scope, not a
  technical question. The lab's discipline includes knowing when not
  to keep building.

### What this lab demonstrates about AI-augmented engineering

Three things, in order of importance:

1. **Discipline scales human-AI collaboration.** TDD, ADRs, typed
   contracts, and a findings catalog gave the agents a structure to
   work within. Without that, AI output would be unverified and
   non-composable. With it, the agent shipped working code at a
   consistent quality bar.

2. **Layered verification beats end-to-end faith.** When the
   conjunctivitis-curse query failed, we didn't ask the model to "try
   harder" — we identified that BM25 was needed (Finding 21), built
   the in-memory index, fused it via LangGraph, and verified the
   specific failing query now succeeded. Agents are great at
   mechanical work; the *measurement loop* is what makes the work
   correct.

3. **The findings catalog is the lab's most valuable output.** Anyone
   can clone an AI-built RAG and demo a happy path. Few have a
   document explaining where their system silently fails and why.
   That's what `docs/lab/findings.md` is — a record of every gotcha,
   most of which were discovered the hard way (by running the system
   and watching it produce a wrong answer).

---

## Engineering discipline

Repo conventions:

- **`docs/adr/pending/`** — 9 ADRs. Each has context / decision /
  alternatives / consequences / rollback. ADRs move to `done/` when
  the change ships.
- **`docs/lab/findings.md`** — 22+ findings. Each has Symptom / Root
  cause / Fix / Lesson. Includes a meta-section grouping them by how
  they were caught (real-output inspection vs architectural
  reflection vs hard-limit hits).
- **`docs/log/`** — append-only daily change log. Records what
  shipped each day, what ADRs progressed, what tests landed.
- **`docs/horcrux_system_design.md`** — the full system design as a
  single readable doc.
- **`docs/lab/toolchain-path.md`** — phase-by-phase walkthrough,
  framed as a learning path.
- **`docs/lab/character-discovery-deep-dive.md`** — review-style
  walkthrough of the three-tier character discovery system.

Code conventions:

- **Strict RAG enforced at three layers** (prompt + schema + runtime).
- **Typed contracts everywhere** — every LLM I/O is a Pydantic model;
  validation retry is automatic via PydanticAI.
- **Centralised config** — `pydantic-settings` singleton in
  `horcrux/config.py`. No `os.getenv` in core logic. Fail-fast
  validation at import time.
- **Subpackage layout** by concern — `corpus/`, `retrieval/`,
  `agents/`, `research/`, `chat/`. You can read the architecture
  from `ls horcrux/`.
- **Reproducible env** — `pyproject.toml`, `Makefile`,
  `docker-compose`, `.env.example`. `make preflight` checks every
  external dependency in one go.
- **203 unit tests + integration + smoke**. Lint-clean per
  per-file-ignores documented in `pyproject.toml`.

---

## What each tool earned

| Tool | Role | Why it stayed | Fine print |
|---|---|---|---|
| **PydanticAI** | Typed boundary at every LLM call | Schema-as-contract; auto-retry on validation failure; `result_type` makes hallucinated structure impossible. | — |
| **LangGraph** | Orchestration for retrieval + research graphs | Conditional edges, parallel fan-out via `Send`, native streaming via `astream`. The streaming-debug events are what make research mode's visible reasoning work. | First place that actually earned `Send` over `asyncio.gather`: when each branch needs to be independently visible in stream events. |
| **Temporal** | Durable ingest workflow | Crash mid-OCR, restart, resume from last completed batch — proven by deliberate kill test on a 22-min run. | Dev-server only (in-memory, single-binary). Production cluster is out of scope. |
| **Qdrant** | Vector storage | Two collections × payload-filtered ANN. Clean API for the lab's volume. | For production with existing OpenSearch, **use OpenSearch hybrid search instead** — see [ADR-0008](docs/adr/pending/ADR-0008-qdrant-for-the-lab-opensearch-for-production.md). The lab uses Qdrant deliberately because the lab evaluates Qdrant. |
| **LiteLLM** | Model router (proxy at `localhost:4000`) | One-line model swap via YAML; provider-agnostic; spend tracking; in-memory response cache during dev. | — |
| **LangSmith** | Observability | Full graph trace per query; visualises conditional routing as a tree. Auto-instrumented via two env vars. | Optional — set `LANGCHAIN_API_KEY=` to enable. |

Three additional libraries earn quiet credit:

- **`bge-large-en-v1.5`** for dense embeddings (Qdrant gRPC + cosine).
- **`rank-bm25`** for the in-memory sparse index. ~6ms queries on
  5,500 chunks; chosen over Qdrant native sparse for lab-scale
  pragmatism (see ADR-0008's BM25 alternative section).
- **`fastcoref`** + spaCy NER for character-discovery Tier 1.

---

## Running it

You will need a legally-obtained PDF of the corpus you want to research.
**Nothing copyrighted is shipped with this repo.**

### Prerequisites

| Tool | Why | Install |
|---|---|---|
| **`uv`** | Python package manager. | <https://docs.astral.sh/uv/> |
| **`docker`** | Runs the Qdrant container. | <https://docs.docker.com/engine/install/> |
| **`tesseract`** | OCR engine for the ingest pipeline. | `apt install tesseract-ocr` (Ubuntu) · `brew install tesseract` (macOS) |
| **`temporal`** | Bundled dev server (frontend + history + matching + visibility, all in one binary). | `curl -sSf https://temporal.download/cli.sh \| sh` — then add `~/.temporalio/bin` to PATH |
| **`litellm[proxy]`** | Model-routing proxy. Installed by `uv sync`. | covered by `uv sync` |

```bash
make preflight     # six green ticks confirms everything's wired
```

### One-time setup

```bash
uv sync                                          # python deps
cp .env.example .env                             # then add ANTHROPIC_API_KEY, LITELLM_MASTER_KEY
cp /path/to/corpus.pdf data_lake/corpus.pdf      # your legally-obtained PDF
make test                                         # 203 unit tests should pass
```

### Three terminals

```bash
# Terminal 1 — Qdrant
make local

# Terminal 2 — LiteLLM proxy
make proxy

# Terminal 3 — Temporal dev server (only if running ingest)
make temporal
```

### Ingest (one-time, ~45 min for a 3600-page corpus)

```bash
make worker        # in another terminal, start the worker
make ingest        # trigger the workflow
                   # safe to kill and restart; resumes from last completed batch.
                   # watch progress at localhost:8233 (Temporal UI)
```

Then build the gold-tier artefacts:

```bash
make chapters      # raw_pages.json → chapters.json
make chunks        # chapters + alias-dict → chunks.json
make embed         # chunks.json → Qdrant
```

### Query

```bash
make answer Q="who killed Cedric Diggory"           # single-shot
make chat                                            # multi-turn with history
make research Q="tell me about Snape's story arc"   # multi-step with visible reasoning
make search Q="..."                                  # retrieval only (no synthesis)
```

The `chat` REPL has slash commands: `/research`, `/trace` (re-render
last research run), `/history`, `/clear`, `/help`.

---

## Repository layout

```
horcrux/                      # the package — concern-grouped subpackages
├── config.py                 # pydantic-settings singleton
├── models.py                 # Pydantic types — single source of truth
├── main.py / worker.py       # entry points
├── corpus/                   # PDF → chunks ready to embed
├── retrieval/                # finding relevant chunks (4-way hybrid)
├── agents/                   # LLM-driven synthesis (single-shot, planner, aggregator)
├── research/                 # multi-step research mode with streaming UX
├── chat/                     # multi-turn conversation primitives
└── pipelines/ocr/            # Temporal workflow for OCR

scripts/                      # CLI entry points (answer / chat / research / search / build_*)
tests/{unit,integration,smoke}/   # 203 unit tests; integration + smoke against live infra
docs/
├── adr/{pending,done}/       # 9 ADRs
├── lab/                      # findings.md, toolchain-path.md, deep-dives
└── log/                      # append-only daily change log
data_lake/                    # gitignored — corpus PDF lives here
data/                         # processed artefacts (chapters / chunks / aliases) — gitignored
```

---

## What's deliberately out of scope

- **Web UI** — the CLI is the product.
- **Production deployment** — no Terraform / CI / cloud. Local-only lab.
- **Multi-tenant auth, rate limiting at the app layer** — out of frame.
- **Embedding model fine-tuning** — bge-large off-the-shelf.
- **Long-context single-shot mode** — the lab's whole point is that
  you can do better than throwing 1M tokens at a model. See ADR-0009.
- **SQLite checkpointer for clarification interrupts** — designed in
  ADR-0003 but not built; queued as the next phase if the lab
  resumes.
- **Grounded planner** (planner reads retrieved snippets before
  decomposing) — would extend the lab to private corpora where the
  planner can't lean on training knowledge. Discussed in ADR-0009.
- **Query optimisation, context optimisation, planner optimisations**
  — each is weeks of work; deferred.

---

## Honest scoping

A long weekend, deliberately. The point was to evaluate whether these
six tools compose well on a real-shaped problem — not to ship a SaaS.

If something took longer than planned, the lab is honest about it —
that's what [`docs/log/`](docs/log/) and [`docs/lab/findings.md`](docs/lab/findings.md)
record. Most of the value here isn't in the artefact's polish; it's
in the documentation of the *path*.

The current state:

- **Phases shipped:** OCR ingest, chapter detection, character
  discovery (3-tier), semantic chunking, dense embedding, hybrid
  retrieval (granularity + modality), single-shot synthesis,
  conversational chat, multi-step research with streamed reasoning.
- **9 ADRs** documenting non-trivial decisions.
- **22+ findings** documenting empirical lessons from real runs.
- **203 unit tests** + integration + smoke.
- **21 commits** over the lab's life, each one a coherent step.

The next reasonable additions (clarification interrupts, grounded
planning, query rewriting via LLM, learned re-ranking, contextual
chunking) are each their own multi-week project. Stopping here is the
disciplined call.

---

## Reading the lab

If you have an hour:

1. **Watch a research run** — clone, set up, `make research Q="..."`. See the streaming UX in action.
2. **Read [`docs/horcrux_system_design.md`](docs/horcrux_system_design.md)** — the full architectural picture in one document.
3. **Skim [`docs/lab/findings.md`](docs/lab/findings.md)** — 22+ failure modes documented. The Snape arc, the conjunctivitis curse, the UUID transcription bug — all have entries.
4. **Pick one ADR** — [ADR-0001](docs/adr/pending/ADR-0001-dual-orchestrator-architecture.md) (Temporal vs LangGraph), [ADR-0008](docs/adr/pending/ADR-0008-qdrant-for-the-lab-opensearch-for-production.md) (Qdrant vs OpenSearch), or [ADR-0009](docs/adr/pending/ADR-0009-research-mode-planner-aggregator.md) (research-mode design) — to see how decisions were structured.
5. **Read [`docs/log/`](docs/log/)** chronologically — three days of building, including the dead-ends.

If you have ten minutes: just run `make research Q="tell me about Snape's story arc"` and watch the agent reason. That's the demo.
