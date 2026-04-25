# CLAUDE.md — Horcrux

> Engineering charter for the Horcrux lab.
> **Authoritative for AI agents and human contributors.** All work must follow these rules.

---

## What this is

Weekend lab evaluating six tools side-by-side on a non-trivial RAG problem:
**PydanticAI · LangGraph · Temporal · Qdrant · LiteLLM · LangSmith.**

- **CLI app**, no web UI. The console is the product (rendered with Rich).
- **Local-only**: docker-compose + Temporal dev server + LiteLLM proxy.
  No AWS, no cloud deploy, no Terraform.
- **Time-boxed**: production process discipline (ADRs, logs, tests) applies;
  cloud deployment / multi-tenant / SLA rules don't.

Authoritative documents:

- [`docs/horcrux_system_design.md`](docs/horcrux_system_design.md) — full design.
- [`docs/adr/MANIFEST.md`](docs/adr/MANIFEST.md) — index of decisions.
- [`docs/lab/toolchain-path.md`](docs/lab/toolchain-path.md) — phased walkthrough.
- [`README.md`](README.md) — recruiter-facing landing page.

If a CLAUDE.md rule and a more recent ADR conflict, **the ADR wins** — open
an ADR to update this file.

---

## Stack — what each tool earns

| Tool | Role | ADR |
|---|---|---|
| **PydanticAI** | Typed boundary at every LLM call; auto validation retry on bad output. | ADR-0001 |
| **LangGraph** | Query-pipeline orchestration: conditional routing, parallel retrieval, human-in-the-loop interrupts. | ADR-0001, ADR-0003 |
| **Temporal** | Durable ingest workflow; survives crash mid-OCR via event-history replay. | ADR-0001 |
| **Qdrant** | Vector storage; two collections at different granularities. | ADR-0001 |
| **LiteLLM** | Model router (proxy at `localhost:4000`); provider-agnostic, swap-by-config. | ADR-0002 |
| **LangSmith** | Live observability; auto-instrumented via env vars. | ADR-0001 |
| **`pydantic-settings`** | Typed config singleton (`horcrux/config.py`). | ADR-0004 |
| **Rich** | CLI rendering and structured logging via `RichHandler`. | — |

---

## Architecture — non-negotiables

These are encoded as schema invariants, runtime checks, or process gates.
Do not weaken them without an ADR.

### Strict-RAG grounding (ADR-0001)
Synthesis answers come **only** from retrieved passages. Three layers of enforcement:
1. **Prompt** — `synthesis_agent` system prompt forbids parametric knowledge.
2. **Schema** — `Finding.source_ids` has `min_length=1`.
3. **Runtime check** — every `source_id` must exist in `scored_candidates` passed to the agent.

Model bleed (LLM filling gaps from training data) destroys the feedback loop
that makes the lab worth running. Hold the line.

### Centralised config (ADR-0004)
- **No `os.getenv` in core logic.** Use `from horcrux.config import settings`.
- Adding a new external service = a new `BaseSettings` block in `horcrux/config.py`.
- Required fields have no default — `Settings()` raises on import if missing.
- Secrets are `SecretStr`. Never `print(settings)` in code that hits logs.

### Two-collection chunking (ADR-0001)
- Only `hp_chapters` and `hp_paragraphs`. **No `hp_entity_mentions` collection.**
- Relational queries are served by a `characters: list[str]` payload filter on
  `hp_paragraphs`. Populate this at ingest via the canonical character list.

### LiteLLM proxy as model gateway (ADR-0002)
- All LLM calls go through `localhost:4000`.
- Model strings are aliases (`"haiku"`, `"sonnet"`) defined in `litellm_config.yaml`.
- Application code uses `settings.litellm.haiku_alias` / `sonnet_alias` —
  never raw provider model IDs (`"claude-haiku-4-5-..."`).
- Verify Anthropic `cache_control` headers actually flow through end-to-end
  before relying on prompt caching.

### Interrupt-based clarification (ADR-0003)
- Vague queries pause the graph via `interrupt()`, not via terminate-and-rerun.
- SQLite checkpointer is required (`langgraph-checkpoint-sqlite`).
- Each query has a unique `thread_id` (UUID, generated at CLI invocation).
- Cap clarification at **2 rounds** per query.

### Conviction calibration
- Synthesis system prompt embeds the rubric verbatim (see system design doc).
- Contradictions cap conviction at 3 and are surfaced in `gaps`.
- `Field(description=...)` instructs the model to pick the lower number when uncertain.

---

## Tool-specific rules

### Temporal

- **Workflows must be deterministic.** No `datetime.now()`, `random`, file I/O,
  `await asyncio.sleep`, or imports with side effects in workflow code.
- Use `workflow.now()`, `workflow.uuid4()`, `workflow.sleep()`.
- All non-deterministic work goes in **activities**.
- Every activity call must pass `start_to_close_timeout` and an explicit
  `RetryPolicy`. No silent defaults.
- `non_retryable_error_types=["FileNotFoundError"]` for config errors.
- The dev server (`temporal server start-dev`) is in-memory; restart wipes
  history. Acceptable for the lab.
- Workflow + activity registration lives in `horcrux/worker.py`. One worker process.

### LangGraph

- Nodes return **partial state**, never full state. `return {"intent": x}` merges.
- Conditional edges return a node-name **string** or `END` (the imported constant).
- Routing must be **dry-runnable** without LLMs — write tests that seed state
  manually and assert the next-node decision.
- `interrupt()` requires a checkpointer + `thread_id` (see ADR-0003).
- Parallel fan-out within a node uses `asyncio.gather`, not `Send`. Send is
  for true graph-level parallel branches; in-node gather is enough here.

### PydanticAI

- Every agent declares `result_type=SomePydanticModel`. **Never untyped output.**
- `Field(description=...)` is part of the tool schema the model sees. Better
  descriptions = fewer validation retries — this is the highest-leverage
  prompt engineering in the codebase.
- Test agents in isolation before composition. Each agent has a `scripts/`
  smoke-test entrypoint.
- Agents reference models via `settings.litellm.haiku_alias` / `sonnet_alias`.
- `result.data` is the only safe accessor — typed and validated.

### LiteLLM

- Proxy mode only. The library mode is not used in this codebase.
- `litellm_config.yaml` defines aliases, caching, and rate limits.
- `litellm_config.local.yaml` (gitignored) is the per-developer override.
- Spend tracking and call inspection happen at `localhost:4000/ui` —
  complementary to LangSmith's graph view.

### Qdrant

- Two collections only: `hp_chapters`, `hp_paragraphs`.
- **Payload indexes created before upsert** (`book_num` INTEGER, `chapter_num`
  INTEGER, `characters` KEYWORD, `chunk_type` KEYWORD).
- bge-large-en-v1.5 is **asymmetric**:
  - Passages: no prefix.
  - Queries: `"Represent this sentence for searching relevant passages: "` prefix.
  - Wrap encoding so the two paths cannot be mixed up.
- `normalize_embeddings=True` + `Distance.COSINE`. Mismatch is a silent
  precision hit.

### Rich

- `RichHandler` for stdlib `logging` (no JSON logger, no python-json-logger).
- `rich.progress.Progress` for ingest loops.
- `rich.panel.Panel` + `rich.markdown.Markdown` for `ResearchReport` rendering.
- `rich.table.Table` for verbose-mode scoring breakdowns.
- `rich.prompt.Prompt.ask` for clarification interrupt input.

---

## Coding rules

- **Python 3.12.** Modern type hints throughout (`list[str]`, `X | None`).
- **`uv`** for package management. `uv sync` after dependency changes.
- **Ruff** formatted. Line length 100. Import sorting on (`I`).
  - select: `E`, `F`, `I`, `B`, `UP`, `N`, `S`, `C90`, `SIM`, `PL`, `RUF`.
  - `S101` ignored only in `tests/`, enforced via `pyproject.toml`:
    ```toml
    [tool.ruff.lint.per-file-ignores]
    "tests/**" = ["S101"]
    ```
- **TDD** where it adds signal — pure functions, chunking thresholds,
  routing logic. Skip TDD for thin glue (e.g. wiring an agent into the graph)
  where the test would just mirror the wiring.
- **`pytest`** — every test tagged `unit`, `integration`, or `smoke` via
  marker. Unit tests must never touch live services (Anthropic, Qdrant,
  Temporal). Use stubs / fixtures.
- **Pydantic v2** throughout. `pydantic-settings` for config.
- **`httpx`**, never `requests`.
- **No naked `except:`.** Errors handled in place or re-raised with context.
  Error taxonomy: `domain` / `infra` / `validation`.
- **No `os.getenv` in core logic.** `from horcrux.config import settings`.
- **Fail fast on missing config.** Enforced by `pydantic-settings` validation.
- **Comments only when the WHY is non-obvious.** Identifiers are the primary
  documentation.

### Libraries explicitly NOT used in this lab

- `SQLModel` / `SQLAlchemy` — no relational store.
- `Tenacity` — Temporal handles retry for activities; PydanticAI handles
  retry for validation. Adding Tenacity layers retry on retry.
- `Sentry` / CloudWatch — local lab; observability is LangSmith + LiteLLM UI + Rich.
- `python-dotenv` — `pydantic-settings` reads `.env` natively.
- `boto3` / AWS SDKs — no cloud.

---

## Project structure

```
horcrux/                    # the package
├── __init__.py
├── config.py               # the config singleton (ADR-0004)
├── models.py               # all Pydantic models — single source of types
├── ocr.py                  # PDF → raw text
├── chapters.py             # chapter detection
├── chunking.py             # semantic chunking
├── characters.py           # canonical names + alias resolution
├── ingest.py               # Qdrant setup + upsert
├── retrieval.py            # search + RRF merge
├── agents.py               # PydanticAI agents
├── graph.py                # LangGraph pipeline
├── workflows.py            # Temporal workflows
├── activities.py           # Temporal activities
├── worker.py               # Temporal worker entrypoint
└── main.py                 # CLI entry point

data_lake/                  # raw inputs (gitignored except README)
  └── corpus.pdf            # your legally-obtained PDF (gitignored)

scripts/                    # exploratory scripts; superseded by horcrux/ when stable
tests/
  ├── unit/
  ├── integration/
  ├── smoke/
  └── fixtures/             # synthetic fantasy text — no copyrighted content
docs/
  ├── horcrux_system_design.md
  ├── adr/{pending,done}/
  ├── log/
  └── lab/toolchain-path.md
litellm_config.yaml         # model aliases, caching, rate limits
docker-compose.yaml         # Qdrant only (Temporal runs as dev server)
.env.example                # populated template; .env is gitignored
Makefile
pyproject.toml
```

One obvious entry point per surface: `horcrux.main` (CLI), `horcrux.worker` (Temporal worker).

---

## Local infrastructure (3 services, 3 terminals)

```
Terminal 1:  make local                 # docker compose up — Qdrant on :6333
Terminal 2:  make temporal              # Temporal dev server on :7233 (UI :8233)
Terminal 3:  make proxy                 # LiteLLM proxy on :4000 (UI :4000/ui)
```

Then a fourth for the worker (`make worker`) and a fifth to invoke queries
(`make run Q="..."`).

### Required system tools (not installed by `uv sync`)

`uv`, `docker`, `tesseract`, `temporal` (CLI binary). `make preflight`
checks all of these and `.env` presence in one shot. See README.md for
install commands. The Temporal CLI installer's default PATH line writes
to `~/.bashrc` — zsh users need to add it to `~/.zshrc` explicitly.

---

## Makefile (required targets)

| Target | Description |
|---|---|
| `make run` | Run the CLI for a single query (use after infra is up). |
| `make worker` | Start the Temporal worker. |
| `make local` | Spin up Qdrant via docker-compose. |
| `make test` | Run unit tests. |
| `make integration-test` | Run integration tests against local Qdrant + Temporal. |
| `make lint` | `ruff check`. |
| `make format` | `ruff format`. |
| `make doctor` | Imports `horcrux.config.settings` and prints redacted summary. |

No Terraform targets. No deploy targets.

---

## ADR Process

ADRs live in `docs/adr/` and follow a strict lifecycle:

```
docs/adr/
  pending/        # In-flight — ADR proposed, change not yet implemented
  done/           # Implemented — ADR accepted and change shipped
  MANIFEST.md     # Master index
```

| State | Meaning |
|---|---|
| `pending` | Change is in-flight; ADR written but not yet shipped |
| `done` | Change implemented and ADR finalised |

> ADRs are never deleted. Superseded ADRs move to `done/` with a note
> referencing the new ADR.

### Naming

```
docs/adr/pending/ADR-XXXX-short-title.md
docs/adr/done/ADR-XXXX-short-title.md
```

### Template

```markdown
# ADR-XXXX: Title

**Date:** YYYY-MM-DD
**Status:** pending | done
**Pattern:** (e.g. Adapter, Event-Driven, Singleton)

## Context
What is the problem or opportunity?

## Decision
What was decided?

## Alternatives Considered
What else was evaluated and why was it rejected?

## Consequences
Trade-offs, risks, follow-ups.

## Rollback
*(Required for any change with state, schema, or process impact.)*
How do we undo this if it goes wrong?
```

### `MANIFEST.md`

Markdown table updated whenever an ADR is created or its status changes.
Format demonstrated in `docs/adr/MANIFEST.md`.

---

## Change Log

**Every day with repo changes must have a corresponding log file** in `docs/log/`.

```
docs/log/YYYY-MM-DD.md   # one file per calendar day, append-only
```

Created or appended to at the **end of every working session** that changed
code, config, or documentation. **Append-only** — do not edit historical entries.

### Format

```markdown
# Change Log — YYYY-MM-DD

## Summary
Brief description of what changed.

## Changes
- [ADR-XXXX] Short description of change
- [HOTFIX] Short description if no ADR required
- [CHORE] Dependency bumps, formatting, refactor

## ADRs Progressed
- ADR-XXXX moved from pending → done
```

---

## Definition of Done

A change is not done until all of the following are true:

- [ ] Tests added or updated (unit + integration where the surface area warrants).
- [ ] No `os.getenv` calls — config flows through `settings`.
- [ ] No raw provider model strings — agents use `settings.litellm.*` aliases.
- [ ] If a `Finding` is constructed by code, `source_ids` is non-empty.
- [ ] If a Temporal workflow is touched, determinism rules verified.
- [ ] ADR written and added to `docs/adr/pending/` (if non-trivial).
- [ ] `docs/adr/MANIFEST.md` updated.
- [ ] `docs/log/YYYY-MM-DD.md` entry added.
- [ ] `make lint` passes.
- [ ] No copyrighted source material committed (verify with `git check-ignore`
      on `data_lake/*` and any `*.pdf`).

---

## Code Review Rules

Reviewer checklist:

- [ ] Strict-RAG invariants intact (`source_ids` non-empty, no parametric fallback).
- [ ] Config flows through the singleton (no `os.getenv` slips).
- [ ] Temporal workflows deterministic; activities have explicit timeouts + retries.
- [ ] Tests cover the behaviour, not the implementation.
- [ ] No secrets in code, logs, or outputs.
- [ ] No copyrighted text in fixtures or commits.
- [ ] ADR present if non-trivial; MANIFEST and change log updated.
- [ ] Backward compatibility on data models — `ChapterChunk` schema changes
      require an ADR with a re-ingest plan.

---

## Out of scope (explicit non-goals)

- Web UI of any kind.
- Production deployment (no AWS, Terraform, Bitbucket Pipelines, or CI/CD).
- Multi-tenant isolation, auth, rate limiting at the application layer.
- Cost guardrails beyond what LiteLLM's proxy provides.
- Embedding model fine-tuning.
- SLAs, alarms, on-call rotations.

If a feature request lands in this list, the answer is "out of scope for the
lab" — possibly captured as a follow-up ADR for a hypothetical productionised
version, but not implemented.
