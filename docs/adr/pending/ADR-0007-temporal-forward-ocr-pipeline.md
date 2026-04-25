# ADR-0007: Temporal-forward OCR pipeline

**Date:** 2026-04-25
**Status:** pending
**Pattern:** Durable workflow with parallel batched activities; per-pipeline
directory layout for compositional growth.

## Context

The original plan placed full-pipeline Temporal wrapping in Phase 4
(after chunking, embedding, and Qdrant upsert were built). Phase 1's
single-threaded OCR completed book 1 in 6:42, with the full-corpus
extrapolation at ~46 minutes. Phase 2 work (Tier 2 LLM character merge,
chunking threshold sweep) genuinely benefited from full-corpus data —
Marauder nicknames, the Voldemort/Tom Riddle reveal arc, and many
semantic aliases first appear in books 3-7, none of which were in our
520-page slice.

Two things during Phase 2 made the pull-forward attractive:

1. fastcoref hit a transformers API incompatibility mid-loading, losing
   ~30s of model-loading work. A small preview of "non-durable pipelines
   waste your time."
2. The user identified the symmetric opportunity: "shouldn't this be
   running on Temporal?" The pedagogical readiness was there.

Decision point: run full single-threaded OCR (~46 min wait, no Temporal
learning), or pull a *slice* of Phase 4 forward and wrap OCR alone as a
Temporal workflow (~22 min on laptop hardware, durable, real Temporal
experience).

## Decision

Pull Temporal forward for OCR specifically. Wrap **only** the OCR step;
leave chunking / embedding / upsert as future workflow additions in
Phase 4.

### Architectural commitments

**1. Per-pipeline directory layout.**

```
horcrux/pipelines/
├── __init__.py
└── ocr/
    ├── __init__.py
    ├── workflow.py      ← IngestOCRWorkflow (sandbox-safe imports only)
    └── activities.py    ← ocr_batch_activity, count_pages_activity, merge_batches_activity
```

Future pipelines (chunking, embedding) land alongside as
`horcrux/pipelines/chunking/` etc. Workflow + activities are tightly
coupled and live together; pipelines are not coupled and live apart.

**2. Single worker registers all pipelines.**

`horcrux/worker.py` imports from each pipeline subdirectory and
explicitly lists workflows + activities. Adding a new pipeline is two
import lines plus two list extensions. No magic registry, no
auto-discovery.

**3. On-disk batch files as the data-durability layer.**

Each `ocr_batch_activity` writes its result to
`data/processed/raw_pages_batches/batch_NNNNN_NNNNN.json` and returns
*just the path string*. The workflow stores tiny strings in event
history; the actual data sits on disk. Crash recovery has two layers:

- *Workflow* knows which batches completed (event history).
- *Disk* has each completed batch's output independently.

After all batches complete, `merge_batches_activity` reads them in
page order and writes the canonical `data/processed/raw_pages.json`.

**4. Conservative tuning for laptop hardware.**

Operational settings encoded in code:
- `BATCH_SIZE = 50` — small enough for ~75-100s per batch under
  contention, big enough to keep activity count reasonable (~73).
- `max_concurrent_activities = 4` — half of CPU core count, leaves
  headroom for the desktop and Tesseract subprocess overhead.
- `start_to_close_timeout = 15 minutes` — 4-5x the realistic batch
  duration; cheap on the happy path, prevents cascade timeout failures.
- `heartbeat_timeout = 2 minutes` — server knows the worker is alive
  even on slower batches.
- `_heartbeat_loop` task pings every 30s while the OCR is running.

These are all *production hardware-dependent*. Workstation deployments
should bump concurrency and batch size; smaller machines should
reduce them.

## Alternatives Considered

**Run single-threaded full OCR.** Simplest path, ~46 min blocking wait.
Rejected because (a) the user had already internalised Temporal's value
proposition through the demo crash test and the fastcoref incident,
(b) we'd lose the durability story for the single most expensive step
in the pipeline, and (c) the per-batch artefacts are useful for
inspection and selective re-runs even outside the durability story.

**Wrap the entire ingest pipeline (OCR + cleansing + chapters +
chunking + embedding + upsert) in one workflow now.** Too much work for
a single milestone. Saved for Phase 4. The OCR-only wrap is bounded
scope, fits the lab pacing.

**Keep workflows.py / activities.py at horcrux/ root.** Singular naming
implies "one of these" — works for the OCR pipeline but breaks down
when chunking and embedding pipelines arrive. Rejected on architectural
foresight; the directory pattern is what we want long-term.

**Higher concurrency (max_concurrent_activities=8) with 5-min timeout.**
Tested, failed. Concurrent OCR on a laptop runs each batch much slower
than baseline (thermal throttling, memory bandwidth, subprocess startup
overhead). Each batch exceeded the 5-min timeout; one batch failure
cascaded through `asyncio.gather` and cancelled all the others;
30-minute run, every activity failed. Documented in iteration history
below.

**Larger batches (100 pages) with the same concurrency.** Tested, failed
for the same timeout reason. Smaller batches degrade more gracefully on
retry — max wasted work per crash is one batch's worth, not 100 pages.

## Consequences

**Positive**

- ~22-minute full-corpus OCR on laptop hardware, vs 46 minutes
  single-threaded. ~2x speedup with ample headroom against the 15-min
  per-batch timeout.
- Crash-test pattern proven on real work, not just the demo. The
  pattern transfers directly to production-shaped pipelines.
- Per-pipeline directory layout sets up Phase 4's chunking and
  embedding pipelines to land cleanly without restructuring.
- The iteration through real failure modes (timeout cascade, worker
  restart semantics, batch-size tuning) is now genuine engineering
  experience documented in the iteration history.

**Negative / risks**

- Three running services (Temporal dev server, Qdrant, LiteLLM proxy)
  plus a worker process plus the trigger CLI now needed for ingest. More
  moving parts to remember and document.
- The per-batch on-disk artefact pattern is a Temporal-idiom; new
  contributors need to understand "why are there 73 batch files in
  raw_pages_batches/?" The README and this ADR are the first
  documentation of that.
- Pulled scope from Phase 4 forward. Phase 4 is now smaller (just
  wrapping the remaining steps) but the architectural narrative
  shifts — Temporal isn't a "Phase 4 reveal" anymore.

**Follow-ups**

- Phase 4: extend the pipelines/ tree with chunking, embedding, upsert.
- A deep-dive doc for the OCR pipeline (similar to
  `docs/lab/character-discovery-deep-dive.md`).
- Tune knobs as configurable settings rather than module constants once
  more pipelines exist (`settings.ocr.batch_size`,
  `settings.ocr.max_concurrent`, etc.).
- Workstation-grade hardware can bump concurrency to 8-12 and
  batch_size to 200; document this in the deep-dive when relevant.

## Iteration history

The shipped configuration is V3. V1 and V2 were real failures we
diagnosed and fixed.

**V1.** `max_concurrent_activities=8`, `start_to_close_timeout=5min`,
`BATCH_SIZE=100`, no heartbeating. *Result:* every batch hit timeout
under contention. `asyncio.gather` cancelled all in-flight batches when
the first one failed → cascade. 30+ minutes of runtime, all activities
ultimately failed via `TIMEOUT_TYPE_START_TO_CLOSE` on the failing one
and `CancelledError` on the rest. The screenshot of the red event-history
bar lives in the change log.

**V2.** Bumped `max_concurrent_activities` to 4,
`start_to_close_timeout` to 15min, added a `_heartbeat_loop`, made
`pdf_path` resolve to absolute. Kept `BATCH_SIZE=100`. *Result:* never
ran cleanly — discovered cwd issues and decided to also reduce batch
size before re-running.

**V3 (shipped).** `BATCH_SIZE=50`, `max_concurrent_activities=4`,
`start_to_close_timeout=15min`, `heartbeat_timeout=2min`,
`_heartbeat_loop` every 30s. *Result:* ~22-minute clean run, 73 batches,
all first-attempt success. The visible 4-wide concurrency in the UI
matches the configured value; no retries fired.

Each version's failure mode is now a known regression, captured here in
prose and via the Temporal UI's event history (which the dev server
keeps in memory until restart — a real follow-up artefact would be a
saved screenshot).

## Rollback

The OCR functionality is intact in `horcrux/main.py ocr` (single-
threaded direct CLI), unchanged from Phase 1. `horcrux/ocr.py`'s
`ocr_pages()` function is the same; the Temporal activity is just a
thin async wrapper around it.

Full rollback path:

1. Delete `horcrux/pipelines/ocr/`.
2. Delete `horcrux/worker.py`.
3. Revert the `ingest` subcommand in `horcrux/main.py`.
4. Remove `make worker` and `make ingest` targets.
5. Use `uv run python -m horcrux.main ocr --start 1 --end 3623` for
   single-threaded full OCR.

No data migration required. ~10 minutes of work. Existing
`raw_pages.json` artefact remains valid regardless of which path
produced it.
