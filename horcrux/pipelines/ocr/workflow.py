"""OCR ingest workflow.

Orchestrates parallel batched OCR over the corpus PDF. Three steps:

  1. count_pages_activity        — find total page count
  2. ocr_batch_activity × N      — fan out N batches in parallel; each
                                   writes its result to disk and returns
                                   the path
  3. merge_batches_activity      — read all batch files, write the
                                   canonical raw_pages.json

Sandbox-safe: only stdlib + temporalio at module level. Activity
imports go through `imports_passed_through` so the activity *references*
compile in the workflow sandbox without pulling in pymupdf, Rich, etc.
"""

import asyncio
from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from horcrux.pipelines.ocr.activities import (
        count_pages_activity,
        merge_batches_activity,
        ocr_batch_activity,
    )


# Pages per batch. 50 yields ~73 batches for the HP corpus.
#
# Trade-offs at this size:
#   - Each batch finishes in ~75-100s on a laptop under 4x concurrency,
#     well inside the per-batch start_to_close_timeout headroom.
#   - On worker crash, max wasted OCR work is 50 pages (vs 100). Smaller
#     batches degrade more gracefully.
#   - 73 batches × ~4 events each = ~292 history events, well under the
#     50k Temporal history budget.
#
# Larger (100, 200) is fine for production hardware where each batch
# completes well within the timeout. Smaller (25) is safer for slower
# environments at the cost of more orchestration overhead.
BATCH_SIZE = 50


@workflow.defn
class IngestOCRWorkflow:
    """Run OCR over the full PDF in parallel batches.

    Args:
        pdf_path: path to the source PDF (string, not Path — workflow
                  arguments must be JSON-serialisable).

    Returns:
        Path to the merged `raw_pages.json` artefact.
    """

    @workflow.run
    async def run(self, pdf_path: str) -> str:
        # Step 1 — page count via activity (we can't open the PDF in the
        # workflow because workflows must be deterministic and replay-safe;
        # any I/O lives in activities).
        page_count = await workflow.execute_activity(
            count_pages_activity,
            args=[pdf_path],
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        # Step 2 — fan out one activity per batch. asyncio.gather schedules
        # all batches concurrently; the worker's max_concurrent_activities
        # caps actual parallelism (default 8).
        batch_specs = [
            (start, min(start + BATCH_SIZE - 1, page_count))
            for start in range(1, page_count + 1, BATCH_SIZE)
        ]
        batch_paths = await asyncio.gather(*[
            workflow.execute_activity(
                ocr_batch_activity,
                args=[pdf_path, start, end],
                # 15min headroom — under concurrency with thermal throttling
                # a 100-page batch can easily stretch from 77s baseline to
                # several minutes. Generous timeout costs nothing on the
                # happy path; tight timeouts cause cascade failures.
                start_to_close_timeout=timedelta(minutes=15),
                # Heartbeat every minute so the server knows we're alive.
                heartbeat_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3),
            )
            for start, end in batch_specs
        ])

        # Step 3 — merge per-batch JSON files into the canonical artefact.
        return await workflow.execute_activity(
            merge_batches_activity,
            args=[batch_paths],
            start_to_close_timeout=timedelta(minutes=2),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
