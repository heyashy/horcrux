"""Temporal worker — registers every pipeline in horcrux/pipelines/.

One worker process polls one task queue (`horcrux`) and runs every
workflow + activity registered to it. Adding a new pipeline means
adding two import lines and adding to the workflows/activities lists.
No magic registry; explicit is better than auto-discovery.

Run:
    make worker
    # or:
    uv run python -m horcrux.worker
"""

import asyncio
import logging

from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
log = logging.getLogger("horcrux-worker")


async def main() -> None:
    # Imports inside main so RichHandler is configured before any sandbox
    # validation imports run. Same pattern as the demo worker.
    from temporalio.client import Client
    from temporalio.worker import Worker

    from horcrux.config import settings
    from horcrux.pipelines.ocr.activities import (
        count_pages_activity,
        merge_batches_activity,
        ocr_batch_activity,
    )
    from horcrux.pipelines.ocr.workflow import IngestOCRWorkflow

    client = await Client.connect(
        settings.temporal.address,
        namespace=settings.temporal.namespace,
    )

    worker = Worker(
        client,
        task_queue=settings.temporal.task_queue,
        workflows=[
            IngestOCRWorkflow,
        ],
        activities=[
            count_pages_activity,
            ocr_batch_activity,
            merge_batches_activity,
        ],
        # 4 is conservative for a laptop — OCR is CPU-heavy and 8 concurrent
        # tesseract subprocesses can hit thermal throttling / memory pressure.
        # Bump on workstation hardware.
        max_concurrent_activities=4,
    )

    log.info(
        "[bold]worker started[/]  task_queue=%s  max_concurrent_activities=4",
        settings.temporal.task_queue,
        extra={"markup": True},
    )
    await worker.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("[yellow]worker stopped[/]", extra={"markup": True})
