"""OCR ingest activities.

Activities run in the worker's regular Python process, not the workflow
sandbox — heavy imports (pymupdf, pytesseract, sentence-transformers in
later pipelines) are fine here.

Each activity is responsible for its own durability:
  - ocr_batch_activity     writes its batch JSON before returning the path
  - merge_batches_activity reads all batch files, writes the merged artefact

Idempotent on retry: re-running an activity with the same inputs overwrites
the same output file. Workflow-level event history is the orchestration
durability layer; on-disk batch files are the data durability layer.
"""

import asyncio
import json
from pathlib import Path

import pymupdf
from temporalio import activity

from horcrux.ocr import ocr_pages

_BATCH_DIR = Path("data/processed/raw_pages_batches")
_FINAL_PATH = Path("data/processed/raw_pages.json")


@activity.defn
async def count_pages_activity(pdf_path: str) -> int:
    """Count pages in the source PDF. Cheap (metadata read only).

    Inlined (no `asyncio.to_thread`) — pymupdf metadata read is sub-second,
    not worth the off-thread machinery, and it removes a layer of complexity
    that was making this activity look like a hang to the worker.
    """
    abs_pdf = str(Path(pdf_path).resolve())
    activity.logger.info(f"counting pages in {abs_pdf}")
    with pymupdf.open(abs_pdf) as doc:
        return len(doc)


@activity.defn
async def ocr_batch_activity(pdf_path: str, start_page: int, end_page: int) -> str:
    """OCR pages [start_page, end_page] inclusive. Write batch JSON; return path.

    The on-disk batch file is the durable artefact; the workflow stores
    only the path string in event history (keeping replay state tiny).

    Resolves `pdf_path` to absolute before passing to OCR — defends against
    cwd mismatches between the trigger process and the worker process.
    """
    # Resolve to absolute path so cwd of the worker process doesn't matter.
    abs_pdf = str(Path(pdf_path).resolve())
    activity.logger.info(
        f"OCR batch start: pages {start_page}-{end_page} of {abs_pdf}"
    )

    # Periodic heartbeat keeps Temporal informed that we're alive even
    # during long single-batch runs (e.g. under load).
    heartbeat_task = asyncio.create_task(_heartbeat_loop())
    try:
        pages = await asyncio.to_thread(
            ocr_pages,
            abs_pdf,
            start_page=start_page,
            end_page=end_page,
            show_progress=False,
        )
    finally:
        heartbeat_task.cancel()

    _BATCH_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _BATCH_DIR / f"batch_{start_page:05d}_{end_page:05d}.json"
    output_path.write_text(
        json.dumps([p.model_dump() for p in pages], indent=2, ensure_ascii=False)
    )

    activity.logger.info(
        f"OCR batch done: {len(pages)} pages → {output_path.name}"
    )
    return str(output_path)


async def _heartbeat_loop() -> None:
    """Heartbeat every 30 seconds while the activity is in flight."""
    try:
        while True:
            await asyncio.sleep(30)
            activity.heartbeat()
    except asyncio.CancelledError:
        pass


@activity.defn
async def merge_batches_activity(batch_paths: list[str]) -> str:
    """Read all batch JSONs in page order, write the canonical raw_pages.json."""
    return await asyncio.to_thread(_merge_batches, batch_paths)


def _merge_batches(batch_paths: list[str]) -> str:
    all_pages: list[dict] = []
    for path in batch_paths:
        all_pages.extend(json.loads(Path(path).read_text()))

    # Concurrent activities complete in arbitrary order; sort by page_num
    # to guarantee canonical ordering in the merged artefact.
    all_pages.sort(key=lambda p: p["page_num"])

    _FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    _FINAL_PATH.write_text(
        json.dumps(all_pages, indent=2, ensure_ascii=False)
    )

    activity.logger.info(
        f"merged {len(all_pages)} pages from {len(batch_paths)} batches → {_FINAL_PATH}"
    )
    return str(_FINAL_PATH)
