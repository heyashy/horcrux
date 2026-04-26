"""CLI entrypoint for Horcrux.

Subcommands grow phase by phase.

    uv run python -m horcrux.main ocr --start 1 --end 520
    uv run python -m horcrux.main ingest        # full corpus via Temporal
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from uuid import uuid4

from rich.console import Console
from rich.logging import RichHandler

from horcrux.config import settings
from horcrux.corpus.ocr import ocr_pages
from horcrux.models import RawPage

console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console)],
    )


def _save_pages(pages: list[RawPage], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([p.model_dump() for p in pages], indent=2, ensure_ascii=False)
    )


def cmd_ocr(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf or settings.corpus_path)
    output_path = Path(args.output)

    if output_path.exists() and not args.force:
        console.print(
            f"[yellow]output exists:[/] {output_path}\n"
            f"  delete it or pass [bold]--force[/] to overwrite"
        )
        return 1

    console.print(f"[bold]OCR[/] {pdf_path} → {output_path}")
    pages = ocr_pages(pdf_path, start_page=args.start, end_page=args.end)
    _save_pages(pages, output_path)

    char_total = sum(len(p.text) for p in pages)
    blank = sum(1 for p in pages if not p.text.strip())
    console.print(
        f"[green]done[/] · {len(pages)} pages · "
        f"{char_total:,} chars · {blank} blank"
    )
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Trigger the IngestOCRWorkflow on the Temporal dev server.

    Blocks until the workflow completes; prints the merged artefact path.
    """
    pdf_path = args.pdf or settings.corpus_path

    async def _run() -> int:
        from temporalio.client import Client

        client = await Client.connect(
            settings.temporal.address,
            namespace=settings.temporal.namespace,
        )
        workflow_id = f"ingest-ocr-{uuid4().hex[:8]}"
        console.print(f"[bold]starting workflow[/]  id={workflow_id}")
        console.print(
            f"watch in UI: [link]http://localhost:8233/namespaces/{settings.temporal.namespace}/workflows/{workflow_id}[/link]"
        )
        result = await client.execute_workflow(
            "IngestOCRWorkflow",
            pdf_path,
            id=workflow_id,
            task_queue=settings.temporal.task_queue,
        )
        console.print(f"[green]done[/]  merged → {result}")
        return 0

    return asyncio.run(_run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horcrux", description="Deep research over a literary corpus.")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr", help="OCR a page range from the corpus PDF (single-threaded, no Temporal)")
    ocr.add_argument("--pdf", help=f"PDF path (default: {settings.corpus_path})")
    ocr.add_argument("--start", type=int, default=1, help="1-indexed start page (inclusive)")
    ocr.add_argument("--end", type=int, default=None, help="1-indexed end page (inclusive); default = last")
    ocr.add_argument(
        "--output", "-o", default="data/processed/raw_pages.json",
        help="output JSON path"
    )
    ocr.add_argument("--force", action="store_true", help="overwrite existing output")
    ocr.set_defaults(func=cmd_ocr)

    ingest = sub.add_parser(
        "ingest",
        help="Run the full-corpus OCR pipeline as a Temporal workflow (durable, parallel)",
    )
    ingest.add_argument("--pdf", help=f"PDF path (default: {settings.corpus_path})")
    ingest.set_defaults(func=cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
