"""CLI entrypoint for Horcrux.

Subcommands grow phase by phase. Phase 1 ships `ocr` only.

    uv run python -m horcrux.main ocr --start 1 --end 520
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

from horcrux.config import settings
from horcrux.models import RawPage
from horcrux.ocr import ocr_pages

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="horcrux", description="Deep research over a literary corpus.")
    parser.add_argument("--verbose", "-v", action="store_true", help="DEBUG-level logging")
    sub = parser.add_subparsers(dest="command", required=True)

    ocr = sub.add_parser("ocr", help="OCR a page range from the corpus PDF")
    ocr.add_argument("--pdf", help=f"PDF path (default: {settings.corpus_path})")
    ocr.add_argument("--start", type=int, default=1, help="1-indexed start page (inclusive)")
    ocr.add_argument("--end", type=int, default=None, help="1-indexed end page (inclusive); default = last")
    ocr.add_argument(
        "--output", "-o", default="data/processed/raw_pages.json",
        help="output JSON path"
    )
    ocr.add_argument("--force", action="store_true", help="overwrite existing output")
    ocr.set_defaults(func=cmd_ocr)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
