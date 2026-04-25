"""Silver tier — derive chapters.json from raw_pages.json.

Reads OCR'd pages, applies cleansing, runs PDF-outline chapter detection,
writes a book-grouped JSON artefact suitable for downstream consumption.

    uv run python scripts/build_chapters.py

Inputs:
    data/processed/raw_pages.json   (Bronze tier — OCR cache)
    data_lake/corpus.pdf            (for book titles via TOC)

Output:
    data/processed/chapters.json    (Silver tier — book-grouped chapters)
"""

import json
from pathlib import Path

from rich.console import Console

from horcrux.chapters import dump_chapters_json, extract_chapters
from horcrux.cleansing import cleanse_pages
from horcrux.config import settings
from horcrux.models import Chapter, RawPage

_RAW_PAGES_PATH = Path("data/processed/raw_pages.json")
_CHAPTERS_OUTPUT_PATH = Path("data/processed/chapters.json")


def _load_pages(path: Path) -> list[RawPage] | None:
    """Read OCR pages from disk. Returns None if the file isn't there."""
    if not path.exists():
        return None
    return [RawPage(**p) for p in json.loads(path.read_text())]


def _build_chapters(pages: list[RawPage]) -> list[Chapter]:
    """Cleanse pages and run chapter detection."""
    cleansed = cleanse_pages(pages)
    return extract_chapters(settings.corpus_path, cleansed)


def _render_book_summary(console: Console, output_path: Path) -> None:
    """Print one line per book — quick visual sanity check."""
    data = json.loads(output_path.read_text())
    for book in data["books"]:
        chars_total = sum(len(c["text"]) for c in book["chapters"])
        console.print(
            f"  [cyan]Book {book['book_num']}[/]  "
            f"{book['title'][:50]:50}  "
            f"{len(book['chapters']):2} chapters, "
            f"{chars_total:,} chars"
        )


def main() -> None:
    console = Console()

    pages = _load_pages(_RAW_PAGES_PATH)
    if pages is None:
        console.print(f"[red]missing:[/] {_RAW_PAGES_PATH} — run `make ingest` first")
        return
    console.print(f"[dim]loaded {len(pages)} pages from {_RAW_PAGES_PATH.name}[/]")

    chapters = _build_chapters(pages)
    console.print(f"[dim]extracted {len(chapters)} chapters[/]")

    dump_chapters_json(chapters, settings.corpus_path, _CHAPTERS_OUTPUT_PATH)
    console.print()
    _render_book_summary(console, _CHAPTERS_OUTPUT_PATH)
    console.print(f"\n[green]saved[/] {_CHAPTERS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
