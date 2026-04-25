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
from horcrux.models import RawPage


def main() -> None:
    console = Console()

    raw_path = Path("data/processed/raw_pages.json")
    if not raw_path.exists():
        console.print(f"[red]missing:[/] {raw_path} — run `make ingest` first")
        return

    raw = json.loads(raw_path.read_text())
    pages = [RawPage(**p) for p in raw]
    console.print(f"[dim]loaded {len(pages)} pages from raw_pages.json[/]")

    cleansed = cleanse_pages(pages)
    chapters = extract_chapters(settings.corpus_path, cleansed)
    console.print(f"[dim]extracted {len(chapters)} chapters[/]")

    output_path = Path("data/processed/chapters.json")
    dump_chapters_json(chapters, settings.corpus_path, output_path)

    # Summary by book — quick visual confirmation that the structure is sane.
    data = json.loads(output_path.read_text())
    console.print()
    for book in data["books"]:
        chars_total = sum(len(c["text"]) for c in book["chapters"])
        console.print(
            f"  [cyan]Book {book['book_num']}[/]  "
            f"{book['title'][:50]:50}  "
            f"{len(book['chapters']):2} chapters, "
            f"{chars_total:,} chars"
        )

    console.print(f"\n[green]saved[/] {output_path}")


if __name__ == "__main__":
    main()
