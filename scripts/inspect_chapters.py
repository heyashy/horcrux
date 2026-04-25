"""Run chapter detection on cached OCR output and diff against ground truth.

Reads `data/processed/raw_pages.json`, cleanses, runs chapter detection,
prints a per-chapter table with expected page numbers from the source
PDF's table of contents. Drift indicates regex or OCR edge cases.

    uv run python scripts/inspect_chapters.py
"""

import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from horcrux.chapters import extract_chapters
from horcrux.cleansing import cleanse_pages
from horcrux.config import settings
from horcrux.models import RawPage


# Ground truth from the corpus PDF's table of contents.
GROUND_TRUTH_BOOKS = {
    1: ("Sorcerer's Stone", 7),
    2: ("Chamber of Secrets", 276),
    3: ("Prisoner of Azkaban", 567),
    4: ("Goblet of Fire", 941),
    5: ("Order of the Phoenix", 1562),
    6: ("Half-Blood Prince", 2408),
    7: ("Deathly Hallows", 2966),
}

# Book 1 chapters where the TOC was visible. (Ch 7 and Ch 13 truncated in
# the screenshot — leave blank; detection will still be checked for
# monotonic page progression.)
GROUND_TRUTH_BOOK_1 = {
    1: 12,
    2: 26,
    3: 37,
    4: 50,
    5: 62,
    6: 85,
    8: 123,
    9: 133,
    10: 150,
    11: 164,
    12: 176,
    14: 205,
    15: 217,
    16: 234,
    17: 257,
}


def main() -> None:
    console = Console()

    raw_path = Path("data/processed/raw_pages.json")
    if not raw_path.exists():
        console.print(f"[red]missing:[/] {raw_path} — run `make run Q=...` or the ocr CLI first")
        return

    raw = json.loads(raw_path.read_text())
    pages = [RawPage(**p) for p in raw]
    console.print(f"[dim]loaded {len(pages)} OCR'd pages[/]\n")

    cleansed = cleanse_pages(pages)
    chapters = extract_chapters(settings.corpus_path, cleansed)

    console.print(f"[bold]Detected {len(chapters)} chapters[/]\n")

    by_book: dict[int, list] = {}
    for c in chapters:
        by_book.setdefault(c.book_num, []).append(c)

    for book_num in sorted(by_book):
        truth_name, truth_start = GROUND_TRUTH_BOOKS.get(book_num, ("?", 0))
        book_chapters = by_book[book_num]
        first_page = book_chapters[0].page_start

        console.print(
            f"[bold cyan]Book {book_num}[/]  "
            f"[white]{truth_name}[/]  "
            f"[dim]TOC says starts page {truth_start}; first chapter detected at page {first_page}[/]"
        )

        table = Table(show_header=True, header_style="bold dim")
        table.add_column("Ch", justify="right")
        table.add_column("Title (detected)")
        table.add_column("Pages")
        table.add_column("Expected", justify="right")
        table.add_column("Drift")

        for c in book_chapters:
            expected = (
                GROUND_TRUTH_BOOK_1.get(c.chapter_num) if book_num == 1 else None
            )
            if expected is not None:
                drift = c.page_start - expected
                drift_cell = (
                    "[green]✓[/]" if drift == 0
                    else f"[yellow]{drift:+d}[/]" if abs(drift) <= 2
                    else f"[red]{drift:+d}[/]"
                )
            else:
                drift_cell = "[dim]—[/]"

            title = c.chapter_title[:50] if c.chapter_title else "[dim]<no title>[/]"
            table.add_row(
                str(c.chapter_num),
                title,
                f"{c.page_start}-{c.page_end}",
                str(expected) if expected else "—",
                drift_cell,
            )
        console.print(table)
        console.print()

    # Stats summary
    if 1 in by_book:
        b1 = by_book[1]
        matched = [
            c for c in b1
            if c.chapter_num in GROUND_TRUTH_BOOK_1
            and c.page_start == GROUND_TRUTH_BOOK_1[c.chapter_num]
        ]
        console.print(
            f"[bold]Book 1 accuracy:[/] {len(matched)}/{len(GROUND_TRUTH_BOOK_1)} "
            f"chapters matched exactly."
        )


if __name__ == "__main__":
    main()
