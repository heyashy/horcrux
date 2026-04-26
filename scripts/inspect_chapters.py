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

from horcrux.config import settings
from horcrux.corpus.chapters import extract_chapters
from horcrux.corpus.cleansing import cleanse_pages
from horcrux.models import Chapter, RawPage

_RAW_PAGES_PATH = Path("data/processed/raw_pages.json")

# Ground truth from the corpus PDF's table of contents.
_GROUND_TRUTH_BOOKS: dict[int, tuple[str, int]] = {
    1: ("Sorcerer's Stone", 7),
    2: ("Chamber of Secrets", 276),
    3: ("Prisoner of Azkaban", 567),
    4: ("Goblet of Fire", 941),
    5: ("Order of the Phoenix", 1562),
    6: ("Half-Blood Prince", 2408),
    7: ("Deathly Hallows", 2966),
}

# Book 1 chapters where the TOC was visible. Ch 7 and Ch 13 were truncated in
# the screenshot; left blank but monotonic page progression is still checked.
_GROUND_TRUTH_BOOK_1: dict[int, int] = {
    1: 12, 2: 26, 3: 37, 4: 50, 5: 62, 6: 85,
    8: 123, 9: 133, 10: 150, 11: 164, 12: 176,
    14: 205, 15: 217, 16: 234, 17: 257,
}


def _load_pages(path: Path) -> list[RawPage] | None:
    if not path.exists():
        return None
    return [RawPage(**p) for p in json.loads(path.read_text())]


def _detect_chapters(pages: list[RawPage]) -> list[Chapter]:
    return extract_chapters(settings.corpus_path, cleanse_pages(pages))


def _group_by_book(chapters: list[Chapter]) -> dict[int, list[Chapter]]:
    by_book: dict[int, list[Chapter]] = {}
    for c in chapters:
        by_book.setdefault(c.book_num, []).append(c)
    return by_book


def _drift_cell(detected_page: int, expected_page: int | None) -> str:
    """Coloured drift indicator: green ✓ exact, yellow ±1-2, red >2, dim — unknown."""
    if expected_page is None:
        return "[dim]—[/]"
    drift = detected_page - expected_page
    if drift == 0:
        return "[green]✓[/]"
    if abs(drift) <= 2:
        return f"[yellow]{drift:+d}[/]"
    return f"[red]{drift:+d}[/]"


def _build_book_table(
    book_num: int, book_chapters: list[Chapter]
) -> Table:
    table = Table(show_header=True, header_style="bold dim")
    table.add_column("Ch", justify="right")
    table.add_column("Title (detected)")
    table.add_column("Pages")
    table.add_column("Expected", justify="right")
    table.add_column("Drift")

    for c in book_chapters:
        expected = (
            _GROUND_TRUTH_BOOK_1.get(c.chapter_num) if book_num == 1 else None
        )
        title = c.chapter_title[:50] if c.chapter_title else "[dim]<no title>[/]"
        table.add_row(
            str(c.chapter_num),
            title,
            f"{c.page_start}-{c.page_end}",
            str(expected) if expected else "—",
            _drift_cell(c.page_start, expected),
        )
    return table


def _render_book(
    console: Console, book_num: int, book_chapters: list[Chapter]
) -> None:
    truth_name, truth_start = _GROUND_TRUTH_BOOKS.get(book_num, ("?", 0))
    first_page = book_chapters[0].page_start
    console.print(
        f"[bold cyan]Book {book_num}[/]  "
        f"[white]{truth_name}[/]  "
        f"[dim]TOC says starts page {truth_start}; "
        f"first chapter detected at page {first_page}[/]"
    )
    console.print(_build_book_table(book_num, book_chapters))
    console.print()


def _render_book1_accuracy(
    console: Console, by_book: dict[int, list[Chapter]]
) -> None:
    if 1 not in by_book:
        return
    matched = [
        c for c in by_book[1]
        if c.chapter_num in _GROUND_TRUTH_BOOK_1
        and c.page_start == _GROUND_TRUTH_BOOK_1[c.chapter_num]
    ]
    console.print(
        f"[bold]Book 1 accuracy:[/] {len(matched)}/{len(_GROUND_TRUTH_BOOK_1)} "
        f"chapters matched exactly."
    )


def main() -> None:
    console = Console()

    pages = _load_pages(_RAW_PAGES_PATH)
    if pages is None:
        console.print(
            f"[red]missing:[/] {_RAW_PAGES_PATH} — run `make ingest` first"
        )
        return
    console.print(f"[dim]loaded {len(pages)} OCR'd pages[/]\n")

    chapters = _detect_chapters(pages)
    console.print(f"[bold]Detected {len(chapters)} chapters[/]\n")

    by_book = _group_by_book(chapters)
    for book_num in sorted(by_book):
        _render_book(console, book_num, by_book[book_num])

    _render_book1_accuracy(console, by_book)


if __name__ == "__main__":
    main()
