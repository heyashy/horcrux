"""Chapter and book detection from the PDF's embedded table of contents.

PyMuPDF's `doc.get_toc()` returns the PDF's bookmarks/outline as
[[level, title, page], ...] entries. This is the structural ground truth
the PDF author embedded — no OCR scanning, no regex false positives.

Convention for this corpus:
    level 1, title="Harry Potter and ..."   → book start
    level 1, other titles                   → front matter (skip)
    level 2, title="Chapter N - Title"      → chapter

For corpora without bookmarks, regex-on-OCR-text would be the fallback.
Not implemented here; would live alongside this module.
"""

import re
from pathlib import Path

import pymupdf

from horcrux.models import Chapter, RawPage

# Match "Chapter 5 - Diagon Alley", "Chapter 5: Diagon Alley",
# "Chapter 5 — Diagon Alley", with flexible spacing.
_CHAPTER_TITLE = re.compile(
    r"^\s*Chapter\s+(\d+)\s*[-:—]\s*(.*?)\s*$",
    re.IGNORECASE,
)

# TOC entry as PyMuPDF returns it: [level, title, page (1-indexed)]
TocEntry = tuple[int, str, int]

# Sentence-ending punctuation: if a page ends with one of these (possibly
# followed by a closing quote), the next page is a real paragraph break.
# Otherwise mid-sentence — join with a space, not `\n\n`.
_SENTENCE_END = re.compile(r'[.!?][\'"”’]?\s*$')


def _join_pages(page_texts: list[str]) -> str:
    """Concatenate page texts in a structure-preserving way.

    PDF page boundaries are layout artefacts, not semantic ones. A sentence
    that spans pages 50→51 must remain one sentence to downstream sentence
    segmentation, coref, and chunking.

    Heuristic:
      - Previous page ends with sentence-final punctuation → real paragraph
        break (`\\n\\n`).
      - Otherwise → mid-sentence continuation (single space).

    Imperfect (we don't know real paragraph boundaries within a page) but
    eliminates the worst case of fabricating a paragraph break in the
    middle of a sentence.
    """
    if not page_texts:
        return ""
    parts: list[str] = [page_texts[0].strip()]
    for text in page_texts[1:]:
        text = text.strip()
        if not text:
            continue
        prev = parts[-1] if parts else ""
        if _SENTENCE_END.search(prev):
            parts.append("\n\n" + text)
        else:
            parts.append(" " + text)
    return "".join(parts)


def _parse_chapter_title(title: str) -> tuple[int, str] | None:
    """Parse 'Chapter 5 - Diagon Alley' → (5, 'Diagon Alley')."""
    match = _CHAPTER_TITLE.match(title)
    if not match:
        return None
    return int(match.group(1)), match.group(2).strip()


def chapters_from_toc(
    toc: list[TocEntry],
    pages: list[RawPage],
    last_page: int,
) -> list[Chapter]:
    """Build Chapter objects from raw TOC entries + OCR'd pages.

    Pure function — easy to test with synthetic TOC input.

    Args:
        toc: PyMuPDF TOC entries: (level, title, 1-indexed-page).
        pages: OCR'd RawPages (provides chapter body text).
        last_page: total page count of the source PDF.

    Returns:
        Chapter objects in TOC order. Chapters whose page range has no
        OCR text (because OCR was partial) are skipped.
    """
    if not toc:
        return []

    # Walk TOC: count books, collect chapter entries with book attribution.
    book_num = 0
    chapter_entries: list[tuple[int, int, str, int]] = []  # (book, ch, title, page)

    for level, title, page in toc:
        if level == 1 and title.lower().startswith("harry potter"):
            book_num += 1
        elif level == 2 and book_num >= 1:
            parsed = _parse_chapter_title(title)
            if parsed is not None:
                ch_num, ch_title = parsed
                chapter_entries.append((book_num, ch_num, ch_title, page))

    if not chapter_entries:
        return []

    pages_by_num = {p.page_num: p for p in pages}

    chapters: list[Chapter] = []
    for i, (book, ch_num, ch_title, page_start) in enumerate(chapter_entries):
        # Page range = current chapter's start → next chapter's start - 1
        # (or last_page if this is the final chapter)
        page_end = (
            chapter_entries[i + 1][3] - 1
            if i + 1 < len(chapter_entries)
            else last_page
        )

        page_texts = [
            pages_by_num[p].text
            for p in range(page_start, page_end + 1)
            if p in pages_by_num and pages_by_num[p].text.strip()
        ]
        chapter_text = _join_pages(page_texts)

        # Skip chapters we have no OCR text for (partial OCR run)
        if not chapter_text:
            continue

        chapters.append(
            Chapter(
                book_num=book,
                chapter_num=ch_num,
                chapter_title=ch_title,
                text=chapter_text,
                page_start=page_start,
                page_end=page_end,
            )
        )

    return chapters


def extract_chapters(pdf_path: Path | str, pages: list[RawPage]) -> list[Chapter]:
    """Open the PDF, read its TOC, build Chapter objects.

    Wrapper around `chapters_from_toc` that handles PDF I/O.
    """
    pdf_path = Path(pdf_path)
    with pymupdf.open(pdf_path) as doc:
        toc = doc.get_toc()
        last_page = len(doc)
    return chapters_from_toc(toc, pages, last_page)
