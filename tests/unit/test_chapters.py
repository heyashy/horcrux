"""Chapter detection from PDF TOC entries — pure-function tests."""

import pytest

from horcrux.chapters import _parse_chapter_title, chapters_from_toc
from horcrux.models import RawPage

pytestmark = pytest.mark.unit


# ── _parse_chapter_title ─────────────────────────────────────────

def test_parses_dash_separator():
    assert _parse_chapter_title("Chapter 1 - The Boy Who Lived") == (
        1, "The Boy Who Lived"
    )


def test_parses_colon_separator():
    assert _parse_chapter_title("Chapter 5: Diagon Alley") == (
        5, "Diagon Alley"
    )


def test_parses_em_dash_separator():
    assert _parse_chapter_title("Chapter 7 — The Sorting Hat") == (
        7, "The Sorting Hat"
    )


def test_parses_two_digit_chapter():
    assert _parse_chapter_title("Chapter 37 - The Beginning") == (
        37, "The Beginning"
    )


def test_case_insensitive():
    assert _parse_chapter_title("CHAPTER 1 - foo") == (1, "foo")
    assert _parse_chapter_title("chapter 1 - foo") == (1, "foo")


def test_returns_none_for_non_chapter_title():
    assert _parse_chapter_title("Title Page") is None
    assert _parse_chapter_title("Pottermore.com") is None
    assert _parse_chapter_title("Harry Potter and the Sorcerer's Stone") is None


# ── chapters_from_toc ────────────────────────────────────────────

def _make_pages(start: int, end: int, text: str = "body") -> list[RawPage]:
    """Helper: synthesise RawPages with simple text."""
    return [RawPage(page_num=p, text=text) for p in range(start, end + 1)]


def test_single_book_two_chapters():
    toc = [
        (1, "Harry Potter and the Sorcerer's Stone", 1),
        (2, "Chapter 1 - The Boy Who Lived", 5),
        (2, "Chapter 2 - The Vanishing Glass", 15),
    ]
    pages = _make_pages(1, 25)
    chapters = chapters_from_toc(toc, pages, last_page=25)

    assert len(chapters) == 2
    assert chapters[0].book_num == 1
    assert chapters[0].chapter_num == 1
    assert chapters[0].chapter_title == "The Boy Who Lived"
    assert chapters[0].page_start == 5
    assert chapters[0].page_end == 14
    assert chapters[1].chapter_num == 2
    assert chapters[1].page_start == 15
    assert chapters[1].page_end == 25  # runs to last_page


def test_two_books_chapter_attribution():
    toc = [
        (1, "Title Page", 2),                              # front matter
        (1, "Harry Potter and the Sorcerer's Stone", 7),    # book 1
        (2, "Chapter 1 - The Boy Who Lived", 12),
        (2, "Chapter 2 - The Vanishing Glass", 26),
        (1, "Harry Potter and the Chamber of Secrets", 276),  # book 2
        (2, "Chapter 1 - The Worst Birthday", 281),
    ]
    pages = _make_pages(1, 400)
    chapters = chapters_from_toc(toc, pages, last_page=400)

    assert len(chapters) == 3
    assert chapters[0].book_num == 1
    assert chapters[0].chapter_num == 1
    assert chapters[1].book_num == 1
    assert chapters[1].chapter_num == 2
    assert chapters[1].page_end == 280  # runs up to next chapter start - 1
    assert chapters[2].book_num == 2
    assert chapters[2].chapter_num == 1


def test_skips_chapters_with_no_ocr_text():
    """Partial OCR runs leave gaps; chapters we have no pages for are skipped."""
    toc = [
        (1, "Harry Potter and the Sorcerer's Stone", 1),
        (2, "Chapter 1 - First", 5),
        (2, "Chapter 2 - Second", 15),
        (2, "Chapter 3 - Third", 25),
    ]
    # Only OCR'd up to page 20 — chapter 3 has no text
    pages = _make_pages(1, 20)
    chapters = chapters_from_toc(toc, pages, last_page=30)

    assert len(chapters) == 2
    assert chapters[0].chapter_num == 1
    assert chapters[1].chapter_num == 2


def test_empty_toc_returns_empty():
    assert chapters_from_toc([], [], 10) == []


def test_toc_without_chapters_returns_empty():
    """A TOC that has only front matter and no chapters."""
    toc = [
        (1, "Title Page", 2),
        (1, "Pottermore.com", 4),
        (1, "Contents", 6),
    ]
    assert chapters_from_toc(toc, _make_pages(1, 10), 10) == []


def test_chapter_text_concatenated_in_page_order():
    toc = [
        (1, "Harry Potter and the Sorcerer's Stone", 1),
        (2, "Chapter 1 - First", 1),
    ]
    pages = [
        RawPage(page_num=1, text="page one body"),
        RawPage(page_num=2, text="page two body"),
        RawPage(page_num=3, text="page three body"),
    ]
    chapters = chapters_from_toc(toc, pages, last_page=3)

    assert len(chapters) == 1
    assert "page one body" in chapters[0].text
    assert "page two body" in chapters[0].text
    assert "page three body" in chapters[0].text
