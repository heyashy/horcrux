"""Cleansing rules — small file, but every rule has a regression test
because OCR pipelines are unforgiving when cleaning silently changes."""

import pytest

from horcrux.corpus.cleansing import cleanse_pages, cleanse_text
from horcrux.models import RawPage

pytestmark = pytest.mark.unit


def test_strips_standalone_page_number_lines():
    text = "Some prose.\n42\nMore prose."
    assert cleanse_text(text) == "Some prose.\n\nMore prose."


def test_preserves_numbers_inside_prose():
    text = "Harry was 11 years old."
    assert cleanse_text(text) == "Harry was 11 years old."


def test_preserves_numbers_at_line_end():
    text = "He had 42 galleons left."
    assert cleanse_text(text) == "He had 42 galleons left."


def test_collapses_runaway_newlines():
    text = "Para one.\n\n\n\n\nPara two."
    assert cleanse_text(text) == "Para one.\n\nPara two."


def test_preserves_paragraph_breaks():
    text = "Para one.\n\nPara two."
    assert cleanse_text(text) == "Para one.\n\nPara two."


def test_normalises_double_dash_to_em_dash():
    text = "He hesitated -- then walked away."
    assert cleanse_text(text) == "He hesitated — then walked away."


def test_keeps_real_em_dashes():
    text = "He hesitated — then walked away."
    assert cleanse_text(text) == "He hesitated — then walked away."


def test_strips_outer_whitespace():
    assert cleanse_text("\n\n  hello  \n\n") == "hello"


def test_normalises_carriage_returns():
    text = "line one\r\nline two\rline three"
    assert cleanse_text(text) == "line one\nline two\nline three"


def test_idempotent():
    text = "Some prose.\n\n\n42\n\nMore -- prose.\n\n\n"
    once = cleanse_text(text)
    twice = cleanse_text(once)
    assert once == twice


def test_cleanse_pages_preserves_page_numbers_and_order():
    pages = [
        RawPage(page_num=1, text="first --"),
        RawPage(page_num=2, text="second"),
        RawPage(page_num=3, text="third\n42"),
    ]
    result = cleanse_pages(pages)
    assert [p.page_num for p in result] == [1, 2, 3]
    assert result[0].text == "first —"
    assert result[2].text == "third"
