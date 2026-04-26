"""Page-level text normalisation.

Cleans OCR side-effects (page-number bleed, em-dash variants, runaway
whitespace) without touching authorial style or content. We don't *drop*
pages here — chapter detection downstream uses page-blankness as signal
(book boundaries surface as clusters of short pages). Cleansing prepares
text; pruning happens later.
"""

import re

from horcrux.models import RawPage

# Lines that are *just* digits and surrounding whitespace — page-number
# bleed from the printed header/footer into the OCR'd body.
_PAGE_NUMBER_LINE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)

# Three or more consecutive newlines collapse to two (one blank line).
# Preserves paragraph breaks without letting OCR whitespace runaway.
_RUNAWAY_NEWLINES = re.compile(r"\n{3,}")

# Tesseract emits `--` and `—` inconsistently for the same em-dash glyph.
# Normalise to the Unicode character — sentence segmenters and chunkers
# downstream are more reliable on `—` than on `--`.
_DOUBLE_DASH = re.compile(r"--")


def cleanse_text(text: str) -> str:
    """Apply the cleansing rules to a single page's text.

    Idempotent — running it twice produces the same output as running once.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _PAGE_NUMBER_LINE.sub("", text)
    text = _DOUBLE_DASH.sub("—", text)
    text = _RUNAWAY_NEWLINES.sub("\n\n", text)
    return text.strip()


def cleanse_pages(pages: list[RawPage]) -> list[RawPage]:
    """Apply `cleanse_text` to every page; preserves order and page numbers."""
    return [
        RawPage(page_num=p.page_num, text=cleanse_text(p.text))
        for p in pages
    ]
