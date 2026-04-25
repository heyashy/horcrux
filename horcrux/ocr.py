"""OCR pipeline: PDF → list[RawPage].

Each page is rendered to a PNG at 2x zoom (≈144 DPI), then OCR'd via
Tesseract. Output is the raw OCR text per page, with whitespace preserved.
Imperfect text is fine — chunking and chapter detection downstream tolerate
typical OCR errors (italics dropped, page-number bleed, dialect mangling).

Synchronous function. Phase 4 wraps it in a Temporal activity without
changing this code.
"""

import io
import logging
from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from horcrux.models import RawPage

log = logging.getLogger(__name__)


def ocr_pages(
    pdf_path: Path | str,
    *,
    start_page: int = 1,
    end_page: int | None = None,
    zoom: float = 2.0,
) -> list[RawPage]:
    """Render and OCR a contiguous page range from the PDF.

    Args:
        pdf_path: path to the source PDF.
        start_page: 1-indexed inclusive start page.
        end_page: 1-indexed inclusive end page. None = last page in PDF.
        zoom: render zoom factor. 2.0 ≈ 144 DPI (recommended).

    Returns:
        One RawPage per page in the range, in order.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    matrix = pymupdf.Matrix(zoom, zoom)
    pages: list[RawPage] = []

    with pymupdf.open(pdf_path) as doc:
        total_in_pdf = len(doc)
        if end_page is None:
            end_page = total_in_pdf
        if not 1 <= start_page <= end_page <= total_in_pdf:
            raise ValueError(
                f"invalid page range: start={start_page}, end={end_page}, "
                f"total={total_in_pdf}"
            )

        log.info(
            "OCR pages %d-%d of %d from %s",
            start_page, end_page, total_in_pdf, pdf_path,
        )

        progress_columns = [
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ]

        with Progress(*progress_columns) as progress:
            task = progress.add_task("OCR", total=end_page - start_page + 1)
            for page_idx in range(start_page - 1, end_page):  # 0-indexed for pymupdf
                page = doc[page_idx]
                pix = page.get_pixmap(matrix=matrix)
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                text = pytesseract.image_to_string(image)
                pages.append(RawPage(page_num=page_idx + 1, text=text))
                progress.advance(task)

    return pages
