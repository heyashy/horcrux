"""Pydantic models — single source of truth for the shapes that flow through
the pipeline. Models grow phase by phase; this file only contains what the
current phase needs.
"""

from pydantic import BaseModel, Field


class RawPage(BaseModel):
    """A single OCR'd page from the corpus PDF.

    Carries no chapter or book metadata — that comes from chapter detection
    in Phase 2. Keeping OCR concerns separate from structure-detection
    concerns means each layer can fail independently and be tested in
    isolation.
    """

    page_num: int = Field(ge=1, description="1-indexed page number in the source PDF")
    text: str = Field(description="OCR text for this page, whitespace preserved")
