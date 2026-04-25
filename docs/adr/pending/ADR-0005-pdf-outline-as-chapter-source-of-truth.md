# ADR-0005: PDF outline as the chapter source of truth

**Date:** 2026-04-25
**Status:** pending
**Pattern:** Single source of truth — prefer authored metadata over derived signals.

## Context

The corpus PDF is a multi-book compilation with a structured table of contents
embedded as PDF bookmarks (also called the *outline*). Every book's start and
every chapter's start is annotated by the PDF author, with exact page numbers.

The original Phase 2 design called for chapter detection by regex-scanning
OCR'd text for `CHAPTER ONE/TWO/...` markers, then inferring book boundaries
from chapter-number resets. This works for any PDF — including those without
bookmarks — but has real failure modes:

- Tesseract mangles chapter headings (`P` → `Rt`, etc.) — missed chapters.
- Body-text mentions of "chapter" can produce false positives (mitigated with strict
  case + first-line-only matching, but the risk is non-zero).
- Compound number-words (`TWENTY-ONE`, `THIRTY-SEVEN`) require a thesaurus.
- Chapter title extraction is heuristic and frequently lossy.

PyMuPDF's `Document.get_toc()` returns the embedded outline directly:
`[[level, title, page], ...]`. For this corpus, level-1 entries are book starts
and level-2 entries are chapters in `"Chapter N - Title"` format. This is the
PDF author's structural ground truth — no derivation needed.

## Decision

Use the PDF outline (`get_toc()`) as the primary source of chapter structure.

- `chapters_from_toc(toc, pages, last_page)` — pure function, builds `Chapter`
  objects from raw TOC entries + OCR'd pages. Fully unit-testable with
  synthetic input, no PDF fixtures required.
- `extract_chapters(pdf_path, pages)` — thin wrapper that opens the PDF,
  calls `get_toc()`, and feeds the pure function.
- Book attribution: walk TOC entries; `level=1` titles starting with
  "Harry Potter" mark book starts; `level=2` titles parseable as
  `"Chapter N - Title"` are chapter entries within the current book.
- Chapter page range: from the chapter's TOC page through the next chapter's
  page minus one (or `last_page` for the final chapter).
- Chapters with no OCR text in their range are skipped (handles partial
  OCR runs gracefully).

Regex-on-OCR scanning is the **documented fallback** for corpora without
embedded outlines. Not implemented in this codebase; the alternative is
described here for future reference.

## Alternatives Considered

**Regex on OCR'd text (the original design).** Works on any PDF but reintroduces
the failure modes listed in Context. Rejected because we have authoritative
structural metadata; deriving structure from OCR'd visual artifacts of that
metadata is the wrong direction.

**Manual ground-truth file** (committed JSON of chapter→page mappings).
Rejected because it doesn't scale beyond this specific corpus and requires
human maintenance. The TOC is already the answer to the question this would
manually re-answer.

**Hybrid (TOC primary, regex fallback per-PDF).** Tempting but adds complexity
without benefit for this lab. The regex path adds non-trivial code surface and
a second test suite, all to support corpora we don't have. If a future corpus
lacks bookmarks, add the fallback then.

**Author-supplied chapter metadata via config.** Could declare chapter ranges
in `pyproject.toml` or a YAML file. Same scaling problem as manual ground
truth, with extra config-management burden.

## Consequences

**Positive**
- Chapter detection is exact: 15/15 chapters of book 1 match the source TOC
  with zero drift, verified via `scripts/inspect_chapters.py`.
- Code is materially simpler — `chapters_from_toc` is ~60 lines including
  docstring, no number-word thesaurus, no heuristic title parsing.
- Decoupled testing: the pure function takes synthetic TOC entries; no PDF
  fixtures required for unit tests.
- Robust to OCR quality. Even when Tesseract mangles a chapter heading
  visually, the structural detection remains correct.

**Negative / risks**
- Tied to PDFs with embedded outlines. Corpora without bookmarks won't work
  with this code — would need the regex fallback added back.
- Trusts the PDF author's TOC accuracy. If the PDF has wrong page numbers
  in its outline (unlikely but possible), our chapter ranges inherit those
  errors.
- The `_CHAPTER_TITLE` regex parses `"Chapter N - Title"` and a few
  punctuation variants. PDFs with unusual chapter title formats (`"§ 5"`,
  `"V."`) need format-specific parsing.

**Follow-ups**
- Document the regex-fallback approach in `docs/lab/toolchain-path.md` so
  future readers know how to handle TOC-less PDFs.
- For partial OCR runs, the `Chapter.page_end` reflects the TOC's stated end
  even when we have no OCR text for the final pages of a chapter. Either
  trim `page_end` to the highest OCR'd page in the chapter or add a
  `text_complete: bool` flag. Decide when book 2+ ingest happens.

## Rollback

Pure code change with no schema or state impact:

1. Restore the regex-based `chapters.py` from git history (`git show HEAD~1:horcrux/chapters.py`).
2. Restore `tests/unit/test_chapters.py` to the matching version.
3. Update `inspect_chapters.py` to use `detect_chapters` instead of `extract_chapters`.

No data is affected. ~15 minutes of work.
