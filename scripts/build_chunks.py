"""Gold-tier chunk artefact build.

Reads:
    data/processed/chapters.json       (Silver — book-grouped chapters)
    data/processed/aliases_tier1.json  (Gold — character alias dict, ID-indexed)

Writes:
    data/processed/chunks.json         (Gold — list[ChapterChunk] embedding-ready)

Loads bge-large-en-v1.5 (~1.3GB) once for sentence-similarity computation;
the same model is re-used for Qdrant indexing in Phase 3.

    uv run python scripts/build_chunks.py
"""

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from horcrux.chapters import load_chapters_json
from horcrux.characters import extract_characters
from horcrux.chunking import chunk_chapter
from horcrux.config import settings
from horcrux.models import Chapter, ChapterChunk

_CHAPTERS_PATH = Path("data/processed/chapters.json")
_ALIASES_PATH = Path("data/processed/aliases_tier1.json")
_CHUNKS_OUTPUT_PATH = Path("data/processed/chunks.json")


def _check_inputs(console: Console) -> tuple[list[Chapter], dict] | None:
    """Load chapters + alias dict; return None and print guidance if missing."""
    if not _CHAPTERS_PATH.exists():
        console.print(f"[red]missing:[/] {_CHAPTERS_PATH} — run `make chapters`")
        return None
    if not _ALIASES_PATH.exists():
        console.print(
            f"[red]missing:[/] {_ALIASES_PATH} — run "
            "`uv run python scripts/build_character_aliases.py`"
        )
        return None
    chapters = load_chapters_json(_CHAPTERS_PATH)
    alias_dict = json.loads(_ALIASES_PATH.read_text())
    return chapters, alias_dict


def _build_encoder() -> Callable[[list[str]], np.ndarray]:
    """Load bge-large and return a passage-encoding closure.

    bge-large-en-v1.5 is asymmetric — passages get NO prefix; queries do.
    We're embedding sentences for similarity (passages), so no prefix.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding.model_name)
    return lambda sentences: model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )


def _progress_columns() -> list:
    return [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]


def _chunk_all(
    chapters: list[Chapter],
    encode_fn: Callable[[list[str]], np.ndarray],
    extract_fn: Callable[[str], list[str]],
) -> list[ChapterChunk]:
    """Run the chunker over every chapter with a Rich progress bar."""
    all_chunks: list[ChapterChunk] = []
    with Progress(*_progress_columns()) as progress:
        task = progress.add_task("chunking", total=len(chapters))
        for chapter in chapters:
            all_chunks.extend(chunk_chapter(chapter, encode_fn, extract_fn))
            progress.advance(task)
    return all_chunks


def _render_stats(console: Console, chunks: list[ChapterChunk]) -> None:
    """Distribution summary — sanity check the chunker output."""
    chapter_chunks = [c for c in chunks if c.chunk_type == "chapter"]
    para_chunks = [c for c in chunks if c.chunk_type == "paragraph"]
    para_token_counts = [len(c.text.split()) for c in para_chunks]
    char_tagged = sum(1 for c in chunks if c.characters)

    console.print()
    console.print("[bold green]chunking complete[/]")
    console.print(f"  total chunks:        {len(chunks)}")
    console.print(f"  chapter-level:       {len(chapter_chunks)}")
    console.print(f"  paragraph-level:     {len(para_chunks)}")
    if para_token_counts:
        console.print(
            f"  paragraph tokens:    min={min(para_token_counts)}, "
            f"mean={sum(para_token_counts) // len(para_token_counts)}, "
            f"max={max(para_token_counts)}"
        )
    pct = char_tagged * 100 // len(chunks) if chunks else 0
    console.print(f"  with character tags: {char_tagged} ({pct}%)")


def _save_chunks(chunks: list[ChapterChunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.model_dump() for c in chunks], indent=2, ensure_ascii=False)
    )


def main() -> None:
    console = Console()
    inputs = _check_inputs(console)
    if inputs is None:
        return
    chapters, alias_dict = inputs
    console.print(
        f"[dim]loaded {len(chapters)} chapters, {len(alias_dict)} characters[/]"
    )

    console.print(f"[dim]loading {settings.embedding.model_name} (~1.3GB)…[/]")
    encode_fn = _build_encoder()
    extract_fn = partial(extract_characters, alias_dict=alias_dict)

    chunks = _chunk_all(chapters, encode_fn, extract_fn)
    _render_stats(console, chunks)

    _save_chunks(chunks, _CHUNKS_OUTPUT_PATH)
    console.print(f"\n[green]saved[/] {_CHUNKS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
