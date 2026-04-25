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
from functools import partial
from pathlib import Path

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


def main() -> None:
    console = Console()

    chapters_path = Path("data/processed/chapters.json")
    aliases_path = Path("data/processed/aliases_tier1.json")

    if not chapters_path.exists():
        console.print(f"[red]missing:[/] {chapters_path} — run `make chapters`")
        return
    if not aliases_path.exists():
        console.print(
            f"[red]missing:[/] {aliases_path} — run discovery first "
            "(`uv run python scripts/build_character_aliases.py`)"
        )
        return

    chapters = load_chapters_json(chapters_path)
    alias_dict = json.loads(aliases_path.read_text())
    console.print(
        f"[dim]loaded {len(chapters)} chapters, {len(alias_dict)} characters[/]"
    )

    console.print(f"[dim]loading {settings.embedding.model_name} (~1.3GB)…[/]")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding.model_name)
    # bge-large is asymmetric — passages get NO prefix (queries do).
    # We're embedding sentences (passages) here for similarity, so no prefix.
    encode_fn = lambda sentences: model.encode(
        sentences, normalize_embeddings=True, show_progress_bar=False
    )
    extract_fn = partial(extract_characters, alias_dict=alias_dict)

    all_chunks = []
    columns = [
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ]
    with Progress(*columns) as progress:
        task = progress.add_task("chunking", total=len(chapters))
        for chapter in chapters:
            chunks = chunk_chapter(chapter, encode_fn, extract_fn)
            all_chunks.extend(chunks)
            progress.advance(task)

    # Stats — sanity check chunk distribution
    chapter_chunks = [c for c in all_chunks if c.chunk_type == "chapter"]
    para_chunks = [c for c in all_chunks if c.chunk_type == "paragraph"]
    para_token_counts = [len(c.text.split()) for c in para_chunks]
    char_tagged = sum(1 for c in all_chunks if c.characters)

    console.print()
    console.print("[bold green]chunking complete[/]")
    console.print(f"  total chunks:        {len(all_chunks)}")
    console.print(f"  chapter-level:       {len(chapter_chunks)}")
    console.print(f"  paragraph-level:     {len(para_chunks)}")
    if para_token_counts:
        console.print(
            f"  paragraph tokens:    min={min(para_token_counts)}, "
            f"mean={sum(para_token_counts) // len(para_token_counts)}, "
            f"max={max(para_token_counts)}"
        )
    console.print(f"  with character tags: {char_tagged} ({char_tagged * 100 // len(all_chunks)}%)")

    output_path = Path("data/processed/chunks.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps([c.model_dump() for c in all_chunks], indent=2, ensure_ascii=False)
    )
    console.print(f"\n[green]saved[/] {output_path}")


if __name__ == "__main__":
    main()
