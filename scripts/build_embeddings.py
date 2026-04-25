"""Embed chunks and upsert into Qdrant.

Reads:
    data/processed/chunks.json   (Gold — list[ChapterChunk])

Writes (Qdrant):
    hp_chapters    — chunk_type == "chapter"
    hp_paragraphs  — chunk_type == "paragraph"

Both collections store 1024-dim cosine vectors with payload indexes on
book_num / chapter_num / characters. Idempotent: deterministic UUID5 IDs
mean re-running this script overwrites in place.

    make local            # bring up Qdrant
    uv run python scripts/build_embeddings.py
"""

import json
from pathlib import Path

from qdrant_client import QdrantClient
from rich.console import Console

from horcrux.config import settings
from horcrux.embedding import encode_chunks
from horcrux.ingest import ensure_collection, get_client, upsert_chunks
from horcrux.models import ChapterChunk

_CHUNKS_PATH = Path("data/processed/chunks.json")


def _load_chunks(path: Path) -> list[ChapterChunk] | None:
    """Load chunks.json from disk. Returns None if absent."""
    if not path.exists():
        return None
    return [ChapterChunk(**c) for c in json.loads(path.read_text())]


def _split_by_type(
    chunks: list[ChapterChunk],
) -> tuple[list[ChapterChunk], list[ChapterChunk]]:
    """Partition chunks into (chapter-level, paragraph-level)."""
    chapters = [c for c in chunks if c.chunk_type == "chapter"]
    paragraphs = [c for c in chunks if c.chunk_type == "paragraph"]
    return chapters, paragraphs


def _ensure_collections(client: QdrantClient) -> None:
    """Create both collections + their payload indexes if missing."""
    dim = settings.embedding.dim
    ensure_collection(client, settings.qdrant.chapters_collection, dim=dim)
    ensure_collection(client, settings.qdrant.paragraphs_collection, dim=dim)


def _encode_and_upsert(  # noqa: PLR0913 — distinct concerns: console / client / label / target / data / batching
    console: Console,
    client: QdrantClient,
    label: str,
    collection: str,
    chunks: list[ChapterChunk],
    *,
    batch_size: int,
) -> int:
    """Encode one chunk-type's worth of chunks and upsert into its collection."""
    console.print(f"[bold]encoding {label} chunks[/]")
    vectors = encode_chunks(chunks, batch_size=batch_size, show_progress=True)
    return upsert_chunks(client, collection, chunks, vectors)


def _render_summary(console: Console, n_chapter: int, n_paragraph: int) -> None:
    console.print()
    console.print("[bold green]embedding + upsert complete[/]")
    console.print(f"  {settings.qdrant.chapters_collection}:   {n_chapter} points")
    console.print(f"  {settings.qdrant.paragraphs_collection}: {n_paragraph} points")
    console.print(
        f"\n[dim]inspect at "
        f"http://{settings.qdrant.host}:{settings.qdrant.port}/dashboard[/]"
    )


def main() -> None:
    console = Console()

    chunks = _load_chunks(_CHUNKS_PATH)
    if chunks is None:
        console.print(f"[red]missing:[/] {_CHUNKS_PATH} — run `make chunks`")
        return
    console.print(f"[dim]loaded {len(chunks)} chunks[/]")

    chapter_chunks, paragraph_chunks = _split_by_type(chunks)
    console.print(
        f"[dim]  {len(chapter_chunks)} chapter-level, "
        f"{len(paragraph_chunks)} paragraph-level[/]"
    )

    client = get_client()
    _ensure_collections(client)
    console.print("[dim]collections + payload indexes ready[/]")
    console.print(f"[dim]loading {settings.embedding.model_name} (~1.3GB)…[/]")

    n_chapter = _encode_and_upsert(
        console,
        client,
        "chapter",
        settings.qdrant.chapters_collection,
        chapter_chunks,
        batch_size=8,
    )
    n_paragraph = _encode_and_upsert(
        console,
        client,
        "paragraph",
        settings.qdrant.paragraphs_collection,
        paragraph_chunks,
        batch_size=32,
    )

    _render_summary(console, n_chapter, n_paragraph)


if __name__ == "__main__":
    main()
