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

from rich.console import Console

from horcrux.config import settings
from horcrux.embedding import encode_chunks
from horcrux.ingest import ensure_collection, get_client, upsert_chunks
from horcrux.models import ChapterChunk


def main() -> None:
    console = Console()

    chunks_path = Path("data/processed/chunks.json")
    if not chunks_path.exists():
        console.print(f"[red]missing:[/] {chunks_path} — run `make chunks`")
        return

    raw = json.loads(chunks_path.read_text())
    chunks = [ChapterChunk(**c) for c in raw]
    console.print(f"[dim]loaded {len(chunks)} chunks[/]")

    chapter_chunks = [c for c in chunks if c.chunk_type == "chapter"]
    paragraph_chunks = [c for c in chunks if c.chunk_type == "paragraph"]
    console.print(
        f"[dim]  {len(chapter_chunks)} chapter-level, "
        f"{len(paragraph_chunks)} paragraph-level[/]"
    )

    client = get_client()
    dim = settings.embedding.dim
    ensure_collection(client, settings.qdrant.chapters_collection, dim=dim)
    ensure_collection(client, settings.qdrant.paragraphs_collection, dim=dim)
    console.print("[dim]collections + payload indexes ready[/]")

    # Encode each set in one pass — bge-large is fast enough on this volume
    # that we can afford to hold both matrices in memory rather than streaming.
    console.print(f"[dim]loading {settings.embedding.model_name} (~1.3GB)…[/]")

    console.print("[bold]encoding chapter chunks[/]")
    chapter_vecs = encode_chunks(chapter_chunks, batch_size=8, show_progress=True)
    console.print("[bold]encoding paragraph chunks[/]")
    paragraph_vecs = encode_chunks(paragraph_chunks, batch_size=32, show_progress=True)

    console.print("[bold]upserting to Qdrant[/]")
    n_chapter = upsert_chunks(
        client,
        settings.qdrant.chapters_collection,
        chapter_chunks,
        chapter_vecs,
    )
    n_paragraph = upsert_chunks(
        client,
        settings.qdrant.paragraphs_collection,
        paragraph_chunks,
        paragraph_vecs,
    )

    console.print()
    console.print("[bold green]embedding + upsert complete[/]")
    console.print(f"  {settings.qdrant.chapters_collection}:   {n_chapter} points")
    console.print(f"  {settings.qdrant.paragraphs_collection}: {n_paragraph} points")
    console.print(
        f"\n[dim]inspect at http://{settings.qdrant.host}:{settings.qdrant.port}/dashboard[/]"
    )


if __name__ == "__main__":
    main()
