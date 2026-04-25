"""Run character alias discovery on cached chapters.

    Tier 1a   NER + fuzzy clustering           (orthographic / OCR)
    Tier 1b   Co-reference resolution          (title-forms → named persons)

Tier 2 (LLM semantic merge) runs separately. Output of this script is a
combined Tier 1 alias dictionary.

    uv run python scripts/build_character_aliases.py

Output:
    data/processed/aliases_tier1.json
"""

import json
from pathlib import Path

import spacy
from fastcoref import FCoref
from rich.console import Console
from rich.table import Table

from horcrux.chapters import load_chapters_json
from horcrux.characters import (
    claim_single_word_clusters,
    cluster_aliases,
    count_mentions,
    merge_coref_into_clusters,
    resolve_coref_aliases,
    to_id_indexed,
)


def main() -> None:
    console = Console()

    chapters_path = Path("data/processed/chapters.json")
    if not chapters_path.exists():
        console.print(
            f"[red]missing:[/] {chapters_path} — run `make chapters` first"
        )
        return

    chapters = load_chapters_json(chapters_path)
    console.print(f"[dim]loaded {len(chapters)} chapters from chapters.json[/]")

    # Disable parser/lemmatizer/attribute_ruler — we only need NER.
    console.print("[dim]loading spaCy en_core_web_sm…[/]")
    nlp = spacy.load(
        "en_core_web_sm",
        disable=["parser", "lemmatizer", "attribute_ruler"],
    )

    console.print("\n[bold]Tier 1 — NER pass[/]")
    counts = count_mentions(chapters, nlp)
    console.print(f"[dim]{len(counts)} unique mentions, top 20:[/]")
    for name, c in counts.most_common(20):
        console.print(f"  {c:5}  {name}")

    console.print("\n[bold]Tier 1a — fuzzy clustering[/]")
    clusters = cluster_aliases(counts, min_count=3, similarity_threshold=85)
    console.print(f"[dim]{len(clusters)} clusters after Tier 1a[/]")

    console.print("\n[bold]Tier 1b — co-reference resolution[/]")
    console.print("[dim]loading fastcoref (downloads ~500MB on first run)…[/]")
    coref_model = FCoref()  # auto-detects CUDA

    console.print(f"[dim]running coref over {len(chapters)} chapters…[/]")
    coref_resolutions = resolve_coref_aliases(chapters, nlp, coref_model)
    console.print(f"[dim]{len(coref_resolutions)} title-forms resolved[/]")

    # Show what coref resolved for visibility before merge.
    coref_table = Table(show_header=True, header_style="bold")
    coref_table.add_column("Title-form")
    coref_table.add_column("Resolves to")
    coref_table.add_column("Status")
    for title, names in sorted(coref_resolutions.items()):
        if len(names) == 1:
            coref_table.add_row(title, names[0], "[green]dominant[/]")
        else:
            coref_table.add_row(title, ", ".join(names), "[yellow]ambiguous[/]")
    console.print(coref_table)

    console.print("\n[bold]Merging coref resolutions into clusters[/]")
    intermediate = merge_coref_into_clusters(clusters, coref_resolutions)
    console.print(f"[dim]{len(intermediate)} clusters after Tier 1b[/]")

    console.print("\n[bold]Tier 1c — single-word claim[/]")
    final_clusters = claim_single_word_clusters(intermediate)
    merged_count = len(intermediate) - len(final_clusters)
    console.print(
        f"[dim]{len(final_clusters)} clusters after Tier 1c "
        f"({merged_count} single-word clusters claimed by their unique multi-word owner)[/]\n"
    )

    # Final cluster display
    table = Table(show_header=True, header_style="bold")
    table.add_column("Canonical")
    table.add_column("Aliases")
    table.add_column("Total mentions", justify="right")

    sorted_clusters = sorted(
        final_clusters.items(),
        key=lambda kv: -sum(counts.get(a, 0) for a in kv[1]),
    )
    for canonical, aliases in sorted_clusters:
        total = sum(counts.get(a, 0) for a in aliases)
        if len(aliases) == 1:
            alias_str = "[dim]<no aliases>[/]"
        else:
            alias_str = ", ".join(
                f"{a}" + (f" [dim]({counts[a]})[/]" if a in counts else "")
                for a in aliases if a != canonical
            )
        table.add_row(canonical, alias_str, str(total))
    console.print(table)

    # Convert to industry-standard ID-indexed shape: opaque slug IDs as
    # keys, with separate `label` (display form) and `aliases` (surface
    # forms). Decouples identity from display — see Finding 15.
    id_indexed = to_id_indexed(final_clusters)

    output_path = Path("data/processed/aliases_tier1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(id_indexed, indent=2, ensure_ascii=False))
    console.print(
        f"\n[green]saved[/] {output_path}  "
        f"[dim]({len(id_indexed)} characters, ID-indexed)[/]"
    )
    console.print(
        "\n[bold]Look for these gaps[/] (Tier 2 LLM step will fix):\n"
        "  · 'Voldemort' / 'Tom Riddle' / 'He Who Must Not Be Named' as separate clusters\n"
        "  · Single-word ↔ multi-word merges (e.g. 'Harry' / 'Harry Potter')\n"
        "  · Marauder nicknames: Padfoot/Moony/Prongs/Wormtail (book 3+, not in book 1)"
    )


if __name__ == "__main__":
    main()
