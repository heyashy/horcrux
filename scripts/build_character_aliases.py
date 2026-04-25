"""Run character alias discovery on cached chapters.

    Tier 1a   NER + fuzzy clustering           (orthographic / OCR)
    Tier 1b   Co-reference resolution          (title-forms → named persons)
    Tier 1c   Single-word claim                (Harry → Harry Potter)
    Tier 3    Manual overrides                 (semantic aliases, NER false-positives)

Tier 2 (LLM semantic merge) was dropped — see Finding 11. Tier 3 manual
overrides cover the cases an LLM would have handled, without the
non-determinism that would corrupt every chunk's character payload across
runs.

    uv run python scripts/build_character_aliases.py

Output:
    data/processed/aliases_tier1.json
"""

import json
from collections import Counter
from pathlib import Path

import spacy
from fastcoref import FCoref
from rich.console import Console
from rich.table import Table
from spacy.language import Language

from horcrux.chapters import load_chapters_json
from horcrux.characters import (
    apply_overrides,
    claim_single_word_clusters,
    cluster_aliases,
    count_mentions,
    merge_coref_into_clusters,
    resolve_coref_aliases,
    to_id_indexed,
)
from horcrux.models import Chapter

_CHAPTERS_PATH = Path("data/processed/chapters.json")
_OVERRIDES_PATH = Path("data/overrides/character_overrides.json")
_OUTPUT_PATH = Path("data/processed/aliases_tier1.json")


# ── Stage functions ──────────────────────────────────────────────


def _load_chapters(console: Console) -> list[Chapter] | None:
    """Read chapters.json. Returns None and prints guidance if absent."""
    if not _CHAPTERS_PATH.exists():
        console.print(f"[red]missing:[/] {_CHAPTERS_PATH} — run `make chapters` first")
        return None
    chapters = load_chapters_json(_CHAPTERS_PATH)
    console.print(f"[dim]loaded {len(chapters)} chapters from chapters.json[/]")
    return chapters


def _load_ner_pipeline() -> Language:
    """spaCy with parser/lemmatizer/attribute_ruler disabled — NER only."""
    return spacy.load(
        "en_core_web_sm",
        disable=["parser", "lemmatizer", "attribute_ruler"],
    )


def _tier1a_cluster(
    chapters: list[Chapter], nlp: Language
) -> tuple[Counter[str], dict[str, list[str]]]:
    """Tier 1a: NER pass + fuzzy + share-significant-token clustering."""
    counts = count_mentions(chapters, nlp)
    clusters = cluster_aliases(counts, min_count=3, similarity_threshold=85)
    return counts, clusters


def _tier1b_coref(
    chapters: list[Chapter], nlp: Language, console: Console
) -> dict[str, list[str]]:
    """Tier 1b: streamed coref over all chapters, returns title-form resolutions."""
    console.print("[dim]loading fastcoref (downloads ~500MB on first run)…[/]")
    coref_model = FCoref()  # auto-detects CUDA
    console.print(f"[dim]running coref over {len(chapters)} chapters…[/]")
    return resolve_coref_aliases(chapters, nlp, coref_model)


def _tier3_apply_overrides(
    id_indexed: dict[str, dict], console: Console
) -> dict[str, dict]:
    """Tier 3: hand-curated drops + force_merge groups. Skipped if file absent."""
    if not _OVERRIDES_PATH.exists():
        console.print(
            f"\n[yellow]no overrides file at {_OVERRIDES_PATH} — skipping Tier 3[/]"
        )
        return id_indexed

    console.print("\n[bold]Tier 3 — manual overrides[/]")
    overrides = json.loads(_OVERRIDES_PATH.read_text())
    before = len(id_indexed)
    after = apply_overrides(id_indexed, overrides)
    dropped = before - len(after)
    merged = sum(
        len(g) - 1 for g in overrides.get("force_merge", []) if isinstance(g, list)
    )
    console.print(
        f"[dim]dropped {dropped} clusters, merged {merged} via force_merge[/]"
    )
    return after


def _save_aliases(id_indexed: dict[str, dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(id_indexed, indent=2, ensure_ascii=False))


# ── Render helpers ───────────────────────────────────────────────


def _render_top_mentions(
    console: Console, counts: Counter[str], n: int = 20
) -> None:
    console.print(f"[dim]{len(counts)} unique mentions, top {n}:[/]")
    for name, c in counts.most_common(n):
        console.print(f"  {c:5}  {name}")


def _render_coref_resolutions(
    console: Console, resolutions: dict[str, list[str]]
) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Title-form")
    table.add_column("Resolves to")
    table.add_column("Status")
    for title, names in sorted(resolutions.items()):
        if len(names) == 1:
            table.add_row(title, names[0], "[green]dominant[/]")
        else:
            table.add_row(title, ", ".join(names), "[yellow]ambiguous[/]")
    console.print(table)


def _render_final_clusters(
    console: Console,
    clusters: dict[str, list[str]],
    counts: Counter[str],
) -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Canonical")
    table.add_column("Aliases")
    table.add_column("Total mentions", justify="right")

    sorted_clusters = sorted(
        clusters.items(),
        key=lambda kv: -sum(counts.get(a, 0) for a in kv[1]),
    )
    for canonical, aliases in sorted_clusters:
        total = sum(counts.get(a, 0) for a in aliases)
        if len(aliases) == 1:
            alias_str = "[dim]<no aliases>[/]"
        else:
            alias_str = ", ".join(
                f"{a}" + (f" [dim]({counts[a]})[/]" if a in counts else "")
                for a in aliases
                if a != canonical
            )
        table.add_row(canonical, alias_str, str(total))
    console.print(table)


# ── Main orchestration ───────────────────────────────────────────


def main() -> None:
    console = Console()

    chapters = _load_chapters(console)
    if chapters is None:
        return

    console.print("[dim]loading spaCy en_core_web_sm…[/]")
    nlp = _load_ner_pipeline()

    console.print("\n[bold]Tier 1a — NER + fuzzy clustering[/]")
    counts, clusters = _tier1a_cluster(chapters, nlp)
    _render_top_mentions(console, counts)
    console.print(f"[dim]{len(clusters)} clusters after Tier 1a[/]")

    console.print("\n[bold]Tier 1b — co-reference resolution[/]")
    coref_resolutions = _tier1b_coref(chapters, nlp, console)
    console.print(f"[dim]{len(coref_resolutions)} title-forms resolved[/]")
    _render_coref_resolutions(console, coref_resolutions)

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
    _render_final_clusters(console, final_clusters, counts)

    # Convert to industry-standard ID-indexed shape: opaque slug IDs as keys
    # with separate `label` (display) and `aliases` (surface forms). Decouples
    # identity from display — see Finding 15.
    id_indexed = to_id_indexed(final_clusters)
    id_indexed = _tier3_apply_overrides(id_indexed, console)

    _save_aliases(id_indexed, _OUTPUT_PATH)
    console.print(
        f"\n[green]saved[/] {_OUTPUT_PATH}  "
        f"[dim]({len(id_indexed)} characters, ID-indexed)[/]"
    )


if __name__ == "__main__":
    main()
