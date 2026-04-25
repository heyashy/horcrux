# ADR-0006: Three-tier character discovery for the alias dictionary

**Date:** 2026-04-25
**Status:** pending
**Pattern:** Layered enrichment — each tier has a defined responsibility and a
defined failure mode; tiers compose without coupling.

*Tier nomenclature:* Tier 1a (orthographic), 1b (coref), 1c (single-word claim)
are deterministic. Tier 2 (LLM merge) is non-deterministic but cached. Tier 3
(manual overrides) is human-authored.

## Context

Relational queries ("trace Snape's loyalty across the series") rely on a
`characters: list[str]` payload populated on every chunk at ingest time —
that payload is what Qdrant filters against to narrow candidate chunks
before synthesis. Filter quality determines retrieval recall.

The naive approach — a hand-curated `dict[canonical, list[aliases]]` — was
the original Phase 2 design. It was rejected once we faced the corpus
properly: HP has ~50 characters with 5+ aliases each (Voldemort alone has
six), and a hand-curated approach doesn't generalise to a different corpus.

We need a discovery pipeline that:

- Finds canonical character names from the corpus automatically.
- Captures orthographic variants (Harry / Harry Potter / Potter) without
  human intervention.
- Tolerates OCR errors (Hermione / Hcrmione).
- Resolves title-forms in context (Mr. Weasley → which Weasley).
- Catches semantic aliases the orthographic / coref passes can't see
  (Voldemort / Tom Riddle / He Who Must Not Be Named).
- Admits a manual override for residual edge cases (NER false positives,
  ambiguous mentions).

## Decision

Four tiers, applied in order, with each tier owning a specific class of
alias resolution:

### Tier 1a — Orthographic clustering (NER + fuzzy)

`horcrux/characters.py`: `count_mentions` + `cluster_aliases`.

- spaCy `en_core_web_sm` produces PERSON entity mentions per chapter.
- Counter aggregates across the corpus; threshold cuts the long tail.
- Union-find clusters names by:
  - **Multi-word ↔ multi-word subset** ("Harry Potter" ⊂ "Harry James Potter").
  - **Single-word ↔ single-word fuzz** (rapidfuzz `fuzz.ratio` ≥ 85).
- Possessive suffixes (`'s`, curly `'s`) are stripped at count time so
  possessive forms fold into their root name's count.
- Internal whitespace (including line-wrap newlines from OCR) is collapsed
  to single spaces so `Professor\nMcGonagall` clusters with
  `Professor McGonagall`.

**Deliberately conservative.** Single-word ↔ multi-word subset matching is
NOT performed in Tier 1a because the family-name failure mode (e.g.
"Weasley" subsuming "Ron Weasley", "Fred Weasley", "Percy Weasley", ... into
a single mega-cluster) would corrupt every cluster in the dict. Single↔multi
merges are deferred to Tier 2.

### Tier 1b — Co-reference resolution (fastcoref)

`horcrux/characters.py`: `resolve_coref_aliases` + `merge_coref_into_clusters`.

- `fastcoref.FCoref` runs RoBERTa-based coref over each chapter.
- For each coref cluster, identify the unique named PERSON (if any) and
  collect non-named title-form mentions.
- Multiple distinct named persons in a single coref cluster → reject the
  cluster entirely (coref made an error, can't recover).
- Title-form mentions are filtered to exclude pronouns, articles, prose
  fragments (≤4 words, no commas/quotes/conjunctions).
- Attribution requires the alias and target to share at least one
  **significant token** — non-generic (`Mr.`, `Mrs.`, `the`, `Father`,
  etc. excluded) and non-empty intersection.
- Across the corpus, a title-form needs ≥2 corpus-wide occurrences to be
  promoted (filters one-off coref noise).
- A title-form whose top resolution doesn't dominate (≥2× the
  second-most-frequent) is left as ambiguous (multi-tagged at chunk time).

**Deliberately strict.** Coref over long fiction is noisy; we trade recall
for precision. Cases where coref would suggest a non-orthographic merge
("Mr. Dursley" → Dumbledore, "Fluffy" → Flitwick) are blocked at the
share-token gate. Those resolutions, when they're correct, belong to Tier 2.

### Tier 1c — Single-word claim (deterministic merge)

`horcrux/characters.py`: `claim_single_word_clusters`.

After Tier 1a + 1b finishes, a single-word cluster `S` is folded into a
multi-word cluster `M` iff:

- `S`'s canonical has exactly one significant token `t` (after stripping
  `Mr.`, `Mrs.`, articles, and other generics).
- Exactly one multi-word cluster's canonical contains `t` as one of its
  significant tokens.

The "exactly one" gate is what distinguishes a personal name from a family
name. `Hermione` has one match (`Hermione Granger`) → fold. `Weasley` has
eight matches (Ron Weasley, Fred Weasley, Percy Weasley, ...) → leave alone
as the family-level cluster.

When merging, the **multi-word form wins as canonical** for human-readable
reports — `Harry Potter` becomes the canonical of the merged cluster even
though `Harry` is far more frequent.

This step closes the gap left by Tier 1a's deliberate single↔multi
exclusion *without* LLM cost or non-determinism. Tier 2 LLM is left to
handle the genuinely semantic merges that no algorithm can derive
(`Severus` → Snape, `Voldemort` → `Tom Riddle` → `He Who Must Not Be
Named`).

### Tier 2 — Semantic merge (LLM-assisted) — *pending*

To be implemented in `horcrux/character_merge.py`:

- One Haiku call reads the Tier 1 cluster list and identifies clusters
  that refer to the same character via semantic aliases.
- Output: list of merge groups (e.g. `[Voldemort, Tom Riddle, He Who Must
  Not Be Named, the Dark Lord]`).
- Cached to disk; rebuilt only on `--rebuild`.
- Cost: ~$0.003 per discovery run, model pinned to `claude-haiku-4-5-20251001`.

**Why LLM is right here.** Semantic aliases require world knowledge no
deterministic algorithm has. NER doesn't know "Padfoot" refers to Sirius.
Coref doesn't link "Tom Riddle" with "Voldemort" without a co-occurring
explanation in the same paragraph. Haiku has both.

### Tier 3 — Manual overrides — *pending*

`data/processed/character_overrides.json`:

- `force_merge`: groups of clusters to merge unconditionally.
- `force_split`: pairs that must remain separate.
- `drop`: aliases or canonicals to remove (NER false positives — places,
  objects, magical concepts that spaCy mistagged as PERSON).
- `context_aliases`: ambiguous title-forms (e.g. "Mr. Weasley" → both
  Ron and Arthur), multi-tagged at chunk time.

Applied as the final pass; always wins over auto-discovery.

## Alternatives Considered

**Hand-curated alias dictionary.** Original Phase 2 design. Rejected on
generalisability and maintenance grounds — locks the lab to HP, requires
human authoring per corpus.

**NER alone (no clustering).** Take spaCy's PERSON entities verbatim as
canonicals. Rejected because the same character appears as 5-10 distinct
strings ("Harry", "Harry Potter", "Potter", "Mr. Potter", ...) and the
character payload would fragment.

**Full neural co-reference (no orthographic step).** Run fastcoref over the
whole corpus, take its clusters as canonical. Rejected because coref over
long fiction has high error rates — we observed multi-character coref
clusters that would have produced catastrophic merges (Voldemort epithet →
Dumbledore, Peeves → Harry).

**LLM doing everything.** Single Haiku call processing the full corpus.
Rejected on three grounds: cost (~$1+ per run vs $0.003 for the targeted
merge), reproducibility (full-corpus prompts drift), and privacy (sends
the entire corpus to Anthropic).

**Hybrid with a shared "is family name" detector.** Considered as part of
Tier 1a refinement. Rejected as not generalising — distinguishing "Harry"
(personal) from "Weasley" (family) statistically requires NER labels that
aren't reliable on this data.

## Consequences

**Positive**

- Each tier has a single responsibility and a known failure mode.
- Tier 1 is fully deterministic. Tier 2 isolates LLM non-determinism in
  one cached artefact. Tier 3 is human-authored and audit-friendly.
- The pipeline degrades gracefully — turning off Tier 2 produces less
  recall but no incorrect merges. Turning off Tier 1b drops title-forms
  but keeps the orthographic backbone.
- The discovery output is reviewable. A human can inspect the JSON and
  spot bad clusters before chunk tagging consumes them.

**Negative / risks**

- Three-tier complexity. Implementer needs to know which tier owns which
  concern — documented in the module docstring.
- Tier 2's LLM call introduces non-determinism in what was previously a
  fully reproducible pipeline. Mitigation: pin the model version, cache
  the result, treat it as a versioned artefact.
- Tier 3 manual overrides require human maintenance for each corpus.
  Acceptable for a single-corpus lab; would need governance in a
  multi-corpus production deployment.

**Follow-ups**

- Implement Tier 2 LLM merge.
- Author initial Tier 3 override list (drop NER false positives identified
  during Tier 1b inspection — Hogwarts, Quidditch, Privet Drive, etc.).
- Promote `chapters.json` to a tracked intermediate artefact (currently
  recomputed each run from `raw_pages.json`).
- Wrap discovery as a Temporal workflow once Phase 4 lands; each tier
  becomes an activity with durable intermediate outputs.

## Implementation iteration history

The design above is the V6 of an iterative refinement. Worth documenting
because each version was driven by what the data actually did, not by
speculation.

**V1 (regretted) — naive subset matching for all pairs.**
Single-word ↔ multi-word subset clustering caused "Weasley" to subsume
every Weasley (Ron, Fred, George, Percy, Ginny, Arthur) into one
2538-mention mega-cluster. Same failure mode for Malfoy (Lucius + Draco).

**V2 — drop single-to-multi subset, conservative Tier 1a.**
Multi-word subsets only, plus single-word fuzz. Eliminated the family-
name pollution; clusters became precise but lost recall (Harry / Harry
Potter as separate clusters).

**V3 — add Tier 1b coref via fastcoref.**
First version of `resolve_coref_aliases`. Trusted the most-frequent named
person in each coref cluster as the cluster owner; attributed all other
mentions to it. Result: catastrophic over-attribution. Coref's multi-
character clusters caused "He-Who-Must-Not-Be-Named" → Dumbledore,
"Peeves" → Harry, "Mr. Dursley" → Dumbledore.

**V4 — strict Tier 1b: reject multi-PERSON coref clusters.**
If a single coref cluster contains 2+ distinct NER PERSON entities, we
can't trust the attribution; drop the cluster entirely. Also matched
PERSON entities by lowercase text (not just span) to handle NER ↔ coref
boundary mismatches. Cleaner but still let through "Fluffy" → Flitwick
and "Father" → Dumbledore (single-PERSON clusters where coref erred
on the non-named mention).

**V5 — share-token gate.**
An alias is only attributed to a target if they share at least one
significant (non-generic-title) token. "Mr. Filch" / Filch share
'filch' → allowed. "He-Who-Must-Not-Be-Named" / Dumbledore share
nothing → blocked. This deliberately *excludes* semantic aliases, on the
basis that those belong in Tier 2 where the LLM has world knowledge.
Plus stricter alias-candidate filtering (≤4 words, no commas/quotes,
no conjunctions).

**V6 — structure-preserving page join.**
The `extract_chapters` step concatenated page texts with `\n\n`
unconditionally. For sentences spanning page boundaries, this fabricated
a paragraph break mid-sentence, corrupting sentence segmentation, coref
clustering, and chunking. Fix: only insert `\n\n` when the previous
page ends with sentence-final punctuation (`.`, `!`, `?`, optionally
followed by a closing quote); otherwise join with a single space.

**V7 — single-word claim (Tier 1c).**
After running V6 against book 1+2 we noticed the protagonist himself was
split across two clusters: `Harry` (~2400 mentions) and `Harry Potter`
(~88 mentions) sat as separate canonicals because Tier 1a deliberately
doesn't merge single ↔ multi-word and coref happened not to bridge them.
Same for `Hermione` ↔ `Hermione Granger`, every Weasley first-name vs
their full name, etc. Solution: a deterministic post-pass that merges a
single-word cluster into the *unique* multi-word cluster that owns its
significant token. Family names (Weasley, Malfoy) have multiple owners
and stay separate by design. The multi-word form wins as canonical for
reports. This recovered ~30 unifications in book 1+2 without any LLM cost
or non-determinism, leaving Tier 2 to handle only true semantic merges.

Each version's failure mode is regression-tested in
`tests/unit/test_characters.py` so we don't regress to V1-V4 behaviour
later.

## Rollback

Pure code change; no schema or persistent state impact.

- Tier 1a alone: keep `count_mentions` + `cluster_aliases`; remove
  `resolve_coref_aliases` and the coref import. Revert to V2 behaviour.
- Drop coref entirely: remove fastcoref dependency, remove Tier 1b
  function, remove tests. ~30 minutes.
- Drop the entire system: revert to a hand-curated dict in
  `data/processed/character_aliases.json`, replace `extract_characters`'s
  loader with a static dict load. ~1 hour.

The cluster output is regenerated from the corpus on every discovery
run, so there's no migrated data to roll back.
