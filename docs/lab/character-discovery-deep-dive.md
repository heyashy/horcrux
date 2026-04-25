# Character Discovery — Deep Dive

> A review-style walkthrough of the three-tier character extraction system
> built in Phase 2.
> Cross-references: [system design](../horcrux_system_design.md),
> [ADR-0006](../adr/pending/ADR-0006-three-tier-character-discovery.md),
> [`horcrux/characters.py`](../../horcrux/characters.py),
> [`tests/unit/test_characters.py`](../../tests/unit/test_characters.py).

---

## 30-second elevator pitch

> *"For the relational-query side of a RAG pipeline — answering things like
> 'trace Snape's loyalty across the series' — every chunk needs to know
> which characters appear in it so retrieval can filter by character.
> I built a deterministic three-tier discovery pipeline: orthographic
> clustering with NER and fuzzy matching, co-reference resolution to catch
> title-forms in context, and a single-word claim pass to unify personal
> names with their full-name forms. Family names stay correctly
> disambiguated. Semantic aliases like 'He Who Must Not Be Named' →
> Voldemort are deliberately deferred to a fourth LLM-assisted tier where
> world knowledge actually helps. Every failure mode I hit during
> development is regression-tested so the system can't silently slip back."*

---

## 1. What we built

**The problem.** A user asks *"every interaction between Snape and
Dumbledore."* The synthesis layer can read passages, but the *retrieval*
layer needs to filter the corpus down to candidate chunks fast. If we tag
every chunk with `characters: ["Severus Snape", "Albus Dumbledore", ...]`
at ingest time, Qdrant can do that filter as a payload query in a single
round trip.

But characters appear under many names. *Harry Potter* appears as `Harry`,
`Potter`, `Mr. Potter`, `the Boy Who Lived`, `the Chosen One`. *Voldemort*
appears as `Tom Riddle`, `He Who Must Not Be Named`, `the Dark Lord`. A
naive approach — match every name literally — fragments a single character
into many "people" the filter can't unify.

So we needed an **alias dictionary**: for each character, a canonical name
and the set of strings the corpus uses to refer to them. That dictionary
is built once at ingest, used every time a chunk is tagged.

**The output is one JSON file** — `data/processed/aliases_tier1.json` —
shaped:

```json
{
  "Harry Potter": ["Harry Potter", "Harry", "Mr. Potter"],
  "Voldemort":    ["Voldemort", "Lord Voldemort"],
  "Aunt Petunia": ["Aunt Petunia", "Petunia"],
  ...
}
```

That's it — one canonical, a list of aliases. Chunk extraction is just
substring-matching aliases against text and tagging with the canonical.

---

## 2. How we built it — the three deterministic tiers

### Tier 1a — orthographic clustering (NER + fuzzy)

```
input:  list[Chapter]
output: dict[canonical, list[aliases]]
```

**Pipeline:**

1. spaCy `en_core_web_sm` runs NER over each chapter, returns every
   PERSON entity.
2. `Counter` aggregates mentions across the whole corpus. Possessive
   suffixes (`'s`, `'s`) and internal whitespace (handles OCR line-wrap)
   are normalised before counting.
3. `min_count` threshold drops the long tail of NER noise.
4. Union-find clusters survivors by similarity:
   - **Multi-word ↔ multi-word**: subset matching
     (`Harry Potter` ⊂ `Harry James Potter`).
   - **Single-word ↔ single-word**: rapidfuzz character-level ratio ≥ 85
     (catches OCR errors: `Hermione` ≈ `Hcrmione`).
   - **Single-word ↔ multi-word**: deliberately *not* matched — see
     "Why" below.
5. Each cluster's most-frequent variant becomes the canonical.

**Key design choice — single↔multi exclusion.** Naively, you'd say
`Harry` ⊂ `Harry Potter` should cluster. But the same rule applied to
`Weasley` ⊂ `Ron Weasley` *also* triggers, and then transitively pulls in
`Fred Weasley`, `Percy Weasley`, `Ginny Weasley`, all into one mega-cluster
called "Ron". The same rule works for some pairs and breaks for others,
with no purely orthographic way to tell them apart. Tier 1a refuses to
make that call; defers it to Tier 1c.

### Tier 1b — co-reference resolution (fastcoref)

```
input:  Tier 1a clusters + list[Chapter]
output: title-form → resolved canonical attributions
```

**Pipeline:**

1. fastcoref (RoBERTa-based, ~500MB on GPU) runs over each chapter.
   Output: groups of mention-spans that all refer to the same entity in
   context.
2. For each coref cluster:
   - Find named PERSON entities (cross-reference against spaCy's NER, by
     both span and text — boundaries don't always agree).
   - **If more than one distinct named PERSON appears in the cluster,
     drop the whole cluster.** Coref made an error; we can't recover the
     right attribution.
   - Otherwise the unique named PERSON is the cluster's "owner".
     Non-named mentions (title-forms like `Mr. Filch`, `Aunt Petunia`)
     get attributed to the owner.
3. **Share-significant-token gate**: alias and target must share at least
   one non-generic token. `Mr. Filch` (signif: `{filch}`) shares `filch`
   with `Filch` → allowed. `He-Who-Must-Not-Be-Named` shares nothing with
   `Dumbledore` → blocked. This deliberately excludes semantic aliases —
   they're Tier 2's job.
4. Across the corpus, attribution counts aggregate. Title-forms with ≥ 2
   occurrences and clear dominance (top resolution at least 2× the second)
   get promoted to aliases.

**Key design choice — strict precision, accept lost recall.** Coref over
long-form fiction is genuinely noisy. We trade recall (sometimes a real
co-reference doesn't promote to an alias) for precision (we don't poison
the dict with false attributions). The lost recall is cheap to recover at
Tier 2; a false attribution corrupts every chunk thereafter.

### Tier 1c — single-word claim (deterministic merge)

```
input:  clusters dict (after Tier 1a + 1b)
output: clusters dict with single-word folds applied
```

**Pipeline:**

1. Build an index: `significant_token → list[multi-word canonicals containing it]`.
2. For each single-word cluster `S`:
   - Get `S`'s significant tokens (after stripping `Mr.`, `the`, etc.).
   - If `S` has exactly one significant token, look it up in the index.
   - **If exactly one multi-word cluster owns it, merge `S` into that
     cluster.**
   - If multiple multi-word clusters share it (family name like `Weasley`)
     or none do (standalone like `Voldemort`), leave alone.
3. When merging, the multi-word form wins as canonical (better for
   human-readable reports).

**Why this works.** The "exactly one match" rule does the heuristic work
no orthographic algorithm could do alone: it answers *"is this single-word
a personal name or a family name?"* by *counting* how many multi-word
names contain it.

- `Hermione` is in only `Hermione Granger` → personal name → fold.
- `Weasley` is in eight `X Weasley` names → family → don't fold.

Pure structure of the data tells you which is which.

---

## 3. Why this design — answering "why not just X?"

| Alternative | Why we rejected it |
|---|---|
| **Hand-curated alias dictionary** | Doesn't scale beyond one corpus. Maintenance burden grows with every new corpus or every name added. |
| **NER alone (no clustering)** | Same character appears as 5-10 strings; the filter fragments. Ron Weasley searches miss "Ron" mentions. |
| **Full neural co-reference (no orthographic step)** | Coref errors compound. Multi-character coref clusters become wrong attributions. We tested this; it produced "Voldemort epithet → Dumbledore" as a result. |
| **LLM doing everything** | Three real costs. Dollar (full-corpus prompts vs targeted merge — orders of magnitude difference); reproducibility (LLM output drifts across days/models); privacy (entire corpus content leaves the network). |
| **Hybrid with a "is family name" detector** | Considered. The Tier 1c "exactly one owner" rule *is* such a detector — derived from the data structure, no extra ML model needed. |

---

## 4. The iteration story

The architecture above is V7. We didn't design it; we discovered it.
Each version had a specific failure mode that drove the next refinement.

**V1 — naive subset matching.**
Single-word ↔ multi-word subset for *all* pairs. Result: `Weasley` pulled
in every Weasley first name + their full names into one mega-cluster.
2538 mentions, eight distinct people merged.

**V2 — drop single-to-multi subset.**
Conservative Tier 1a. Cleaner clusters, but lost recall — Harry / Harry
Potter / Hermione / Hermione Granger all stayed split.

**V3 — add fastcoref Tier 1b** with naive "trust the most-frequent-named
person in each coref cluster" attribution. Result: catastrophic. Coref's
multi-character clusters caused `He-Who-Must-Not-Be-Named` → Dumbledore,
`Peeves` → Harry, `Fluffy` → Flitwick.

**V4 — reject multi-PERSON coref clusters.**
If a single coref cluster contains two distinct named PERSONs, throw it
out — coref erred and we can't recover. Also matched PERSON entities by
*text* not just span (NER and coref disagree on boundaries). Better, but
still let through `Father` → Dumbledore and the like — single-PERSON
clusters where coref erred on the non-named mention.

**V5 — share-significant-token gate.**
Alias must share at least one non-generic token with target, otherwise
blocked. Deliberately excludes semantic aliases (`Boy Who Lived` / Harry,
`Padfoot` / Sirius) on the basis that those need the LLM with world
knowledge — Tier 2's job. Plus stricter alias-candidate filter (≤ 4
words, no commas/quotes/conjunctions).

**V6 — structure-preserving page join.**
Found while debugging coref. Sentences spanning page boundaries had
`\n\n` jammed mid-sentence, fabricating a paragraph break that corrupted
sentence segmentation, coref clustering, and (later) chunking. Fix: join
with `\n\n` only when previous page ends with sentence-final punctuation;
otherwise single space.

**V7 — single-word claim (Tier 1c).**
Found by inspecting Tier 1a+1b output: `Harry` (~2400 mentions) and
`Harry Potter` (~88 mentions) sat as separate canonicals. Tier 1a
deliberately doesn't bridge them; Tier 1b's coref happened not to.
Solution: deterministic single-word claim — merge if exactly one
multi-word owns the token. Recovers ~30 personal-name unifications
without LLM cost.

**Each version's failure mode is now a regression test.**
We literally cannot ship V1 again — the test that proves "Weasley doesn't
pull in all Weasleys" is permanent code in `tests/unit/test_characters.py`.

---

## 5. Deep-dive questions

### "Walk me through your character extraction system."

Use the elevator pitch. If they want depth, walk through the three
deterministic tiers in order, then mention Tier 2/3 as the layered LLM
and manual safety nets.

### "How did you handle the case of family names like Weasley?"

Tier 1a refuses to subset-cluster single-word into multi-word *for any
name*, because the algorithm can't tell `Harry` (personal) from `Weasley`
(family) just from the strings. Tier 1c then re-introduces the merge with
a precision rule: a single-word folds only if exactly one multi-word name
owns its significant token. Family names have multiple owners, so they're
correctly identified as ambiguous and stay separate. The "exactly one"
gate is the structural insight that distinguishes the two cases.

### "Why three tiers? Why not just an LLM?"

Three real costs.

- **Dollar cost**: ~$0.003 targeted vs ~$1+ full-corpus.
- **Reproducibility**: once an LLM is in the data pipeline, your output
  is no longer bit-identical across runs; that becomes a real ops
  concern.
- **Privacy**: for a literary corpus it doesn't matter, but the pattern
  travels to internal corpora where corpus content leaves your network
  and that's a compliance issue.

The deterministic tiers do most of the work for free; the LLM tier is
reserved for what it's actually best at — semantic aliases (`He Who Must
Not Be Named` → Voldemort) that require world knowledge.

### "What were the failure modes you hit?"

Four big ones, each driving a refinement:

1. **Family name over-clustering** — solved by dropping single↔multi subset.
2. **Coref multi-character cluster errors** — solved by rejecting
   multi-PERSON clusters.
3. **Cross-character semantic mis-attribution** — solved by the share-
   token gate.
4. **Page-join fabricating paragraph breaks mid-sentence** — solved by
   sentence-aware joining.

Each one I caught by inspecting real output, not by writing tests in
advance.

### "How would you measure quality?"

Three signals:

- **Cluster count vs ground truth** — for HP we know roughly how many
  distinct characters there are.
- **Top-N cluster precision** — manually inspect the largest 20
  clusters; how many are clean?
- **Retrieval recall on a held-out query set** — does filtering by
  `characters: ["Snape"]` actually return passages mentioning Snape?
  Measure F1.

For this lab I used (b) — manual inspection of the top clusters at each
iteration. For production scale you'd want (c) with a curated query/answer
set.

### "How would this scale to a much larger corpus?"

The bottleneck is Tier 1b's coref pass — quadratic in chapter length on
the model side, linear in number of chapters. For a million-page corpus,
you'd want to:

1. Batch coref aggressively.
2. Shard by chapter and parallelise across workers.
3. Consider a smaller coref model for the discovery step.

Tier 1a (NER + fuzzy) and Tier 1c (single-word claim) are both linear and
cheap. The whole pipeline naturally fits a Temporal workflow — each tier
as a durable activity with checkpointed intermediate output.

### "What would you do differently?"

Three things:

1. **Promote `chapters.json` to a tracked intermediate artefact** instead
   of recomputing from `raw_pages.json` each run.
2. **Add a measurement harness** — currently I inspect output by eye.
   A held-out query set with expected character matches would let us
   tune `min_count`, `similarity_threshold`, and `dominance_ratio`
   empirically rather than by feel.
3. **Wrap discovery as a Temporal workflow earlier** — hit a
   `transformers` compatibility error mid-iteration and lost the coref
   pass; with Temporal that work would have been preserved.

---

## 6. The "honest takeaways" answer if asked "what did you learn"

> *"That the design always emerged from the data, not the spec. Every
> refinement past V1 was driven by something I saw in the actual output
> that the synthetic tests didn't catch. So the iteration history is more
> important than the final architecture — it shows the system was
> empirical, not aspirational. And it shows the limits: we still defer
> semantic aliases to an LLM tier we haven't written yet, because there's
> no orthographic algorithm that can know 'Padfoot' is Sirius. Knowing
> where to stop pushing the deterministic approach was as important as
> the rules themselves."*

---

## Where the artefacts live

| Concern | File |
|---|---|
| Algorithm | [`horcrux/characters.py`](../../horcrux/characters.py) |
| Tests (each iteration's failure mode regression-locked) | [`tests/unit/test_characters.py`](../../tests/unit/test_characters.py) |
| Discovery script | [`scripts/build_character_aliases.py`](../../scripts/build_character_aliases.py) |
| Cluster output (gitignored) | `data/processed/aliases_tier1.json` |
| Decision rationale + V1→V7 history | [`docs/adr/pending/ADR-0006-three-tier-character-discovery.md`](../adr/pending/ADR-0006-three-tier-character-discovery.md) |
| Daily progression | [`docs/log/2026-04-25.md`](../log/2026-04-25.md) |
