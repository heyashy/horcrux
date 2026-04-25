# Lab Findings

> Cross-cutting engineering findings from running the Horcrux lab.
> Each one is an observation that surfaced from running real code on real
> data, not a design-time prediction. The lessons travel beyond this lab.

Format per finding:

```
Finding N — short title
  Symptom    — what we saw
  Root cause — why it happened
  Fix        — what changed
  Lesson     — what travels
```

ADRs and component deep-dives capture the *what we built*. This doc captures
the *what we learned by running it*.

---

## Finding 1 — Library defaults for batched inference assume "small enough"

**Symptom.** `coref_model.predict(texts=[all_chapters])` on the full corpus
(~200 chapters of ~30KB each) maxed out 16GB RAM and Linux's OOM killer
terminated the process. The same call on book 1+2 (~33 chapters) ran fine.

**Root cause.** fastcoref's `predict` accepts a list of texts and tries to
process them in internal batches as model tensors. Each chapter's tensor
state is held in memory until the call returns. At ~200 chapters, peak
memory exceeds available RAM. The library's default behaviour assumes
the input is "small enough to batch" — which is true at development scale
and false at corpus scale.

**Fix.** Stream chapters one at a time. Call `predict(texts=[chapter.text])`
in a loop. Add `gc.collect()` + `torch.cuda.empty_cache()` every 5
iterations to reclaim leftover tensors. Trade ~5-10% overhead for bounded
peak memory.

**Lesson.** *Memory limits are part of the algorithm's signature.* Same
algorithm, two implementations: batched (fast, memory-explosive at scale)
and streaming (slower, memory-bounded). The right one depends on input
size. ML library defaults often pick batched because it's faster on the
small examples library authors tested with — which silently breaks at
production scale. Failure mode is OOM, not graceful slowdown, which makes
it look catastrophic. Always check whether a library's batched API
streams under the hood; if not, write the streaming wrapper yourself.

---

## Finding 2 — Concurrency tuning is hardware-dependent, not code-dependent

**Symptom.** Temporal OCR workflow with `max_concurrent_activities=8` and
`start_to_close_timeout=5min` per 100-page batch failed catastrophically:
every batch hit timeout, `asyncio.gather` cancelled the rest in cascade,
30+ minutes of runtime, all 36 activities ultimately failed.

**Root cause.** Single-threaded baseline: 0.77s/page. Naive extrapolation:
8 concurrent batches at 100 pages each = 77s wall-clock. Reality on a
laptop: ~5+ minutes per batch under contention. Thermal throttling, memory
bandwidth, Tesseract subprocess startup overhead, and the kernel
scheduler all compound when 8 simultaneous tesseract processes saturate
the system. Linear extrapolation from single-threaded throughput is wrong
on shared-resource hardware.

**Fix.** Reduce `max_concurrent_activities` to 4, increase
`start_to_close_timeout` to 15 minutes (4-5x realistic batch duration),
reduce batch size to 50 pages. Add heartbeating every 30s so the server
knows the worker is alive on slow batches. Result: clean ~22-minute run,
all 73 batches first-attempt success.

**Lesson.** *Operational settings depend on deployment hardware, not code.*
A laptop runs `max_concurrent_activities=4`; a workstation runs `=12`;
production might run `=64`. Same code, same workflow, different config.
ADRs are the right home for "we picked these values because of these
constraints" — not as in-code constants pretending they're universal.
Also: `asyncio.gather` cascade-cancels its other tasks when one fails,
so what looks like "everything broke" is usually one real failure plus
35 cascade-cancellations. Diagnose the *first* genuine failure, not the
visible noise.

---

## Finding 3 — Authored metadata beats derived signal when both exist

**Symptom.** Original Phase 2 plan: regex-scan OCR'd text for `CHAPTER ONE`
markers, then infer book boundaries from chapter-number resets. We
implemented it (V1-V2). Worked, but with edge cases: missed chapters
where Tesseract mangled the heading, false positives in body text where
"chapter" appeared in a sentence, compound number-words ("TWENTY-ONE")
needing a hand-curated thesaurus.

**Root cause.** We were re-deriving structure from OCR'd visual artefacts
of structure that *already existed in the PDF as authored metadata*. The
PDF's `get_toc()` returns `[level, title, page]` entries: book starts,
chapter starts, page numbers — exact, no derivation needed.

**Fix.** Replace regex-on-OCR with `pymupdf.Document.get_toc()` extraction.
Result: 15/15 chapters of book 1 match the source TOC with **zero drift**.
Captured as ADR-0005.

**Lesson.** *Prefer authored metadata over derived signal.* When a system
provides both, the derived path is reinventing the answer the author
already gave. The orthographic regex approach is the right *fallback* for
corpora without bookmarks (we documented it as such), but it should never
be the primary path when the metadata is available. This generalises:
prefer EXIF over filename parsing, prefer DB schema over column-name
heuristics, prefer OpenAPI specs over response-shape inference.

---

## Finding 4 — "Exactly one match" disambiguates personal from family names

**Symptom.** Subset clustering caused `Weasley` to subsume every Weasley
first name (Ron, Fred, George, Percy, Ginny, Arthur) into one
2538-mention mega-cluster. Same failure for Malfoy (Lucius + Draco
merged). The natural rule (`Harry` ⊂ `Harry Potter` should cluster) and
the failure mode (`Weasley` ⊂ `Ron Weasley` should NOT cluster) can't be
distinguished by orthography alone.

**Root cause.** A single-word name is ambiguous: it might be a personal
name (only one full-name expansion exists) or a family name (multiple
distinct full names share it). Pure orthographic algorithms can't tell
them apart from the strings.

**Fix.** Tier 1c — count multi-word clusters that contain the single-word
as a significant token. If exactly one matches, it's a personal name →
fold into that cluster. If multiple match, it's a family name → leave
alone. The structure of the cluster set tells you which is which.

**Lesson.** *When two cases look identical at the level you're operating
at, look one level up.* The orthographic level can't distinguish personal
from family. The cluster-population level can — count owners. The same
move applies broadly: when two failures look identical at the line-of-code
level, look at the call graph. When two metrics look identical at the
service level, look at the request graph. The level you can't disambiguate
is rarely the level the answer lives at.

---

## Finding 5 — Layout artefacts corrupt downstream semantic processing

**Symptom.** Coref clustering produced bizarre results — sentence boundaries
not respected, character mentions splitting across what should have been
single sentences. Tracked down to chapter text where a paragraph break
had been inserted *mid-sentence*.

**Root cause.** `extract_chapters` joined page texts with `"\n\n"`
unconditionally. PDF page boundaries are *layout artefacts* — sentences
routinely span page boundaries. Joining with `"\n\n"` (a paragraph break
in plain text convention) fabricates a structural signal that wasn't
authored, and downstream consumers (sentence segmenters, coref models,
chunkers) treat it as semantically meaningful.

**Fix.** Structure-aware page join: insert `\n\n` only when the previous
page ends with sentence-final punctuation (`.`, `!`, `?`, optional
closing quote). Otherwise concatenate with a single space. Imperfect (we
don't recover *real* paragraph breaks within a page) but eliminates the
worst case of falsifying structure.

**Lesson.** *Layout is not structure.* Page boundaries, line wraps,
column breaks, font changes — all cosmetic in the source format,
semantically meaningless in plain text. Whenever you concatenate
fragments back together, ask which boundaries are *authored* and which
are *layout*. Treating layout as structure poisons downstream NLP. Same
shape: don't trust file modification times as creation time, don't trust
SQL row order without `ORDER BY`, don't trust visual alignment as
syntactic indentation.

---

## Finding 6 — Coref multi-entity clusters are unrecoverable, not fixable

**Symptom.** Tier 1b coref attribution went catastrophically wrong on
some clusters: `He-Who-Must-Not-Be-Named` → Dumbledore, `Mr. Dursley`
→ Dumbledore, `Peeves` → Harry, `Fluffy` → Flitwick. Distinct named
characters got merged into wildly unrelated clusters.

**Root cause.** Coref over long fiction has high error rates. A single
coref cluster sometimes contains *two distinct named PERSON entities*
plus various pronouns and partial references. The naive algorithm
(attribute everything to "the most-frequent named person in the
cluster") trusted these noisy clusters and produced wrong canonical
assignments — and wrong assignments are worse than no assignments,
because they corrupt every chunk's character payload thereafter.

**Fix.** Two layers. (1) **Reject** any coref cluster containing more
than one distinct named PERSON — coref erred and we can't recover. (2)
**Share-significant-token gate**: alias and target must share at least
one non-generic token. `Mr. Filch` (signif: `{filch}`) shares `filch`
with `Filch` → allowed. `He-Who-Must-Not-Be-Named` shares nothing with
`Dumbledore` → blocked. Semantic aliases (no shared tokens) get
deferred to Tier 2 LLM where world knowledge actually helps.

**Lesson.** *Trust your noisy upstream proportionally to its noise.*
Coref is a probabilistic model with imperfect output. Rather than
papering over its errors with heuristic post-processing, identify
when the model has *confessed uncertainty* (multi-PERSON clusters,
no-shared-token attributions) and decline to use those outputs at all.
Defer them to a layer that can actually handle them — in this case,
an LLM with world knowledge. The error budget for orthographic clustering
is dramatically lower than for semantic clustering, and the right
architecture honours that asymmetry.

---

## Finding 7 — NER models have domain-specific blind spots

**Symptom.** spaCy `en_core_web_sm` PERSON entities on the HP corpus
included a long tail of non-people: places (`Hogwarts`, `Privet Drive`,
`Knockturn Alley`, `Burrow`, `Smeltings`, `Gringotts`), magical concepts
(`Quidditch`, `Floo`, `Cloak`, `Howler`, `Mandrake`, `Bludger`, `Quaffle`),
animals (`Fang`, `Errol`, `Norris`, `Norbert`, `Aragog`, `Dobby`, `Hedwig`),
dialect particles (`yeh`, `Blimey`, `Nah`), and even prose fragments
(`Ron muttered`, `Fred, George`, `'s Malfoy`). All survived the min_count
threshold. All polluted the alias dictionary as standalone "characters."

**Root cause.** spaCy's small English NER model is trained on news,
Wikipedia, and web text — corpora where capitalised single-token
sequences usually *are* people. Literary fiction breaks that
distribution. A magical place name (`Hogwarts`) has the same surface
features as a person's name; a dialect word (`yeh`) starts a sentence
in capitals; a prose fragment (`Ron muttered`) looks structurally like
"FirstName Surname". The model has no way to know what's a "person"
versus what's a "place / object / verb / dialect" in this corpus.

**Fix.** Two layers. (1) Statistical: `min_count` threshold drops the
rarest false positives. (2) Manual: Tier 3 override JSON with a `drop`
list for the survivors that the lab's user identifies by inspection
(planned). For very high precision, a domain-tuned NER model or even
a hand-curated allowlist would beat both — at the cost of generality.

**Lesson.** *Probabilistic upstream models have systematic blind spots
in domains they weren't trained on.* The blind spots aren't random
errors — they're correlated with the domain shift, and they survive
naive thresholds. Plan for them: (a) inspect a sample of NER output
on real corpus before trusting it; (b) maintain a manual override layer
so humans can correct what the model can't see; (c) document the
override list as part of the data, not as a code constant. The same
pattern travels to any "ML output → downstream pipeline" architecture:
LLM hallucinations, OCR systematic errors, classifier confidence
mis-calibration. The defensive layer isn't optional.

---

## Finding 8 — Probabilistic models disagree at entity boundaries

**Symptom.** Tier 1b coref attribution was failing to recognise that
`"Mr. Filch"` in a coref cluster was the same entity as the spaCy NER
PERSON span tagged elsewhere in the same chapter. Both agreed on the
identity; they disagreed on whether the span included the trailing
punctuation, the leading honorific's period, or a single trailing space.
Naive `(start_char, end_char)` equality check missed the match. Same
identity, different bytes.

Separately: spaCy NER captured chapter-heading names spanning a line
break — `"Professor\nMcGonagall"` came back as a single PERSON entity
with an embedded newline character. Equality check against
`"Professor McGonagall"` (single space) failed.

**Root cause.** Two probabilistic models making the same boundary
decision on the same span will *usually* agree but not *always*.
Tokenizers differ: one might include the period in `Mr.`, the other
might treat it as a sentence-end marker. Whitespace handling differs:
one might consume newlines, the other might preserve them. Layout
artefacts (line wraps in source) leak into surface forms.

**Fix.** (1) Match by *both* span tuple and lowercased text — text
match catches boundary disagreements. (2) Normalise mention text
aggressively at the model boundary: collapse internal whitespace
(handles newlines), strip possessive suffixes, lowercase for comparison.
Centralise this in a `_normalise_mention()` function that every
consumer goes through.

**Lesson.** *When integrating two probabilistic systems' outputs,
identity is fuzzier than equality.* Span tuples are a serialisation
artefact, not the entity. A single `_normalise_mention` boundary that
every output passes through prevents bug-per-consumer. This generalises:
when two systems return "the same" entity, ask which level you should
compare on (bytes? normalised string? canonical id?). Bytes-level is
brittle whenever either system has any flexibility. Layout artefacts
(newlines, double spaces, smart quotes) creep in everywhere; normalise
once at the boundary, not per-consumer.

---

## Finding 9 — Transitive union-find merges siblings through shared full-name ancestor

**Symptom.** Full-corpus discovery produced this cluster:

```
"Harry Potter": ["Harry Potter", "James Potter", "Harry James Potter"]
```

`James Potter` (Harry's father, distinct character) merged into Harry's
cluster. Same risk shape exists for any two characters whose two-word
names are both subsets of a three-word form that appears in the corpus.

**Root cause.** Tier 1a clustering uses union-find over multi-word subset
relationships. `Harry Potter` ⊂ `Harry James Potter` → cluster A.
`James Potter` ⊂ `Harry James Potter` → cluster B. Because
`Harry James Potter` is a member of both clusters, union-find merges A
and B. The transitive merge is correct given the rule, but the rule
treats "shares a parent in the subset graph" as "is the same entity,"
which it isn't when the parent is a full-name expansion that happens
to contain both shorter forms as sub-tokens.

**Fix (proposed, not yet shipped).** Two options:

- **Restrict multi-word subset to two-token-into-two-token only.** Drops
  the rare `Harry Potter` ↔ `Harry James Potter` merge, but Tier 1c's
  single-word claim already handles `Harry` ↔ `Harry Potter`, so most
  expansions are recovered another way.
- **Detect "ancestor merges" specifically and reject.** When clustering
  A and B through a shared parent C, check if A and B share any tokens
  beyond C's tokens; if not, refuse the transitive merge.

The first is simpler and lower risk; the second is more precise but
adds graph-walking complexity to a previously linear algorithm.

**Lesson.** *Transitive closures aren't always the same as semantic
equivalence.* Union-find is the right data structure for "merge classes
under a relation," but the relation has to actually be an equivalence
relation. Subset-via-shared-parent isn't transitive *in meaning* even
though it is transitive in the data structure. Whenever you reach for
union-find, ask whether the relation you're feeding it is transitive in
the domain you care about, not just transitive mathematically.

---

## Finding 10 — Per-chapter context windows leak across distinct same-surname entities

**Symptom.** Full-corpus discovery merged `Albus Dumbledore` into the
`Ariana Dumbledore` cluster:

```
"Ariana Dumbledore": [
    "Ariana Dumbledore", "Albus Dumbledore",
    "Dumbledore", "Professor Dumbledore", "Dumbledore himself", ...
]
```

These are *different people* (siblings). Both are NER PERSON entities
across the corpus. Yet "Albus Dumbledore" passed the share-token gate
(shares `dumbledore` with `Ariana Dumbledore`) and was attributed as
an alias.

**Root cause.** Tier 1b's `resolve_coref_aliases` builds
`person_text_set` **per chapter**, then checks `if mention.lower() in
person_text_set: continue` to prevent NER PERSON entities from being
attributed as aliases of other named entities. The check is correct in
*scope* (per-chapter NER output) but wrong in *intent* — the goal is
"don't ever attribute one named character to another named character,"
which requires a *corpus-wide* person set. In chapters where Ariana is
NER-tagged but Albus is not (e.g. a Deathly Hallows scene focused on
Ariana's story), Albus's mentions in that chapter pass the gate and
get attributed.

**Fix (proposed, not yet shipped).** Build `person_text_set` once at the
start of `resolve_coref_aliases` from *all* chapters' NER output, then
use the corpus-wide set for the gate check throughout the per-chapter
attribution loop. One pass adds O(N_chapters) work; the prevention is
absolute.

**Lesson.** *Be explicit about the scope of "known" sets used as gates.*
Local context (per-document, per-chapter, per-batch) is the right scope
for *positive* identification — what entities exist *here*. It's the
*wrong* scope for *negative* gates — what entities exist *anywhere we'd
care about*. The two questions look syntactically similar (`if x in
set: continue`) but have different correctness criteria. Whenever a
filter uses a "known" set, ask: *is the set's scope the same as the
domain the filter must protect?*

This is structurally the same as cross-tenant data leaks in multi-tenant
systems, just at a smaller scale: the wrong scope gate lets data
"leak" between domains that should have been isolated. The ML output
case here is benign (an alias dictionary error); the same pattern in a
permission-check would be a security incident.

---

## Finding 11 — LLM in the ingest pipeline destroys reproducibility

**Symptom.** Tier 2 was planned as an LLM-assisted semantic merge: send
the Tier 1 cluster set to Haiku, ask "which clusters refer to the same
character?", merge based on the response. Cost was small (~$0.003 per
discovery run). But once we held this up against the rest of the pipeline,
the architectural cost became visible.

**Root cause.** Each ingest run that includes an LLM call produces a
slightly different alias dictionary. Even with `temperature=0`, models
drift across versions, sampling has residual variance, and the prompt's
exact tokenisation can shift output. **Two ingest runs of the same
corpus stop producing identical alias dictionaries.** Every chunk's
character payload now depends on which ingest run produced it. Filter
results differ across runs. The whole retrieval surface area drifts.

For *queries* this is fine — stochasticity is bounded per-query, the
user gets one report, can re-run if they want. The blast radius is one
report.

For *ingest* this is catastrophic — stochasticity propagates through
the alias dictionary into thousands of chunk payloads, then into every
filter result, then into every retrieved candidate set, then into every
synthesis. The blast radius is the entire retrieval surface.

**Fix.** Drop Tier 2 from the architecture. Tier 1a + 1b + 1c stays
deterministic. Tier 3 (manual hand-curated overrides) handles the cases
Tier 1 deliberately leaves open — semantic aliases like
`Padfoot → Sirius`, `Moony → Lupin`, `Snuffles → Sirius`,
`Voldemort → Tom Riddle → He Who Must Not Be Named`. The override JSON
is version-controlled, reviewable, and produces identical results across
runs by definition.

**Lesson.** *Stochasticity at ingest is qualitatively worse than
stochasticity at query.* Same model, same probability of drift, but the
ingest case amortises the drift across every downstream consumer. If
the same operation can be done at either layer, do it at the layer where
non-determinism is bounded — almost always the query layer. ML
literature talks about "training time vs inference time" trade-offs;
ETL has the same shape, "ingest time vs query time," and the principle
is the same: keep the stable thing stable; let the variable thing vary
within a small blast radius.

This generalises beyond LLMs. Any non-deterministic step in an ingest
pipeline (random initialisation, network-dependent enrichment,
external-API lookups) carries the same risk. The deterministic-by-default
posture is what gives the system audit-friendly reproducibility.

---

## Finding 12 — NER mentions include modifier+name noise

**Symptom.** Tier 1b coref attribution accepted alias candidates like
`"Voldemort himself"`, `"Dear Harry"`, `"only Crookshanks"`,
`"poor old Ripper"`, `"Harry blankly"`. These survived all earlier filters
(not pronouns, not articles, ≤4 words, no commas, no conjunctions) and
became aliases of their respective characters in the dict. None of them
are real character names — they're prose fragments where NER captured
both a modifier and a name as a single PERSON entity.

**Root cause.** spaCy NER's default model treats anything capitalised
near a name token as part of the entity. "Dear Harry," "Mr Filch" — both
two-word capitalised sequences. The model has no reliable way to
distinguish "honorific + name" (real alias) from "affection + name"
(noise) from "adverb + name" (NER error). All three look the same at
the surface level.

**Fix.** Extend the alias-candidate filter with a `_STOP_MODIFIERS` set:
reflexive pronouns (himself, herself, themselves), affection forms
(dear, darling, sweet), restrictive modifiers (only, even, just, alone),
quality adjectives (poor, old, young), and the specific adverbs that
appeared in our coref attribution noise (blankly, grumpily, hastily, etc).
A candidate containing any of these tokens is rejected entirely.

We don't lose retrieval power: chunk-time substring matching with word
boundaries (`\bHarry\b`) still finds the underlying name *inside* phrases
like "Dear Harry". We just don't promote the noisy phrase to a separate
alias key.

**Lesson.** *NER mentions are noisy in literary fiction in characteristic
patterns.* The patterns are knowable from inspection: reflexive pronouns
trail names ("Voldemort himself"), affection words precede them ("Dear
Tom"), quality adjectives wrap them ("poor old Ripper"). Fiction-specific
noise filters belong with the NER consumer (us), not with the NER
provider (spaCy). The general lesson: when integrating a domain-mismatched
ML model, the integration layer is the right home for domain-specific
post-processing, not the model itself. We can't fix spaCy's training
distribution; we can list the noise patterns we observe in our domain
and reject them. This is the structural complement to Finding 7's
"manual override layer" — both are layers that exist to compensate for
upstream domain mismatch.

---

## Finding 13 — Overfitting at the rule level: when special cases share a feature shape with non-special cases

**Symptom.** When debugging Finding 10 (Albus Dumbledore being merged into
Ariana Dumbledore's cluster via the share-token gate), an obvious-looking
fix presented itself: add a "first-name conflict" rule. If alias and
target share a surname but their unique non-shared tokens fuzz-mismatch,
they're distinct family members → block the merge.

The rule worked for the offending case: `albus` ↔ `ariana` are
orthographically very different → block.

Then we tested it on `Mad-Eye Moody` ↔ `Alastor Moody`. Same shape:
shared `moody`, unique tokens `mad-eye` vs `alastor`, also
orthographically very different. The rule would block this merge — but
these *are* the same person (nickname + given name).

Same orthographic feature shape:
  • shared surname + mutually-distant unique tokens
Different correct outcomes:
  • Albus / Ariana Dumbledore → block (different people)
  • Mad-Eye / Alastor Moody    → allow (same person, nickname)

**Root cause.** The proposed rule overfit to the example we saw. It
treated an *orthographic* feature ("unique tokens are distant") as a
proxy for a *semantic* property ("these refer to different people"),
but the proxy correlates only weakly with the property. Real-world
naming conventions include nicknames, code names, formal vs informal,
diminutives — all of which produce "shared surname + distant unique
parts" while still being the same person. The orthographic feature
*overlaps* with the discriminator but isn't *equivalent* to it.

**Fix (the rule we didn't ship).** Don't add the rule. Instead:
  1. Use the *generic mechanism* — corpus-wide person_text_set
     (Finding 10's actual fix) — which blocks all NER-tagged PERSON
     entities from being attributed as aliases. This handles the
     Albus/Ariana case structurally (Albus is a NER PERSON, so it
     can never be attributed) without making semantic claims.
  2. Use Tier 3 manual override for residual cases, including any
     where the generic mechanism mis-handles a nickname like
     Mad-Eye → Alastor.

The decision rule splits along the right axis: **the deterministic
layer uses generic mechanisms, the override layer handles known
specifics.** No middle layer of structural-but-overfit rules.

**Lesson.** *When a proposed rule fits one case, sanity-check it against
two or three other cases that share its surface features. If correct
outputs differ across them, the rule is using the wrong feature.*
Orthographic distance correlates with "different name" only when the
ground truth has no nicknames. As soon as nicknames exist, the
correlation breaks; you need a different feature (or no rule at all).

This is the same trap as feature engineering in ML: a feature that
"works" on the training set isn't useful if its causal relationship to
the label is incidental rather than structural. The defensive posture
is the same here: prefer generic mechanisms (catch-all rules with low
false-negative rate) plus a manual override layer for the residuals,
over clever structural rules that look principled but encode hidden
assumptions about the data.

---

## Finding 14 — Multi-stage pipelines need rule consistency across stages

**Symptom.** After shipping Finding 9's anchor-count rule (multi-word
forms with 3+ significant tokens and low count don't bridge subset
merges in Tier 1a), the `Harry Potter` cluster *lost* its proper alias
set. `Harry` no longer folded into `Harry Potter` via Tier 1c's
single-word claim. Looking at the dict you'd think Fix 9 had broken
character discovery wholesale.

**Root cause.** Fix 9 was applied at Tier 1a's clustering loop:
multi-word names with 3+ significant tokens don't anchor subset merges.
But Tier 1c (single-word claim) builds its *own* index of "multi-word
owners" — and indexed every multi-word cluster regardless of significant-
token count. So `Harry James Potter` survived as a cluster (count ≥
min_count), got indexed as an owner of token `harry`, and competed with
`Harry Potter` as a candidate owner. Tier 1c saw two owners, declared
the case ambiguous, and refused to fold `Harry` into either. The
single-word claim quietly stopped working for any name where a 3+-token
expansion exists in the corpus.

**Fix.** Apply the same anchor logic at Tier 1c. When building the
owner index, exclude multi-word canonicals with 3+ significant tokens.
Both stages now agree that "full-name expansions are not personal-name
anchors."

**Lesson.** *Concepts that span multiple stages of a pipeline need to
be applied consistently at every stage.* The anchor-count concept is
"a 3+-significant-token form is too rare/specific to anchor shorter
forms" — it's a property of the data, not a property of one
clustering loop. Applying it at Tier 1a alone gave one stage a
correctness rule that the next stage didn't have, and the inconsistency
manifested as silent regression of the user-visible output.

The same shape appeared a second time in the same session: the
stop-modifier filter (`_is_alias_candidate`) ran at Tier 1b coref
attribution but not at the NER count step. Prose-fragment NER false
positives like `"Harry blankly"`, `"Harry muttered"` survived as
their own Tier 1a clusters (count ≥ 3, no stop-modifier filter at this
stage). They then polluted Tier 1c's owner index — `Harry` saw eight
multi-word `Harry X` competitors and refused to fold into any of them.
Same architectural lesson: filters must be applied at the earliest
stage they're correct at. The stop-modifier filter is a "this isn't a
useful mention" filter; correct anywhere mentions enter the pipeline.
Applying it only at the late attribution stage left the early
clustering stage unprotected.

This is a subtype of the broader pattern: when a refactor changes the
*invariant* a stage depends on, every downstream stage that consumed
the old invariant has to be updated too. The "shape" of the data
changed (full-name forms are now first-class clusters, not bridges in
disguise), and Tier 1c's index-building was relying on the old shape
without realising it. The right defensive posture: when you tighten a
rule, grep the codebase for every consumer of the invariant the rule
governs, and verify each one explicitly. Cross-stage concept
consistency is the architecture issue underneath the symptom.

---

## Finding 15 — Identifier and surface form are independent decisions

**Symptom.** The original alias dictionary used canonical character names
(`"Harry Potter"`, `"Albus Dumbledore"`) as both the dict key *and* the
display label. Discussion surfaced four operations that touch character
strings — storage, comparison, display, identity — and we were
implicitly conflating "identity" with "display."

**Root cause.** Surface forms are not stable identifiers. Three real
production failure modes the surface-as-identifier pattern hits:

  1. **Multi-language**: `"Harry Potter"` is `"Гарри Поттер"` in Russian.
     The same character has different canonical forms in different
     languages; surface-as-identifier breaks at internationalisation.
  2. **Identity stability**: a label correction (typo, rename) requires
     rewriting every chunk's character payload. Opaque IDs decouple the
     two.
  3. **Disambiguation collisions**: two entities sharing a label
     (`"Tom Riddle"` and `"Tom the bartender"` in our corpus). Surface-
     as-identifier requires tie-breaking gymnastics; opaque IDs
     handle it natively.

**Fix.** Adopt the industry-standard separation:

  - **Storage**: original case preserved, in the `label` field.
  - **Comparison**: lowercase at the boundary (already in place via
    `extract_characters` and `_is_alias_pair`).
  - **Display**: read the `label` field for human-facing output.
  - **Identifier**: opaque slug derived from the canonical name,
    deterministic and stable across re-runs.

Concretely the JSON shape changed:

```json
// Before — surface as identifier
{
  "Harry Potter": ["Harry Potter", "Harry", "Arry", "Barry"]
}

// After — ID indexed
{
  "harry_potter": {
    "label": "Harry Potter",
    "aliases": ["Harry Potter", "Harry", "Arry", "Barry"]
  }
}
```

`extract_characters` now returns slug IDs (`["harry_potter", ...]`),
not display labels. Callers do ID→label lookup via `lookup_label` for
display. Tier 3 manual overrides will be slug-keyed (cleaner than
typing case-sensitive labels).

**Lesson.** *Canonicalisation is four independent decisions, not one.*
Storage case, comparison case, display case, and identifier scheme are
all separately choosable. Production systems collapse them on purpose
where the trade-offs say so:

  - Search engines (Elasticsearch, Solr): lowercase match, preserve
    case in `_source`, document-ID for identity.
  - Entity-linking systems (Wikidata, BLINK, GENRE): opaque QIDs as
    identity, multi-language label tables, lowercase normalisation
    for matching.
  - Postgres `CITEXT`: case-preserving, case-insensitive comparison.

The single most common mistake — and the one we made initially — is
**implicitly using surface form as identifier** because it's
syntactically convenient. It works until it doesn't, and the failure
modes (i18n, renames, collisions) only surface at scale. Doing the
separation up-front costs little (a slug helper, a wrapper struct);
adding it later requires touching every consumer of the old shape.

For this lab specifically: we get the benefit immediately by being
able to write `"force_merge": [["padfoot", "sirius_black"]]` in Tier 3
overrides without worrying about case or whitespace, and indirectly
when Tier 3 wants to fix a typo'd label without ripping through
the chunk payloads.

---

## Finding 16 — Synthetic test fixtures must match the real distribution

**Symptom.** A unit test for `chunk_chapter_text` used synthetic text
of the form `"Word " * 30 + "this is sentence A."` repeated four times,
expecting spaCy's sentencizer to find four sentences. It found one. The
test fixture was too unrealistic for the rule-based sentencizer to
parse correctly.

While debugging, I caught a separate real bug: `current[-0:]` in Python
returns the whole list, not an empty slice — so `overlap_sentences=0`
was *appending* the next sentence to the existing chunk instead of
resetting. The unit test wouldn't have caught this with the broken
fixture either, since it never reached the cut path.

**Root cause.**
1. spaCy's `sentencizer` pipe is rule-based, tuned for prose patterns
   (subject-verb-object structure, varying token shapes, real
   punctuation). A repeating-token pattern doesn't trigger its
   boundary rules even when the punctuation is correct.
2. `lst[-0:]` is a known Python negative-indexing edge case: `0` and
   `-0` are the same integer, so the slice becomes `lst[0:]` — the
   whole list — not an empty slice as the negative-index intuition
   suggests.

**Fix.**
1. Test fixture changed to prose-shaped sentences with subject-verb-
   object structure. spaCy now finds the four sentences correctly.
2. Production code: explicit guard:
   ```python
   overlap = current[-overlap_sentences:] if overlap_sentences > 0 else []
   ```

**Lesson.** *Synthetic fixtures must match the real distribution your
code will process.* If your code uses an ML model trained on prose,
your tests need prose-shaped inputs. The temptation to use
`"a a a a"` patterns for clarity is real, and the failure mode is
silent — the test reports passing or failing for reasons unrelated
to the logic you're trying to test.

The Python `-0` quirk is its own micro-lesson: any time you slice with
a parameter that could be zero, **explicitly handle the zero case** —
don't rely on the negative-indexing intuition. This bug would have
shipped silently if the test fixture had worked properly.

---

## Finding 17 — Embedding models silently truncate; "chapter-level" is really "chapter-opening-level"

**Symptom.** The two-collection design (`hp_chapters` for breadth,
`hp_paragraphs` for precision) embeds whole chapters as single points
into `hp_chapters`. Chapter texts are 3,000–5,000 tokens. The embedder
is `bge-large-en-v1.5` with a 512-token max sequence length. Sentence-
transformers tokenises, **truncates to 512 tokens**, and embeds — with
no warning. The "chapter-level" embedding is therefore a
"first-paragraph-of-chapter" embedding, not a whole-chapter
representation.

**Root cause.** Dense embedding models have fixed context windows;
sentence-transformers' default behaviour for over-length input is silent
truncation, not an error or even a log line. The mismatch between
"point in `hp_chapters`" and "vector representing the chapter" is
invisible at the storage layer — Qdrant stores 1024-dim vectors
identically whether they came from 512 tokens or 50,000.

**Fix.** None applied at this stage — the lab accepts the limitation.
Two options if it bites later:
1. Mean-pool windowed embeddings: split the chapter into 512-token
   windows, embed each, average. Standard fix for long-document
   embedding; non-trivial because pooling strategy (mean vs. attention-
   weighted vs. CLS) affects retrieval quality.
2. Use a long-context embedder (`bge-m3`, `text-embedding-3-large`,
   `nomic-embed-text-v1.5`) — most cap at 8k tokens, enough for
   most chapters without windowing.

For the lab, the truncation bias is *probably acceptable*: HP chapter
openings reliably establish setting and characters, so the first 512
tokens carry signal that's representative of the chapter's broader
topic. But this is an assumption, not a measurement. If chapter-level
retrieval ever lands wrong-but-confidently in evaluation, this is the
first place to look.

**Lesson.** *Storage-layer types lie about semantic content.* A field
called `hp_chapters` storing "chapter-level vectors" hides whether the
vector actually represents the chapter or just its opening. The model's
context window is a load-bearing constraint that doesn't appear
anywhere in the schema, the collection name, or the upsert call. Three
layers of abstraction — embedder API, vector store, application code —
each pass the truncation through without flagging it.

The defensive pattern: any time you embed text whose length isn't
bounded by your code, **assert it fits the model's context window
explicitly** (or accept the truncation and document it where future-you
will re-encounter the question). Naming a thing for what you wish it
were doesn't make it that.

---

## Finding 18 — Qdrant silently brute-forces below the indexing threshold

**Symptom.** After upserting 5,575 vectors across `hp_chapters` (198)
and `hp_paragraphs` (5,377), both collections report
`indexed_vectors_count: 0` despite a fully-configured `hnsw_config`
(`m=16`, `ef_construct=100`). HNSW exists in the config but is not
actually built; queries fall back to brute-force cosine over every
point in the collection.

**Root cause.** Qdrant's `optimizer_config.indexing_threshold` defaults
to **20,000 vectors per segment**. Below that, building an HNSW graph
costs more in indexing-time and memory than it saves at query-time, so
the optimizer skips it. With 5k points in 8 segments (~700/segment),
no segment ever crosses the threshold, and the HNSW config is purely
declarative — it describes what *would* be built if the collection
grew. A naive reading of "we're using HNSW" is wrong: at this scale,
we're using exhaustive search.

**Fix.** None applied — at this corpus size, brute-force is genuinely
faster than HNSW (HNSW is approximate, with overhead that beats the
savings only past ~10k points). Two options if explicit HNSW behaviour
matters (e.g., portfolio-grade demonstration):
1. Lower `optimizer_config.indexing_threshold` to ~1,000 in
   `ensure_collection` — forces HNSW to build even for small
   collections, useful for showing the index path actually runs.
2. Live with brute force, document the threshold explicitly in the
   code, and let scale dictate when HNSW kicks in.

For the lab, option 2 is correct: brute-force is faster *and* more
accurate at this size. But the gap between "config says HNSW" and
"server actually does HNSW" is a real failure mode for anyone
benchmarking ANN performance on a small dataset and concluding their
HNSW tuning matters when it doesn't.

**Lesson.** *Configuration declares intent; runtime decides
behaviour.* The HNSW config is real, just not active yet. Three
defaults conspire to obscure this:
- HNSW config in the API surface (suggests HNSW is on).
- `indexed_vectors_count` field returning `0` (the only honest signal).
- No log line, no warning, no doc-popup explaining the threshold.

The defensive pattern is the same as Finding 17: **assert the system is
in the state you think it's in, don't assume.** For Qdrant, that means
inspecting `indexed_vectors_count` after upsert and either accepting
brute-force or lowering the threshold deliberately. For any vector
store, "I configured an index" doesn't imply "the index is built";
verify with a dedicated check.

---

## Finding 19 — A feared failure mode that hybrid design masked

**Symptom (predicted, not observed).** When chapter-level chunks were
introduced (one point per chapter in `hp_chapters`), bge-large's
512-token context window meant only the chapter opening was actually
embedded (Finding 17). The worry: when the synthesis agent receives a
chapter chunk in its candidate set, it might over-cite — treating the
chunk as though it represented the full chapter content, including
parts of the chapter the chunk's vector never saw.

End-to-end testing on three queries (Cedric Diggory's killer; whether
Harry was in love with Ron; what Crouch Jr. confessed under
veritaserum) showed the leak **did not materialise**. Synthesis
output stayed grounded in the cited passages.

**Root cause of the non-failure.** The hybrid retrieval design pulls
from **both** collections in parallel — chapters AND paragraphs. For
queries whose answers live deep in a chapter (Crouch Jr.'s
confession, mid-Veritaserum-chapter), the paragraph-level chunks
covering those passages get retrieved alongside the chapter-level
chunk. Paragraph chunks contain the actual evidence content; chapter
chunks contribute breadth via the RRF score bonus when they overlap
on topic.

The mechanism that "should have" leaked — chapter-only citation —
doesn't get a chance to fire because there's almost always a
paragraph chunk in the candidate set carrying the real content.

**Fix.** None needed. The architecture defended itself.

**Lesson.** *Defence in depth makes individual layers robust to
their own known weaknesses.* F17 is a real limitation of the chapter
collection in isolation — but the chapter collection isn't used in
isolation. The paragraph collection compensates. When you design
multi-layered systems, individual-layer brittleness becomes acceptable
if the upper layers route around it.

The harder question this finding raises: how do you genuinely
demonstrate the leak? A query whose answer is *only* in a
chapter-not-paragraph chunk (i.e. only in tokens 1-512 of a chapter,
nowhere else) is artificial — paragraph chunks span the same text. To
test the hypothesis cleanly you'd need to either remove paragraph
chunks from the candidate set, or construct a synthetic corpus.
Neither is the right thing to do for the lab.

This is also worth recording as evidence that **the lab's
"surface failure modes" framing applies symmetrically — sometimes
end-to-end verification surfaces the *absence* of an expected
failure, which is itself informative.** Hybrid retrieval was added
in Phase 4 for performance + recall reasons; this finding documents a
secondary benefit that wasn't the reason for the design but is a
real consequence of it.

---

## Finding 20 — Conviction calibration anchors high without literal-statement anchoring

**Symptom.** Across multiple end-to-end queries, the synthesis agent
returned `conviction=5/5` even when the passages supported the answer
only indirectly:

- *"Was Harry in love with Ron?"* — passages prove "Harry loved
  Ginny" (the kissing scene, the "Ginny or Ron?" inner monologue).
  They do not prove "Harry didn't love Ron" — that's an absence of
  evidence, not evidence of absence. Agent returned 5/5.
- *"Who killed Cedric Diggory?"* — Dumbledore's eulogy ("Cedric was
  murdered by Lord Voldemort") supports the headline answer at 5/5,
  but the agent's full response included Crouch-Jr-Imperiused-Krum
  details where the citation chain is more indirect. Agent returned
  5/5 on the whole answer, including the indirect parts.

Critically, when the passages contained **no** evidence for the
question (the conjunctivitis-spell query, see Finding 21), the agent
correctly returned `conviction=1/5`. So the rubric isn't broken at
the extremes — it just doesn't bite at the boundary between "literal
statement" (5) and "supported with caveats" (3).

**Root cause.** The system prompt's conviction rubric defines 5 as
"unambiguous direct evidence" and adds "pick the lower number when
uncertain." Sonnet 4.6 honours the second rule when there's no
evidence (returns 1) and when there are explicit contradictions (the
rubric caps at 3). It doesn't honour it on the **inferential
boundary** — when *some* of the answer is literally stated and *some*
is inferred, the model averages up to 5 rather than down to 3 or 4.

This is a known LLM behaviour: when given a multi-part judgement
("how confident are you in this answer?"), models tend to anchor on
the strongest part rather than the weakest. The rubric needs to
explicitly anchor on the weakest claim, not the strongest.

**Fix (proposed; not applied).** Two complementary edits:

1. Sharpen the rubric to require literal-statement support for 5:
   *"5 requires that every load-bearing claim in the answer is
   literally stated in a cited passage. If any part of the answer is
   inferred or follows logically from the passages without being
   stated, conviction caps at 4."*
2. Add an explicit weakest-claim anchor:
   *"Conviction is bounded by the weakest claim in the answer, not
   the strongest. If your answer has three claims and one of them is
   only weakly supported, the whole answer's conviction is the
   weakest claim's conviction."*

Optional third layer: a small self-check pass after the main
synthesis call that scrutinises the conviction value against the
rubric. Doubles cost; arguably worth it if calibration matters more
than latency.

**Lesson.** *Rubrics that work at the extremes can fail at the
boundary.* Empirical testing across the 1-5 conviction scale showed
the rubric works correctly when there's clearly no evidence (1) and
clearly direct evidence (5 on a single direct quote), but fails on
the mixed-inference cases that account for most real questions.
"Pick the lower number when uncertain" isn't sharp enough — the
model needs to be told **which dimension** to be uncertain along
(the weakest claim), and **what threshold** distinguishes "stated"
from "inferred".

Prompt engineering at this level is the work of LLM applications.
The strict-RAG schema and runtime checks catch *fabricated* citations
and *empty* citations; they don't catch *over-confident* citations.
That has to live in the prompt.

---

## Finding 21 — Dense-only retrieval misses rare-keyword queries

**Symptom.** A query for *"what is the name of the conjuncatvitus
spell?"* (typo deliberate) returned 10 candidates from
`hybrid_search`, none of which contained the Conjunctivitis Curse.
The synthesis agent correctly returned `conviction=1/5` with `gaps`
listing the spells the passages *did* cover (Levicorpus, Mufflato,
Imperius, etc.) and explicitly noting the absence.

The Conjunctivitis Curse is in the corpus — it appears in *Goblet of
Fire* chapters 19, 20, 23 (Krum's dragon attack, the First Task, the
Yule Ball aftermath) and in *Order of the Phoenix* chapter 20
(Hagrid's Tale). Retrieval missed all of it.

**Root cause.** Three compounding reasons:

1. **Query typo** — `conjuncatvitus` rather than `conjunctivitis`.
   bge-large is somewhat robust to typos at the subword tokenizer
   level, but not infinitely.
2. **Phrasing mismatch** — the books describe Krum casting a curse
   at the dragon's eyes; they don't repeatedly say "the
   Conjunctivitis Spell." Dense embeddings match on semantic gist;
   the gist of *"what is the name of the X spell"* doesn't strongly
   resemble *"Krum sent a stream of red fire from his wand at the
   dragon's eyes."*
3. **Rare keyword vs dense compression** — *"conjunctivitis"*
   appears maybe 4-5 times across 7 books. Dense embeddings
   compress lexical signal away in favour of semantic gist. **This
   is the textbook BM25 case**: a rare token has high IDF and
   sparse retrieval would weight it heavily; dense retrieval treats
   it as one feature among 1024.

**Fix (applied).** Added a BM25 retrieval path. Two implementation
choices were available:

1. **Qdrant native sparse vectors** — supported via `prefetch` +
   `FusionQuery(fusion=RRF)`. Schema migration needed (named-vector
   collections), re-embed with both dense and sparse encoders,
   server-side fusion. Right answer at scale.
2. **In-memory BM25** via `rank-bm25` — load chunks.json once at
   process start, build two `BM25Index` instances (paragraph + chapter),
   ~1s build cost, ~6ms per query. No schema migration, no second
   datastore, no re-embed. Right answer at *this* scale.

We picked (2). The corpus is 17MB of raw text and ~5,500 chunks; an
in-memory BM25 index fits in tens of MB and out-paces Qdrant's
sparse-vector path purely on round-trip overhead. The lab's vector-DB
is no longer doing both modalities — Qdrant remains for dense ANN, an
in-memory index handles BM25. See ADR-0008 for why this choice fits
the lab specifically and would not generalise to production scale.

Architecture after the fix:

```
hybrid_search (LangGraph state machine):
  ─ paragraph_dense  (Qdrant ANN)        ┐
  ─ chapter_dense    (Qdrant ANN)        ├──> RRF fuse ──> top_k
  ─ paragraph_bm25   (in-memory rank-bm25)
  ─ chapter_bm25     (in-memory rank-bm25)─┘
```

Four parallel retrievers fanning out from `START`, converging on a
fuse node. `horcrux/retrieval_graph.py` holds the state machine;
`horcrux/bm25.py` holds the index. Synthesis is unchanged — it consumes
ScoredCandidates regardless of which retriever produced them.

Verification on the original failing query — the literal text
*"...I was going to suggest a Conjunctivitis Curse, as a dragon's
eyes are its weakest point — 'That's what Krum did!' Hermione
whispered..."* (B4.C23 The Yule Ball) now appears in the synthesis
candidate set. Strict-RAG can ground from it.

**Lesson.** *"Hybrid retrieval" is overloaded.* The paragraph+chapter
fusion we built in Phase 4 is **granularity hybrid** — fusing two
chunk sizes of the same modality. The conjunctivitis case shows we
also need **modality hybrid** — dense + sparse. They solve different
problems:

- Granularity hybrid: chapters give breadth, paragraphs give
  precision. Fixes the precision/recall trade-off within a single
  retrieval modality.
- Modality hybrid (now also done): dense gives semantic match, sparse
  (BM25) gives rare-keyword match. Fixes the gist/lexical trade-off.

Most production RAG systems do both. The lab now does both. Composing
the two as a **four-way fusion** (paragraph_dense × paragraph_bm25 ×
chapter_dense × chapter_bm25) was the natural shape — and a clean fit
for LangGraph, which is why `hybrid_search` now lives as a state machine
rather than as a flat async function. The four retrievers fan out from
START, run in parallel, converge on a fuse node, RRF-merge.

A meta-observation worth recording: **the strict-RAG architecture
revealed a retrieval problem rather than masking it.** A pipeline
that allowed parametric fallback would have answered "the
Conjunctivitis Curse" confidently from training data, and we'd
never have known the retrieval missed. Strict-RAG converts a silent
retrieval failure into a visible "the passages don't cover this"
gap. That's exactly what we want from the architecture — it makes
weak retrieval **observable** rather than papered-over.

---

## Pattern across these findings

**The ones we caught by inspecting real output**, not by writing tests in
advance: 1, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14.

**The ones we caught from architectural reflection**, not from a specific
bug: 11, 15. Both are "the design is OK at small scale but won't survive
production scale" calls — the prescient kind, made in time to avoid
later refactors. Synthetic tests proved correctness against
*expected* inputs; real corpus revealed the *unexpected* shapes that
break the system. This is the strongest argument for inspecting real
output at every iteration, not just running unit tests green.

**The ones we caught by hitting a hard limit**: 1, 2. Memory and timeout
are physical constraints that synthetic tests at small scale don't
exercise. The lesson: test at *production-shaped* sizes, not just
correctness-shaped ones.

**The ones with structural fixes (not configurational)**: 1, 3, 4, 5,
6, 8, 9, 10, 11, 12, 13, 14, 15. These are the most pedagogically valuable because the fix changes
the design, not just a parameter. Configurational fixes (Findings 2, 7)
matter operationally but don't teach architecture — though Finding 7's
"manual override layer" is structural in its own right (the *existence*
of the override layer is the design choice, even if its contents are
manual).

**Common failure mode**: trusting upstream *defaults* — library defaults
(Finding 1), naive concurrency assumptions (Finding 2), regex-on-derived-
signal vs use-the-author's-metadata (Finding 3), naive subset clustering
(Finding 4), naive concatenation (Finding 5), naive trust in coref output
(Finding 6), naive trust in NER classification on out-of-domain text
(Finding 7), bytes-level equality across probabilistic boundaries
(Finding 8), silent embedder truncation (Finding 17), silent vector-store
indexing threshold (Finding 18), dense-only retrieval blind to rare
keywords (Finding 21). Eleven findings, eleven instances of "the default
is wrong for this corpus / hardware / scale / domain." Not the libraries'
fault — they're tuned for the median use case. The lesson is structural:
every default in a data pipeline is a hidden assumption you should
surface before relying on it.

**The ones that surfaced from end-to-end testing of the full stack**:
19, 20, 21. Phase 5 (synthesis) was the first time the entire pipeline
ran together; F19 (a feared leak that hybrid retrieval prevented),
F20 (conviction calibration anchoring high), and F21 (rare-keyword
retrieval gap surfaced by strict-RAG, not papered over) all required
the full stack to demonstrate. *Unit-tested layers can be individually
correct and still produce composite failures only the integration
reveals.*

**The strict-RAG architecture's defensive value (Finding 21 in
particular)**: a pipeline that allowed parametric fallback would have
silently masked the retrieval gap. Strict-RAG converts silent
retrieval failures into visible "passages don't cover this" gaps.
This is the architectural equivalent of fail-fast: weak retrieval
becomes **observable** instead of plausible-but-wrong.

**Three findings (6, 7, 8) are specifically about probabilistic-model
output**: coref errors, NER domain shift, NER/coref boundary
disagreement. Common shape: ML-model outputs aren't ground truth, and
treating them as such corrupts every downstream consumer. The defensive
patterns — multi-PERSON cluster rejection, share-significant-token
gates, min_count thresholds, manual override layers, text-based
boundary-tolerant matching — are all variants of "don't trust
probabilistic upstream blindly; build a layer that catches its known
failure modes." This is the single most important architectural
lesson the lab surfaced.

---

*Last updated: 2026-04-25 (Phase 5 verification: F19, F20, F21 added; F21 fixed via in-memory BM25 + LangGraph fusion)*
*Findings list grows as the lab progresses; entries are append-only.*
