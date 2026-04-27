# Agentic-assisted architecture: a field report

*What a long weekend with AI coding agents, a 3,600-page literary corpus,
and explicit engineering discipline taught me about building real
software in the LLM era.*

---

## The premise

Most RAG demos you'll see online hand-wave the parts that actually
matter — chunking strategy, retrieval routing, grounding discipline,
durable ingest, multi-step planning, conversational memory, calibration.
The end result looks polished but breaks the moment you ask it
something the demo wasn't designed for.

I wanted to know what happens when you don't hand-wave any of it. So I
spent a long weekend building a deep-research RAG agent over a
3,600-page literary corpus, with one constraint: every architectural
decision had to be documented, every failure mode had to be catalogued,
and the whole thing had to be built collaboratively with AI coding
agents under explicit engineering discipline.

The output is a working system. But what's actually interesting is the
*method*. This is a field report on what agentic-assisted architecture
looks like in practice — what the agents are good at, where the human
has to stay in the loop, what failure modes you only see when you
actually run the thing, and why I think the most valuable artefact in
the repo isn't the code.

---

## What got built

The system is called **Horcrux** — a deep-research agent over a
literary corpus. It has three modes:

1. **Single-shot answer.** Ask a focused question, get a grounded
   answer with citations. Sub-second retrieval, ~10s synthesis.
2. **Multi-turn chat.** Conversation history threads through, so
   follow-up questions like *"what was his motive?"* can resolve
   pronouns against prior turns.
3. **Multi-step research.** Ask a multi-faceted question; watch a
   planner decompose it into focused sub-questions, see them run in
   parallel, get a coherent multi-paragraph answer with citations
   drawn from across the corpus.

Research mode is the demo. Ask the system *"tell me about Snape's
story arc"* and you watch this happen in real time:

```
▶ Planning…
   ↳ What is Snape's role in the early books and his relationship with Harry?
   ↳ What did Snape's memories in The Prince's Tale reveal about his loyalty?
   ↳ Why did Snape kill Dumbledore?
   ↳ How did Harry's perception of Snape change after learning the truth?

▶ Sub-queries (parallel)
   ⠋ What is Snape's role in the early books…              …running
   ⠋ Memories in The Prince's Tale…                        …running
   ⠋ Why Snape killed Dumbledore…                          …running
   ⠋ Harry's perception change…                            …running
   ✓ Snape's role in early books          10 hits, 7 cited, conviction 5/5, 29s
   ✓ Why Snape killed Dumbledore           10 hits, 4 cited, conviction 4/5, 30s
   ✓ Memories in The Prince's Tale         10 hits, 5 cited, conviction 3/5, 32s
   ✓ Harry's perception change             10 hits, 5 cited, conviction 3/5, 36s

▶ Synthesising final report…

  Severus Snape's story arc spans all seven novels, moving from apparent
  villainy to a posthumous revelation of hidden heroism…

  conviction: 4/5 (high)  citations: 19  gaps: 3
```

Nineteen citations spanning all seven books. Conviction of 4/5, not
5/5 — bounded by the weakest sub-finding, exactly the calibration
pattern good research agents need. Three honest gaps surfaced
explicitly (parts the corpus didn't cover), not papered over with
parametric guesses.

Architecture under the hood: a Temporal-driven durable ingest workflow,
a four-way hybrid retrieval graph (paragraph + chapter granularity ×
dense + BM25 modality, fused via Reciprocal Rank Fusion), a research
graph using LangGraph's `Send` primitive for parallel sub-query
fan-out, and three layers of strict-RAG enforcement (system prompt +
schema invariant + runtime in-range citation check).

That's the system. Now the actually interesting part.

---

## The thesis

**AI coding agents don't replace engineering rigor. They amplify it.**

This is the central claim, and it's worth being concrete about. There
are three states a developer can be in:

1. **Solo, no AI.** A senior engineer building this scope without AI
   assistance is realistically four to six weeks of focused work.
2. **AI as typist.** AI fills in code without engineering structure
   around it. Faster than (1), but ships without ADRs, without a
   findings catalog, without TDD where it adds signal, without
   layered verification. This is what most "AI built it!" demos
   actually are. They look polished and break the moment you ask
   them something the happy path didn't anticipate.
3. **AI plus discipline.** Same agents, but operating inside an
   explicit engineering structure: typed contracts at every LLM
   boundary, ADRs for non-trivial decisions, TDD on pure functions,
   a findings catalog, phase-by-phase verification, change log,
   reproducible setup. A long weekend lands roughly the same place
   a multi-week sprint would.

The compounding effect of (3) over (2) matters more than raw speed.
Most documentation gets deferred under time pressure: ADRs become
"I'll write it later", findings become tribal knowledge, change logs
go unwritten. Here the agents kept docs in lockstep with code because
every task's definition of done *included* the doc. The
meta-acceleration isn't shipping faster — it's shipping with the
documentation that makes future change cheap.

Strip the discipline out and AI output is unverified noise. Keep it,
and what was a multi-week project is a long weekend.

---

## The dev cycle, concretely

Each phase of the lab followed the same eight-step rhythm:

1. **Brainstorm out loud.** Human and agent agree on the shape of
   the next phase. What problem are we solving? What does done look
   like? What might go wrong?
2. **ADR if the decision is non-trivial.** A new file in
   `docs/adr/pending/`, structured: context, decision, alternatives
   considered, consequences, rollback. Writing this *before* the
   work clarifies thinking and surfaces alternatives the agent
   wouldn't otherwise consider.
3. **Tests-first where it adds signal.** Pure functions (chunking
   thresholds, RRF fusion, fuzzy matching, schema invariants) get
   unit tests written before the implementation. Glue code (wiring
   an agent into a graph) skips TDD because the tests would just
   mirror the wiring.
4. **Skeleton with stubs.** Wire the topology end-to-end with
   placeholder logic. For research mode, this meant a stub planner
   returning hardcoded sub-questions and a stub aggregator that
   concatenated outputs. The streaming UX got debugged against
   stubs *before* the real LLMs were added.
5. **Live verification.** Run the system end-to-end. Eyeball the
   output. Notice what's wrong.
6. **Replace stubs with real logic.** Real Haiku planner, real
   Sonnet aggregator. Now the only thing changing is the
   intelligence; the topology is already known to work.
7. **Findings if surprises.** When the system fails in a way that
   matters, write it up: Symptom, Root cause, Fix, Lesson. Append
   to `docs/lab/findings.md`.
8. **One coherent commit.** Push.

This rhythm sounds slow. It isn't. Most phases were two to six hours
of focused work. The skeleton-first step is load-bearing — it lets
you debug each layer in isolation, which means when a bug shows up
later you know exactly where it lives.

---

## Where the agents excelled

Concrete examples from the lab:

**Mechanical refactors.** Late in the project, the `horcrux/`
package was a flat directory of seventeen modules with names like
`embedding.py`, `chapters.py`, `aggregator.py`, `planner.py`. You
couldn't see the architecture from `ls`. Reorganising into five
concern-grouped subpackages (`corpus/`, `retrieval/`, `agents/`,
`research/`, `chat/`) involved moving 17 files with `git mv`,
rewriting roughly 50 import statements across `horcrux/`, `scripts/`,
and `tests/`, and catching a circular-import bug that emerged when
the new `__init__.py` files re-exported from submodules that
themselves imported through the package. End to end, about thirty
minutes. The unit test suite confirmed zero behavioural regression
instantly. Solo, this is the kind of refactor that accumulates as
months of debt because the mechanical work is tedious and error-prone.

**Boilerplate-heavy modules.** Pydantic models, LangGraph state
machines, Rich rendering loops. The patterns are stable; agents fill
them in correctly given clear specs. Asking "build me a typed
`Plan` model with 1-8 sub-questions and a rationale field" produces
working, validated code immediately. Reviewing it takes seconds.

**End-to-end smoke verification.** Agents are excellent at running
queries, parsing live output, and surfacing the specific failure
modes that motivate the next round of changes. The conjunctivitis
curse case (more on this below), the UUID transcription bug, the
rate-limit failure — each was discovered by running the system, and
the agent ran the experiment while the human read the result.

**Documentation discipline.** Once one ADR was written in the
standard shape (context / decision / alternatives / consequences /
rollback), every subsequent ADR matched it. Same for findings —
the Symptom / Root cause / Fix / Lesson template held across all
twenty-two entries without prompting. Consistency that's hard to
maintain solo under time pressure becomes free.

---

## Where the human stayed in the loop

The agents were excellent at execution. Architecture and judgement
stayed human:

**Architectural decisions.** Every ADR's "Decision" section is a
human call after considering the alternatives the agent surfaced.
Example: when adding BM25 retrieval, the agent proposed Qdrant's
native sparse vectors as the textbook fix. The human looked at corpus
size — 17MB of raw text, 5,500 chunks — and said *"an in-memory
`rank-bm25` index is the right call at this scale; Qdrant native
sparse is right at production scale, but we'd be over-engineering
this lab."* The human was correct, and ADR-0008's alternatives
section now documents both paths and the conditions under which each
applies. The agent was *not* wrong — it gave the production answer.
The human was making a contextual judgement.

**Catching parametric leakage.** Late in the lab, while watching the
research planner decompose *"tell me about Snape's story arc"* into
sub-questions, one of those sub-questions was *"What did Snape's
memories in **The Prince's Tale** reveal..."*. That's a chapter title
from the seventh book. The planner *knew* parametrically that this
chapter exists — it has read every Harry Potter book during training.
The agent would have shipped without flagging it. The human caught it
and documented it as a known-good limitation: parametric planning
works for famous corpora and would fail for proprietary ones. The
"grounded planning" alternative (broad-retrieve before decomposing)
got captured as an ADR-0009 limitations section.

**Calibration judgement.** Early synthesis tests returned conviction
5/5 on questions where the evidence was indirect. The agent biased
toward the strongest part of its answer. The human pushed back: *"the
passages prove Harry loved Ginny — they don't prove Harry didn't love
Ron."* Finding 20 documents the drift; the planner-aggregator
architecture organically resolved it because the aggregator's
conviction is bounded by the weakest sub-finding's conviction. That
architectural fix would not have been obvious without the human
calibration push.

**Deciding when to stop.** *"This is complete; the next step is
weeks of optimisation"* is a scope call, not a technical question.
The lab's discipline includes knowing when not to keep building.

---

## The failures — case studies

This is the section that earns the article's claim that the agents
didn't ship a polished demo. They shipped a system that broke in real
ways, and the loop that fixed each failure is the actual interesting
content. Three case studies:

### Case 1: The Conjunctivitis Curse

**The query:** *"What is the name of the conjunctivitis spell?"*

**The bug:** The system returned ten candidates, none of which
contained the Conjunctivitis Curse, and the synthesis agent correctly
returned conviction 1/5 with `"the provided passages do not contain
any reference to a conjunctivitis spell"`.

But the spell *is* in the corpus. It appears in *Goblet of Fire*
chapters 19, 20, and 23 — Krum's dragon attack, the First Task, the
Yule Ball aftermath.

**The root cause:** Three compounding factors. First, dense embeddings
compress lexical signal away in favour of semantic gist; the rare
token "conjunctivitis" appears maybe four or five times across seven
books and gets compressed into one feature among 1,024 dimensions.
Second, the books describe Krum casting "a curse at the dragon's
eyes" — they don't repeatedly say "the Conjunctivitis Spell"
verbatim, so the semantic match is weak. Third, the original query
had a typo (*conjuncatvitus*) that bge-large couldn't recover from.

This is the textbook case for sparse retrieval. BM25 weights tokens
by inverse document frequency; *"conjunctivitis"* has very high IDF
because it's rare; chunks containing it would rank first.

**The fix:** Added an in-memory BM25 index alongside the dense one.
Modified the LangGraph retrieval state machine to fan out to four
retrievers in parallel (paragraph × dense, paragraph × BM25,
chapter × dense, chapter × BM25), fused via Reciprocal Rank Fusion.
~6ms per BM25 query at lab scale, no schema migration, no second
datastore.

After the fix, the same query returns the literal Sirius-letter
passage from the Yule Ball chapter — *"I was going to suggest a
Conjunctivitis Curse, as a dragon's eyes are its weakest point —
'That's what Krum did!' Hermione whispered"* — at conviction 5/5.

**The architectural insight:** *"Hybrid retrieval"* is overloaded.
There's *granularity hybrid* (chapter + paragraph chunks fused) and
*modality hybrid* (dense + sparse fused). They solve different
problems. Most production RAG systems do both. The lab now does
both, with the discovery driven by an actual failure rather than
checked off a best-practices list.

But here's the more important point about strict-RAG. **The
architecture made this failure visible.** A pipeline that allowed
parametric fallback would have answered "the Conjunctivitis Curse"
confidently from training data, and we'd never have known the
retrieval missed. Strict-RAG converts a silent retrieval failure
into a visible *"the passages don't cover this"* gap. That's exactly
what you want from the architecture.

### Case 2: The UUID transcription bug

**The query:** Snape's story arc, mid-development.

**The bug:** The chat REPL crashed mid-answer with `Agent returned
source_ids not in candidate set`. Looking at the IDs:

- Expected: `0fd826bf-b698-54b8-9c4c-99e2abb32017`
- Returned: `9d826bf-b698-54b8-9c4c-99e2abb32017`

The model dropped the leading `0f` and synthesised `9` while
transcribing the UUID into a citation.

**The root cause:** Asking an LLM to copy 36-character alphanumeric
strings character-by-character is asking for trouble. Sonnet is
excellent at reasoning, mediocre at lossless transcription of long
identifiers. The strict-RAG runtime check (Layer 3 of the three
enforcement layers — system prompt, schema invariant, runtime
membership check) correctly rejected the fabricated citation rather
than letting it reach the user.

**The fix:** Stop showing the model UUIDs entirely. Number the
passages `[1]`, `[2]`, `[3]` in the prompt; have the model cite
*"source_ids": ["1", "3"]*; translate digits back to real chunk IDs
in the wrapper. The model can't fail to copy a single-digit number.
The runtime check tightens too — instead of *"is this string in a
set of 10 UUIDs"*, it's *"is this digit in [1..10]"*. Much harder to
fail.

**The architectural insight:** Layer 3 of strict-RAG saved a
user-visible failure. Without it, the citation would have rendered
with a broken link, the user would have lost trust silently, and we'd
have had no signal that anything was wrong. Defensive architecture is
worth the boilerplate.

### Case 3: Chapter chunks blew the rate limit

**The bug:** Mid-Snape-arc query, the chat REPL hit a 429 response
from Anthropic — *"This request would exceed your organization's rate
limit of 30,000 input tokens per minute."* Single query.

**The root cause:** Chapter-level chunks store the full chapter text
in their Qdrant payload (3-5k tokens each), even though only the
first 512 tokens are embedded for similarity search. A top-10
candidate set with two chapter hits was sending ~30k tokens to Sonnet
in one synthesis call.

**The fix:** Truncate chapter chunks to a 200-word head snippet in
the synthesis prompt. Paragraph chunks pass through whole because
they carry the actual evidence. Chapter chunks are in the candidate
set for topic / breadth signal (and for RRF score reinforcement when
they overlap with paragraph hits) — neither requires the chapter's
full text in-prompt.

**The unexpected benefit:** The agent now cites only paragraph chunks
that genuinely contain the answer, not chapter chunks. Constraining
the prompt also constrained over-citation. A bug fix that turned out
to fix something else too.

**The architectural insight:** Embedded representation and stored
representation are independent decisions. The chapter chunk's
embedding only ever saw 512 tokens, but its payload stored the whole
chapter. Downstream consumers (the synthesis prompt) didn't know the
difference until a rate limit forced them to.

---

## The findings catalog as the actual portfolio piece

Anyone can clone an AI-built RAG system and demo a happy path. Few
have a 22-entry document explaining where their system silently fails
and why.

Each entry follows the same template:

> **Symptom:** What you see when the system breaks.
>
> **Root cause:** Why it actually broke (often subtler than it looks).
>
> **Fix:** What changed.
>
> **Lesson:** The general principle to extract.

Some of the entries:

- **F1**: Streaming inference vs batched — coref ran out of memory on
  the full corpus until we streamed per-chapter.
- **F11**: LLM in the ingest pipeline destroys reproducibility — Tier
  2 (LLM-driven character merge) was dropped because non-determinism
  at ingest time corrupts every chunk's character payload across runs.
- **F17**: Embedders silently truncate at the context window —
  bge-large's 512-token cap means *"chapter-level"* embeddings are
  really *"chapter-opening-level"* embeddings. Three layers of
  abstraction passed the truncation through without flagging.
- **F18**: Qdrant silently brute-forces below the indexing threshold
  — the HNSW config we set was descriptive, not active, until the
  collection grew past 20k vectors.
- **F20**: Conviction calibration anchors high without literal-
  statement anchoring — the rubric works at the extremes (1 when
  there's no evidence, 5 when there's a direct quote) but fails at
  the inferential boundary. The planner-aggregator architecture
  resolved this organically by bounding aggregator conviction to the
  weakest sub-finding.
- **F22 candidate** (parametric leakage at the planner): the
  research planner uses Sonnet's training-time knowledge of Harry
  Potter to write sub-questions that name specific chapters. Works
  for famous corpora; fails for proprietary ones; documented as a
  known-good limitation rather than fixed.

Most of these were caught by running the system and watching it
produce a wrong answer. The catalog isn't just engineering hygiene —
it's a literal *map of where the system is unreliable*. That's the
artefact future-me will use when changing this code, and that's the
artefact reviewers should look at when judging whether the developer
understands their own system.

---

## The stack

Six tools, each picked for a specific role:

| Tool | Role | What it earned |
|---|---|---|
| **PydanticAI** | Typed boundary at every LLM call | Schema-as-contract; auto-retry on validation failure; `result_type` makes hallucinated structure impossible. |
| **LangGraph** | Orchestration for retrieval and research graphs | Conditional edges, parallel fan-out via `Send`, native streaming via `astream`. The streaming-debug events are what made research mode's visible reasoning UX work. |
| **Temporal** | Durable ingest workflow | Crash mid-OCR, restart, resume from last completed batch — proven by deliberate kill test on a 22-minute run. |
| **Qdrant** | Vector storage | Two collections × payload-filtered ANN. Clean API at lab scale. **For production with existing OpenSearch, the lab explicitly recommends OpenSearch hybrid search instead** — Qdrant is the lab's choice because the lab evaluates Qdrant. |
| **LiteLLM** | Model router (proxy at `localhost:4000`) | One-line model swap via YAML; provider-agnostic; spend tracking; in-memory response cache during dev. |
| **LangSmith** | Observability | Full graph trace per query; visualises conditional routing as a tree. Auto-instrumented via two env vars. |

Three additional libraries earn quiet credit: `bge-large-en-v1.5` for
dense embeddings, `rank-bm25` for the in-memory sparse index (chosen
deliberately over Qdrant native sparse for lab-scale pragmatism, with
the production trade-off documented in an ADR), and `fastcoref` plus
spaCy NER for the three-tier character discovery system.

The honest framing: **this stack is one defensible answer for a
specific problem at a specific scale**, not a universal best-practice
recommendation. The lab spends real effort calling out where each
choice would change in production — most explicitly in ADR-0008
(Qdrant vs OpenSearch) and ADR-0009 (planner parametric leakage).

---

## Honest scope

A long weekend, deliberately. The point was to evaluate whether these
six tools compose well on a real-shaped problem — not to ship a SaaS.

The current state:

- Phases shipped: OCR ingest, chapter detection, three-tier character
  discovery, semantic chunking, dense embedding, hybrid retrieval
  (granularity + modality), single-shot synthesis, conversational
  chat, multi-step research with streamed reasoning.
- 9 ADRs documenting non-trivial decisions with alternatives and
  rollback.
- 22+ findings documenting empirical lessons from real runs.
- 203 unit tests + integration + smoke.
- 24 commits, each one a coherent step.

The next reasonable additions — clarification interrupts when
conviction is low, grounded planning for proprietary corpora, query
rewriting via LLM, learned re-ranking, contextual chunking — are
each their own multi-week project. Stopping here is the disciplined
call.

What's deliberately *not* in the scope:

- **Web UI.** The CLI is the product.
- **Production deployment.** No Terraform, no CI, no cloud.
  Local-only lab.
- **Long-context single-shot mode.** The lab's whole point is that
  you can do better than throwing a million tokens at a model.
- **Fine-tuning anything.** Off-the-shelf models throughout.

---

## What this means more broadly

A few things I think are true about agentic-assisted architecture
based on this lab:

**The bottleneck has moved from typing to thinking.** AI agents are
genuinely fast at the mechanical work — boilerplate, refactors,
test scaffolding, doc consistency. What's left for the human is the
work that requires judgement: which architecture, which trade-off,
which failure mode matters. The right division of labour is the
agent does the *what*, the human decides the *why* and the *when*.

**Engineering discipline scales the collaboration.** TDD, ADRs,
typed contracts, findings catalogues, layered verification — these
aren't bureaucracy. They're the structure that lets AI output be
*trusted* without being audited line by line. Strip them out and
you're left with unverified noise; keep them and a long weekend
ships what would otherwise be a multi-week project.

**The measurement loop is what makes the work correct.** When
something breaks — and it will — you don't ask the model to "try
harder." You identify the layer at fault, build the fix at that
layer, and verify the specific failing case now passes. The agents
accelerate each step in that loop; the loop itself is the engineer's
contribution.

**Strict-RAG isn't optional.** Three layers of grounding enforcement
caught real failures (the UUID transcription bug, the empty-retrieval
case, the over-citation that emerged from chapter-chunk verbosity).
A pipeline that allows parametric fallback would have shipped wrong
answers confidently. The architecture pays for itself the first
time it catches something.

**The findings catalog is the real portfolio piece.** Working code
demonstrates capability. A document that lists every place the
system silently fails demonstrates *understanding*. Most engineering
hiring signals at this point are about the latter, not the former.

---

## Closing

The repo is at <https://github.com/heyashy/horcrux>. The README has a
dedicated "How it was built" section that walks through the full
agentic + human-in-the-loop workflow, the three failure case studies
above (and more), and what each of the six tools earned. The findings
catalog at `docs/lab/findings.md` is the most useful document if
you've got fifteen minutes — even more so if you skim it
chronologically and watch the questions get harder.

If you're building something similar, working on AI-augmented
engineering tooling, or thinking through how teams should be using
these agents in production — I'd love to hear how you're approaching
it. The honest answer is that the right methodology for working with
AI coding agents is still being figured out by everyone, and
comparing notes seems valuable.

Happy to talk through any of the architectural decisions, the
specific failure modes, or the broader question of how engineering
discipline composes with AI-augmented workflows. DMs open.
