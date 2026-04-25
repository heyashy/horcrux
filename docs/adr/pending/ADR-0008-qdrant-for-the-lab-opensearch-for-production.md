# ADR-0008: Qdrant for the lab, OpenSearch for production

**Date:** 2026-04-25
**Status:** pending
**Pattern:** Tool selection (lab vs production trade-off)

## Context

The lab uses **Qdrant** as its vector store (collections `hp_chapters` and
`hp_paragraphs`, with payload filters on `book_num` / `chapter_num` /
`characters`; see ADR-0001 and Phase 3 / Phase 4 implementation). This
ADR exists because, on reflection, **Qdrant is not the right answer for
production deployments where an OpenSearch (or Elasticsearch) stack
already exists.**

Documenting the rationale matters for two reasons:

1. **Honesty.** This repo is a public portfolio artefact. If a reader
   walks in and sees Qdrant chosen without comment, they could
   reasonably conclude the author thinks Qdrant is universally
   superior. It isn't. Pretending otherwise would be marketing rather
   than engineering.
2. **Specificity.** "Pick the right tool for the job" is empty unless
   the job is named. This ADR names the job: a weekend lab evaluating
   six tools side-by-side, deliberately. The decision logic that
   produced "Qdrant" here would produce "OpenSearch" in a different
   context — and that's the point.

The trigger for writing this: a question during Phase 5 verification —
*"I use OpenSearch in prod for BM25. Technically it does vector search
too, right? So why use Qdrant?"* — which deserves a written, defensible
answer rather than hand-wave deflection.

## Decision

**Qdrant is correct for the lab; OpenSearch is correct for most production
deployments where OpenSearch is already in the stack.** Different decisions
for different contexts.

**For this lab, Qdrant.** The lab's stated purpose is evaluating six
specific tools (PydanticAI, LangGraph, Temporal, Qdrant, LiteLLM,
LangSmith). Removing Qdrant and using OpenSearch would mean the lab no
longer demonstrates Qdrant — defeating one-sixth of the lab's reason for
existing. Tool exposure is the optimization target, not deployment
optimality.

**For production with existing OpenSearch infrastructure, OpenSearch.**
Adding a second specialised datastore introduces real operational cost
(backups, monitoring, version upgrades, security review, on-call
runbooks, capacity planning) that the marginal Qdrant feature wins
generally don't justify. Consolidation almost always beats specialisation
unless the specialised tool offers a step-change capability — and for
typical RAG workloads at most scales, vector search in OpenSearch is
"good enough" that the step change isn't there.

**The two technically-defensible signals that flip the answer back to a
specialised vector store** (Qdrant, Weaviate, Milvus, Pinecone):

1. Vectors are the primary workload (≥80% of queries are vector ANN, not
   text retrieval / filtering / aggregation).
2. You need a specialised vector capability that the general search
   engine doesn't yet support cleanly: e.g. multi-vector retrieval
   (ColBERT-style), per-collection HNSW tuning, advanced quantisation
   (binary / product), or sparse-and-dense fusion via the modern
   server-side fusion API. These are nice-to-haves for most workloads
   and load-bearing for some.

Outside those two signals, OpenSearch (or Elasticsearch) wins on
operational simplicity for any team that already runs one.

## Alternatives Considered

### OpenSearch / Elasticsearch as the only datastore

Rejected for the lab; **the right answer for the user's production
context.**

Pros:
- Lucene-backed BM25 — best-in-class. The F21 case (rare keyword
  retrieval failure) would be solved natively by `match` queries with
  IDF weighting, no second index needed.
- k-NN plugin with HNSW; cosine / L2 / IP distance metrics. Solves the
  same dense-retrieval problem Qdrant solves.
- Hybrid search in a single query — `knn` + `match` clauses combined,
  optionally with a search-pipeline RRF processor (added in OpenSearch
  2.10+). The four-way fusion this lab will need in Phase 4.5
  (paragraph-dense × paragraph-sparse × chapter-dense × chapter-sparse)
  composes naturally in OpenSearch.
- Aggregations, faceting, complex filters, query DSL — full search
  engine surface. The lab's `characters` payload filter would be a
  trivial `terms` query.
- Battle-tested operationally. Snapshots, rolling upgrades, multi-AZ
  replication, security plugin — all production-mature.

Why this lab doesn't use it:
- Removing Qdrant means the lab no longer evaluates Qdrant. The lab is
  a teaching artefact; the tool inventory is the deliverable.
- The lab is single-developer, local, ephemeral. The operational
  benefits OpenSearch offers don't matter at this scale.

### Elasticsearch (the proprietary fork that became Elastic)

Functionally similar to OpenSearch for vector + BM25 purposes;
licensing is the difference. OpenSearch is Apache 2.0; Elastic moved
parts of the stack to SSPL / Elastic License. For most teams the
choice is determined by what's already deployed, not by capability —
the technical features track each other closely. Not relevant to the
lab's choice; included for completeness.

### Pinecone / Weaviate / Milvus

Other specialised vector stores. Rejected for the lab in favour of
Qdrant for stack-fit reasons (Qdrant's API was already represented in
PydanticAI examples and LangChain integrations being evaluated).
Conceptually the same trade-off applies: specialised vector store vs
general search engine.

In production, the choice between Qdrant / Weaviate / Milvus / Pinecone
is its own ADR — managed-vs-self-hosted, cost model, language SDK
quality, multi-tenancy support. Out of scope for this document; the
relevant decision here is "specialised vs general," not "which
specialised."

### Postgres + pgvector

A third option that's increasingly defensible: pgvector lets Postgres
hold vector indexes alongside relational data. Strong choice when the
existing stack already has Postgres and the vector workload is
modest (≤1M vectors, low QPS).

For the lab, pgvector wasn't considered seriously — Qdrant was named in
the lab's tool inventory. For production with an existing Postgres
stack and modest vector needs, pgvector deserves a serious look before
adding either Qdrant or OpenSearch as a new piece of infrastructure.

### Qdrant native sparse vectors (for the BM25 layer specifically)

When Finding 21 surfaced — dense-only retrieval missing rare-keyword
queries — the immediate choice was *how to add BM25*. Two paths:

1. **Qdrant native sparse vectors** with server-side
   `prefetch + FusionQuery(fusion=RRF)`. This is what a production
   system at scale would do. It requires migrating to named-vector
   collections (dense + sparse per point), re-encoding with a sparse
   model (e.g. `fastembed.SparseTextEmbedding(model_name="Qdrant/bm25")`),
   and re-upserting the corpus.
2. **In-memory BM25** via `rank-bm25` over `chunks.json`. ~17MB of raw
   text, ~5,500 chunks, ~1s build cost on process start, ~6ms per
   query. No schema migration, no second encoder, no re-embed.

We picked (2). At lab scale, in-memory BM25 wins on round-trip overhead
alone (a Qdrant gRPC call costs more than a full BM25 scan over 5,500
chunks). The decision generalises to any small-corpus RAG system: if
your corpus fits in RAM, an in-memory BM25 index is genuinely the right
answer over any networked sparse-vector path.

The choice does *not* generalise to production:

- Past ~50k–100k chunks the in-memory scan starts costing real time
  (linear in corpus size).
- Memory grows with corpus and you lose the "rebuild on startup" model.
- No persistence, no multi-process sharing, no incremental updates.

For production, the recommendation in the table above stands —
OpenSearch BM25 if OpenSearch is in the stack; Qdrant native sparse if
you're committed to Qdrant; full-text search in Postgres if you're
small enough for pgvector. The lab demonstrates the *shape* of
modality hybrid retrieval (dense + sparse + RRF fusion) without
committing to a specific scaling story for either side.

## Consequences

### What this lab demonstrates

The lab shows:
- Qdrant collection design (named vectors, payload indexes,
  filter-then-search patterns).
- The two-collection chunking approach (chapter + paragraph) and its
  trade-offs (Findings 17, 19).
- Hybrid retrieval via Qdrant's modern Query API (Phase 4 and the
  planned Phase 4.5 BM25 fusion).
- The Qdrant client → server protocol mismatch issue (server pinned
  v1.12.4, client moves faster — known divergence noted in code).

The lab does *not* demonstrate:
- BM25 quality at scale (Phase 4.5 will partially close this, but
  Lucene's BM25 is the gold standard and we won't be running against
  it).
- Operational realities of running a vector store in production
  (backups, scaling, security, monitoring at scale).
- Cost trade-offs of dedicated vs consolidated infrastructure.

### Recommendation for readers evaluating their own production setup

Default to your existing search engine if you have one:

| Existing stack | Recommendation |
| --- | --- |
| OpenSearch / Elasticsearch already in prod | Use its k-NN + BM25 hybrid. Add Qdrant only for points 1 + 2 in the Decision section above. |
| Postgres only, modest vector volume (≤1M vectors) | Use `pgvector`. Add a vector-specialist DB only when pgvector demonstrably tops out. |
| No existing search infrastructure, vectors are the primary workload | Specialised vector DB is defensible (Qdrant / Weaviate / Milvus / managed Pinecone). Choose by SDK quality, deployment model, and team familiarity. |
| Building greenfield with mixed text + vector + relational needs | OpenSearch + Postgres (or Postgres + pgvector for small-vector workloads) covers most cases. |

The lab's Qdrant choice is **load-bearing for the lab's purpose** and
**not a recommendation for the reader's stack**. Both statements need
to coexist for the lab to be honest.

### Lab-side follow-ups

None needed in code. This ADR is documentation-only — the codebase
already uses Qdrant correctly; nothing changes structurally. The README
is updated to point readers at this ADR for the production-equivalent
question.

## Rollback

Not applicable — this ADR records the rationale for an existing choice.
There is no infrastructure change to undo.

If a future lab phase decided to swap Qdrant for OpenSearch (e.g. to
demonstrate hybrid search in OpenSearch instead), the rollback would be
the implementation work itself: rewrite `horcrux/ingest.py` and
`horcrux/retrieval.py` to use the OpenSearch client, port the payload
filter shape, re-index. ADR-0001's "Qdrant" choice would supersede
this ADR with a new ADR explaining the swap.
