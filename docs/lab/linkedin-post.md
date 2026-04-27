# LinkedIn post — discovery + showcase

*~360 words, ~2 minute read. Designed for the LinkedIn feed: lead with
a counter-intuitive hook, show the demo, prove depth with two specific
failure stories, point at the repo for everything else.*

---

The most interesting artefact in the RAG system I just built isn't the code. It's a 22-entry catalog of every place the system silently fails.

Most AI-built RAG demos show you the happy path. This one (Horcrux — a deep-research agent over a 3,600-page literary corpus) ships with a findings document: 22 failure modes, each with root cause and lesson. The rare-keyword query my dense retrieval missed entirely — fixed by adding in-memory BM25 fusion. The UUID transcription error my strict-RAG runtime check caught before it reached the user — fixed by switching citations to passage numbers. The conviction-anchoring drift the planner-aggregator architecture organically resolved.

The system itself does what good research RAG should. Ask it *"tell me about Snape's story arc"* and you watch this happen:

```
▶ Planning…
   ↳ What is Snape's role in the early books and his relationship with Harry?
   ↳ What did The Prince's Tale reveal about his loyalty to Dumbledore?
   ↳ Why did Snape kill Dumbledore?
   ↳ How did Harry's perception change after learning the truth?

▶ Sub-queries (parallel)
   ✓ Snape's role in early books          conviction 5/5, 29s
   ✓ Why Snape killed Dumbledore           conviction 4/5, 30s
   ✓ The Prince's Tale revelations         conviction 3/5, 32s
   ✓ Harry's perception change             conviction 3/5, 36s

▶ Synthesising final report…
   [coherent multi-paragraph answer; 19 citations spanning all 7 books;
    conviction 4/5 — bounded by the weakest sub-finding, not anchored at 5/5]
```

Every step visible in the terminal as it runs. Multi-step planning, parallel sub-queries via LangGraph's `Send` primitive, strict-RAG enforcement at three layers (system prompt + schema invariant + runtime in-range citation check), 4-way hybrid retrieval (paragraph + chapter granularity × dense + BM25 modality, fused via Reciprocal Rank Fusion), durable Temporal-driven OCR ingest, multi-turn chat with conversational history.

Built collaboratively with Claude Code under explicit engineering discipline: 9 ADRs documenting non-trivial decisions with alternatives + rollback, 203 unit tests, layered phase-by-phase verification, daily change log, subpackage architecture readable from `ls`.

The honest framing: AI coding agents don't replace engineering rigor — they amplify it. Solo without AI: 4-6 weeks. With AI as a typist: fast but undocumented. With AI + the discipline (ADRs, findings, TDD, doc-in-definition-of-done): one long weekend ships what would otherwise be a multi-week project.

Stack: PydanticAI · LangGraph · Temporal · Qdrant · LiteLLM · LangSmith · bge-large · rank-bm25

Repo with full architecture writeup, ADRs, findings catalog, and a *"How it was built"* section that walks through the agentic + human-in-the-loop workflow:

🔗 https://github.com/heyashy/horcrux

If you're working on AI-augmented engineering, RAG architecture, or thinking through how teams should be using these coding agents in production — happy to compare notes.

#AIEngineering #RAG #LangGraph #PydanticAI #LLMOps

---

## Posting notes

- **Length** ~360 words. LinkedIn feed sweet spot is 200-400 — long
  enough to substantiate, short enough to read in 2 minutes.
- **Format on LinkedIn:** the transcript renders fine as plain text
  with the unicode characters (▶ ↳ ✓). LinkedIn doesn't support
  code-block syntax in feed posts but the symbols carry the visual
  hierarchy.
- **Image suggestion:** screenshot of the actual streaming research
  output in a terminal. Posts with images get 2-3x more impressions.
  Capture frame mid-run (with one sub-query still showing
  `…running`) to convey the live-streaming nature.
- **Best post time:** Tuesday-Thursday 9-11 AM in your timezone for
  engineering audience.
- **Tone:** the closing line is deliberately open ("happy to compare
  notes" rather than "looking for opportunities"). Recruiters
  reading this will infer the signal; engineers reading it will
  engage on technical merit. Both work.
- **Hashtags:** five is the sweet spot. Mixed — two hot
  (#AIEngineering, #RAG), three specific to the stack (#LangGraph,
  #PydanticAI, #LLMOps).
