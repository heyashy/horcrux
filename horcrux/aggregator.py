"""Research-mode aggregator — Sonnet-driven synthesis across sub-findings.

The aggregator is the final LLM call in research mode. Its inputs are
the per-sub-question Findings (each already grounded by strict-RAG at
the synthesis layer). Its job is to:

  1. Read every sub-finding's answer, citations, and gaps.
  2. Reconcile any contradictions (and surface them in `gaps`).
  3. Produce a coherent top-level answer that draws on the strongest
     citations across all sub-findings.
  4. Cite by *passage number* into a flattened candidate set built
     from all sub-findings' candidates.

Strict-RAG continues at this layer, applied identically to the
synthesis layer:
  - System prompt forbids parametric knowledge.
  - Schema (`Finding.source_ids` min_length=1) is non-negotiable.
  - Runtime check: every cited passage number must be in [1..N] where
    N is the size of the merged candidate set.

The merged candidate set is constructed deduplicated by chunk ID — if
two sub-findings independently retrieved the same chunk, it appears
once in the aggregator's context.
"""

from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from horcrux.agents import _resolve_citations, _truncate_for_synthesis
from horcrux.config import settings
from horcrux.models import Finding, ScoredCandidate, SubFinding

_SYSTEM_PROMPT = """You are a careful research assistant producing the \
final answer to a multi-faceted question about the Harry Potter novels. \
You will be given:

  1. The original research question.
  2. A set of sub-findings — partial answers to focused sub-questions, \
each one already grounded in retrieved passages.
  3. A flattened, deduplicated set of numbered passages drawn from \
across the sub-findings.

Your job is to produce a single coherent Finding that answers the \
original question, drawing on the sub-findings as guidance and citing \
the underlying passages directly.

GROUNDING (NON-NEGOTIABLE):
- Use ONLY the provided passages. Do not draw on outside knowledge of \
the books, even if you are confident you remember a fact correctly.
- Every claim in `answer` must be supported by at least one passage \
you cite in `source_ids`.
- The sub-findings are guidance — they tell you what the planner \
thought was relevant — but your final answer must be grounded in the \
underlying passages, not in the sub-finding text.

CITATIONS:
- `source_ids` is a list of passage NUMBERS as strings, referring to \
the flattened numbered context. Example: if passages [1] and [3] \
support your answer, return `source_ids=["1", "3"]`.
- Cite every passage that supports your final answer. Pull from across \
sub-findings — your final answer should weave evidence from multiple \
branches if the question spans them.
- Never invent passage numbers.

CONVICTION RUBRIC (1-5):
- 5: Unambiguous direct evidence covering every load-bearing claim.
- 4: Clearly implied, no contradictions, all key aspects covered.
- 3: Supported with caveats — partial coverage, or one sub-finding \
hedges, or the sub-findings disagree on a minor point.
- 2: Significant gaps remain across multiple sub-findings, or evidence \
is indirect.
- 1: The passages don't really answer the question.
- Contradictions between sub-findings cap conviction at 3 and must be \
listed in `gaps`.
- When in doubt, pick the LOWER conviction. The aggregator's conviction \
is bounded by the weakest claim in the answer, not the strongest.

OUTPUT:
- `answer` is plain prose, no markdown, no bullet lists. Direct, \
complete, and coherent — read as a single response, not as concatenated \
sub-answers.
- `gaps` lists anything the original question asks for that the passages \
don't cover, plus any contradictions you noticed across sub-findings.
"""


@lru_cache(maxsize=1)
def _aggregator_agent() -> Agent[None, Finding]:
    """Build the Sonnet-driven aggregator agent. Cached per-process."""
    provider = OpenAIProvider(
        base_url=f"{settings.litellm.base_url}/v1",
        api_key="lab-not-a-real-key",
    )
    model = OpenAIModel(settings.litellm.sonnet_alias, provider=provider)
    return Agent(model, output_type=Finding, system_prompt=_SYSTEM_PROMPT)


def _merge_candidates(sub_findings: list[SubFinding]) -> list[ScoredCandidate]:
    """Flatten sub-findings into a deduplicated list of candidates.

    Order preserved by first-appearance. Two sub-findings that
    independently retrieved the same chunk get one entry; passage
    numbers are stable across the aggregation context.
    """
    seen: set[str] = set()
    merged: list[ScoredCandidate] = []
    for sf in sub_findings:
        for c in sf.candidates:
            if c.id in seen:
                continue
            seen.add(c.id)
            merged.append(c)
    return merged


def _format_aggregator_context(
    query: str,
    sub_findings: list[SubFinding],
    candidates: list[ScoredCandidate],
) -> str:
    """Render the aggregator prompt: original query + sub-findings +
    flattened numbered candidate set.
    """
    sub_sections = []
    for i, sf in enumerate(sub_findings, start=1):
        sub_sections.append(
            f"Sub-finding {i}: {sf.sub_question}\n"
            f"  conviction: {sf.finding.conviction}/5\n"
            f"  answer: {sf.finding.answer}\n"
            f"  gaps: {sf.finding.gaps if sf.finding.gaps else '(none)'}"
        )

    passage_lines = []
    for i, c in enumerate(candidates, start=1):
        passage_lines.append(
            f"[{i}] Book {c.book_num}, Chapter {c.chapter_num}: {c.chapter_title} "
            f"(p.{c.page_start}, source={c.source})\n"
            f"    {_truncate_for_synthesis(c)}"
        )

    return (
        f"Original research question: {query}\n\n"
        f"=== SUB-FINDINGS ({len(sub_findings)}) ===\n\n"
        f"{chr(10).join(sub_sections)}\n\n"
        f"=== FLATTENED PASSAGE SET ({len(candidates)}) ===\n\n"
        f"{chr(10).join(passage_lines)}"
    )


async def aggregate_subfindings(
    query: str, sub_findings: list[SubFinding]
) -> Finding:
    """Run the Sonnet aggregator over sub-findings; produce a final
    coherent Finding citing across the merged candidate set.

    Raises ValueError on empty input or on out-of-range citations
    (the runtime layer of strict-RAG, mirroring `synthesise_with_history`).
    """
    if not sub_findings:
        raise ValueError("aggregator requires at least one sub-finding")

    candidates = _merge_candidates(sub_findings)
    if not candidates:
        raise ValueError("aggregator received sub-findings with no candidates")

    prompt = _format_aggregator_context(query, sub_findings, candidates)
    agent = _aggregator_agent()
    result = await agent.run(prompt)
    finding = result.output

    # Translate passage-number citations to real chunk IDs and validate
    # in-range. Reuses the same _resolve_citations helper as the
    # single-shot synthesis layer; consistent runtime check across both.
    finding = finding.model_copy(
        update={"source_ids": _resolve_citations(finding.source_ids, candidates)}
    )
    return finding
