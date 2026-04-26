"""Research-mode aggregator — merges sub-findings into a final answer.

This module currently exposes a deterministic stub. The real Sonnet-driven
aggregator agent replaces `_stub_aggregate` once the streaming UX is
validated against the stub.

The aggregator is the second LLM call in research mode (after the planner
runs sub-queries through retrieve+synthesise). Its job is to read the
per-sub-question Findings, decide which claims to keep, reconcile any
contradictions, and produce a coherent top-level answer with deduplicated
citations.

Strict-RAG continues to apply at this layer: every citation in the final
answer must come from at least one sub-finding's source_ids; the
aggregator can't introduce new citations.
"""

from horcrux.models import Finding, SubFinding


def _stub_aggregate(query: str, sub_findings: list[SubFinding]) -> Finding:
    """Concatenation aggregator — reads each sub-finding's answer and
    stitches them together with bullet headers.

    Returns the union of source_ids from every sub-finding, deduplicated.
    Conviction is the *minimum* across sub-findings (the aggregator is
    only as confident as its weakest input). Gaps are concatenated.

    Stub behaviour, intentionally lossy: the real aggregator will reason
    over the sub-findings rather than concatenate them. But this gives
    the streaming UX something coherent to render and proves the schema
    composes.
    """
    if not sub_findings:
        raise ValueError("aggregator requires at least one sub-finding")

    sections = []
    for sf in sub_findings:
        sections.append(f"On {sf.sub_question!r}:\n{sf.finding.answer}")
    answer = "\n\n".join(sections)

    seen: set[str] = set()
    source_ids: list[str] = []
    for sf in sub_findings:
        for sid in sf.finding.source_ids:
            if sid not in seen:
                seen.add(sid)
                source_ids.append(sid)

    convictions = [sf.finding.conviction for sf in sub_findings]
    conviction = min(convictions)

    gaps: list[str] = []
    for sf in sub_findings:
        gaps.extend(sf.finding.gaps)

    return Finding(
        answer=answer,
        source_ids=source_ids,
        conviction=conviction,
        gaps=gaps,
    )


async def aggregate_subfindings(
    query: str, sub_findings: list[SubFinding]
) -> Finding:
    """Public API — produce a final synthesis Finding from the per-sub-query
    findings.

    Async-shaped from day one (the real Sonnet aggregator will be async);
    swap in the LLM agent without changing call sites.
    """
    return _stub_aggregate(query, sub_findings)
