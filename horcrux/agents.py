"""PydanticAI agents for the synthesis layer.

The synthesis agent is the *only* place in the pipeline that's allowed to
generate prose. Everything before it is deterministic. The agent's job is
to read retrieved passages and return a typed `Finding` — never to fill
in gaps from training data.

Three layers of strict-RAG enforcement:

1. **System prompt** — explicit "use only the passages" + conviction rubric.
   The first line of defence; cheap and effective.

2. **Schema** — `Finding.source_ids` has `min_length=1`, `conviction` is
   bounded 1-5. The model literally cannot produce an empty-citation answer
   and have it parse. PydanticAI auto-retries on validation errors.

3. **Runtime check** — `synthesise()` verifies every returned source_id
   exists in the candidates we passed in. Catches the case where the model
   invents a plausible-looking ID. This layer lives in `synthesise()`,
   not the agent itself, because the agent's job ends at "produce a typed
   Finding" — citation-set membership is a contract violation, not a
   model-output quality issue.

The model points at LiteLLM proxy (`localhost:4000`) using its OpenAI-
compatible interface. Application code only ever names `settings.litellm.
sonnet_alias` — the alias is resolved by the proxy from `litellm_config.yaml`.
This is the LiteLLM bet from ADR-0002: provider switches happen as a one-
line YAML edit, not a Python change.
"""

from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import Usage

from horcrux.config import settings
from horcrux.models import Finding, ScoredCandidate

# ── pydantic-ai ↔ LiteLLM compatibility shim ─────────────────────
#
# pydantic-ai 0.7.x flattens `prompt_tokens_details` into Usage.details by
# `model_dump()`-ing the OpenAI response object. LiteLLM forwards Anthropic
# prompt-cache fields, one of which (`cache_creation_token_details`) is
# itself a dict, not an int. The default `Usage.incr` then does `int + dict`
# and explodes mid-run.
#
# Cheapest fix: wrap incr to skip non-numeric values. Token accounting is
# a side-channel for the lab (LangSmith already tracks cost); dropping a
# nested-dict detail is not load-bearing.
#
# Remove this shim when we move to pydantic-ai >= 1.x — check whether the
# 1.x rewrite handles nested OpenAI-shaped usage details before pinning.

_original_incr = Usage.incr


def _safe_incr(self: Usage, incr_usage: Usage) -> None:
    if incr_usage.details:
        incr_usage.details = {
            k: v for k, v in incr_usage.details.items() if isinstance(v, int | float)
        }
    _original_incr(self, incr_usage)


Usage.incr = _safe_incr

_SYSTEM_PROMPT = """You are a careful research assistant answering questions \
about the Harry Potter novels. You will be given a question and a set of \
numbered passages retrieved from the books. Your job is to produce a typed \
Finding object with these rules:

GROUNDING (NON-NEGOTIABLE):
- Use ONLY the provided passages. Do not draw on outside knowledge of the \
books, even if you are confident you remember a fact correctly.
- If the passages don't contain enough to answer, say so in `gaps` and pick \
a low conviction. Do not pad the answer with parametric guesses.
- Every claim in `answer` must be supported by at least one passage you cite \
in `source_ids`.

CITATIONS:
- `source_ids` is a list of the passage NUMBERS that support your answer, \
as strings. Each passage in the context is prefixed with `[N]` where N is \
its number — use exactly that number (without the brackets). Example: if \
passages [1] and [3] support your answer, return `source_ids=["1", "3"]`.
- Cite every passage that supports the answer, not just the first one.
- Never invent passage numbers. Only cite numbers that appear in the \
provided context.

CONVICTION RUBRIC (1-5, pick the lower number when uncertain):
- 5: Unambiguous direct evidence. The passages literally state the answer.
- 4: Clearly implied by the passages, no contradictions, all key aspects \
covered.
- 3: Supported by the passages but with caveats — partial coverage, mild \
ambiguity, or one passage hedges.
- 2: Plausible from the passages but significant gaps remain, or the \
evidence is indirect.
- 1: The passages don't really answer the question.
- Contradictions between passages cap conviction at 3 and must be listed in \
`gaps`.

OUTPUT:
- `answer` is plain prose, no markdown, no bullet lists. Direct and complete.
- `gaps` lists anything the question asks for that the passages don't cover, \
plus any contradictions you noticed. Empty list if the passages fully cover \
the question.
"""


@lru_cache(maxsize=1)
def _synthesis_agent() -> Agent[None, Finding]:
    """Build the synthesis agent. Cached so the model + HTTP client are
    constructed once per process — the proxy connection is reused.
    """
    provider = OpenAIProvider(
        base_url=f"{settings.litellm.base_url}/v1",
        # LiteLLM proxy doesn't require a real key — any non-empty string
        # passes the SDK's "is set" check. The real provider key is
        # configured in litellm_config.yaml on the proxy side.
        api_key="lab-not-a-real-key",
    )
    model = OpenAIModel(settings.litellm.sonnet_alias, provider=provider)
    return Agent(model, output_type=Finding, system_prompt=_SYSTEM_PROMPT)


# Chapter chunks store the full chapter text (3-5k tokens each); a top-10
# candidate set with a few chapter hits trivially blows past Anthropic's
# default 30k input-tokens-per-minute rate limit. The chapter chunks are
# in the candidate set for *topic / breadth* signal, not for fine-grained
# evidence — paragraph chunks carry the evidence. Cap chapter snippets
# at this many words; paragraph chunks pass through whole. Tuned so a
# top-10 candidate set fits comfortably under the per-call budget.
_CHAPTER_SNIPPET_WORDS = 200


def _truncate_for_synthesis(candidate: ScoredCandidate) -> str:
    """Trim chapter chunks to a head snippet; paragraphs pass through.

    The synthesis agent only ever needs paragraph-level chunks for
    citation. Chapter chunks are useful for "what's the topic of B4
    chapter 23?" routing and for RRF score reinforcement when they
    overlap with paragraph hits — neither requires the chapter's full
    text in-prompt.
    """
    if candidate.source != "chapter":
        return candidate.text
    words = candidate.text.split()
    if len(words) <= _CHAPTER_SNIPPET_WORDS:
        return candidate.text
    return " ".join(words[:_CHAPTER_SNIPPET_WORDS]) + " […chapter continues…]"


def _format_context(candidates: list[ScoredCandidate]) -> str:
    """Render candidates as a numbered context block for the agent.

    Each entry shows its passage number ([1], [2], ...) and location
    (book/chapter/title/page). UUIDs are *not* shown — the model cites
    by number, and `synthesise_with_history` translates numbers back to
    real chunk IDs after the agent returns. Asking the model to copy a
    36-character UUID character-by-character produces transcription
    errors; numbers don't.

    Chapter-source chunks are truncated to a head snippet; paragraph
    chunks pass through whole. See `_truncate_for_synthesis`.
    """
    lines = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"[{i}] Book {c.book_num}, Chapter {c.chapter_num}: {c.chapter_title} "
            f"(p.{c.page_start}, source={c.source})\n"
            f"    {_truncate_for_synthesis(c)}"
        )
    return "\n\n".join(lines)


def _resolve_citations(
    raw_ids: list[str], candidates: list[ScoredCandidate]
) -> list[str]:
    """Translate model-returned passage numbers (`["1", "3"]`) to real
    chunk IDs.

    Accepts either bare digits (`"1"`) or bracketed forms (`"[1]"`) —
    Sonnet usually returns bare digits but the schema field is `str`,
    so anything is possible. Strips brackets and surrounding whitespace
    before parsing.

    Raises ValueError on out-of-range or non-numeric citations — that's
    the runtime layer of strict-RAG. Caller surfaces the error so the
    user sees the failure rather than getting a confidently-wrong
    answer with broken links.
    """
    resolved: list[str] = []
    invalid: list[str] = []
    for raw in raw_ids:
        cleaned = raw.strip().lstrip("[").rstrip("]").strip()
        try:
            idx = int(cleaned)
        except ValueError:
            invalid.append(raw)
            continue
        if not 1 <= idx <= len(candidates):
            invalid.append(raw)
            continue
        resolved.append(candidates[idx - 1].id)
    if invalid:
        raise ValueError(
            f"Agent returned passage numbers not in [1..{len(candidates)}]: "
            f"{invalid}. Sonnet sometimes hallucinates citations under load — "
            "rerun the query, or simplify it."
        )
    return resolved


async def synthesise(query: str, candidates: list[ScoredCandidate]) -> Finding:
    """Run the synthesis agent and verify its citations. One-shot, no
    conversational context — for `make answer` and similar single-query
    flows.

    Raises:
        ValueError: if no candidates supplied (cannot synthesise from
            nothing) OR if the agent returns a source_id that isn't in
            the candidate set (the runtime layer of strict-RAG).
    """
    finding, _ = await synthesise_with_history(query, candidates, message_history=None)
    return finding


async def synthesise_with_history(
    query: str,
    candidates: list[ScoredCandidate],
    *,
    message_history: list | None = None,
) -> tuple[Finding, list]:
    """Synthesis variant that threads conversational context.

    Pass the running message log from the previous turn as
    `message_history`; receive back the new message log to feed into
    the next turn. PydanticAI handles the role/content assembly — we
    only carry the opaque list around.

    The strict-RAG invariants apply identically: empty `source_ids` is
    a schema violation (PydanticAI auto-retries), and the runtime check
    rejects fabricated citations.

    Returns a `(Finding, all_messages)` tuple. The messages include the
    system prompt, every prior user/assistant turn, and the new turn —
    pass it back unmodified next call.
    """
    if not candidates:
        raise ValueError("synthesise requires at least one candidate")

    prompt = (
        f"Question: {query}\n\n"
        f"Numbered passages:\n\n{_format_context(candidates)}"
    )

    agent = _synthesis_agent()
    result = await agent.run(prompt, message_history=message_history)
    finding = result.output

    # Translate the agent's passage-number citations into real chunk IDs.
    # `_resolve_citations` raises on out-of-range or non-numeric values —
    # that's the runtime layer of strict-RAG enforcement.
    finding = finding.model_copy(
        update={"source_ids": _resolve_citations(finding.source_ids, candidates)}
    )
    return finding, result.all_messages()
