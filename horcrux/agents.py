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
- `source_ids` must use the exact ID strings shown in the numbered context \
(the bracketed UUIDs at the start of each passage). Never invent IDs.
- Cite every passage that supports the answer, not just the first one.

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


def _format_context(candidates: list[ScoredCandidate]) -> str:
    """Render candidates as a numbered context block for the agent.

    Each entry includes the ID (so the model can cite it), location
    (book/chapter/title/page for human-readable trace), and the text.
    """
    lines = []
    for i, c in enumerate(candidates, start=1):
        lines.append(
            f"[{i}] id={c.id}\n"
            f"    Book {c.book_num}, Chapter {c.chapter_num}: {c.chapter_title} "
            f"(p.{c.page_start}, source={c.source})\n"
            f"    {c.text}"
        )
    return "\n\n".join(lines)


async def synthesise(query: str, candidates: list[ScoredCandidate]) -> Finding:
    """Run the synthesis agent and verify its citations.

    Raises:
        ValueError: if no candidates supplied (cannot synthesise from
            nothing) OR if the agent returns a source_id that isn't in
            the candidate set (the runtime layer of strict-RAG).
    """
    if not candidates:
        raise ValueError("synthesise requires at least one candidate")

    valid_ids = {c.id for c in candidates}
    prompt = (
        f"Question: {query}\n\n"
        f"Numbered passages:\n\n{_format_context(candidates)}"
    )

    agent = _synthesis_agent()
    result = await agent.run(prompt)
    finding = result.output

    invalid = [sid for sid in finding.source_ids if sid not in valid_ids]
    if invalid:
        raise ValueError(
            f"Agent returned source_ids not in candidate set: {invalid}. "
            f"Valid IDs were: {sorted(valid_ids)}"
        )
    return finding
