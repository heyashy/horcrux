"""Research-mode planner — decomposes a query into focused sub-questions.

Planner is a Haiku-driven PydanticAI agent. Cheap, fast, and bounded by
the `Plan` schema (1-8 sub-questions with min_length / max_length
constraints — the model literally cannot produce a degenerate plan).

Why Haiku not Sonnet:
- Decomposition is a structural task, not a reasoning task. Sonnet's
  capability margin doesn't help here; latency does.
- Cost — research mode runs N+2 LLM calls per query (1 plan + N
  sub-syntheses + 1 aggregation). Keeping the planner cheap matters.
- Sonnet does the heavy lifting at the synthesis and aggregate layers
  where reasoning quality genuinely affects output.

Strict-RAG doesn't apply here directly — the planner produces
sub-questions, not claims. But the schema bounds (min/max sub_questions)
prevent the planner from exploding the workload (e.g., a 50-sub-question
plan that would burn the rate limit).
"""

from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from horcrux.config import settings
from horcrux.models import Plan

_SYSTEM_PROMPT = """You are a research planner. You will be given a \
research question about the Harry Potter novels, and your job is to \
decompose it into focused sub-questions that, taken together, will \
produce a complete answer.

DECOMPOSITION RULES:
- Aim for 3-5 sub-questions. Use fewer if the question is already \
focused (e.g. "who killed Cedric Diggory" can be answered directly \
without decomposition — return it as a single sub-question). Use more \
only for genuinely multi-faceted questions.
- Each sub-question should be answerable from a localised set of \
passages. Avoid sub-questions that themselves require synthesising \
evidence from across many books.
- Sub-questions should be COMPLEMENTARY, not overlapping. Do not \
produce two sub-questions that would retrieve the same passages.
- Phrase sub-questions as direct questions or instructions ("What did \
X say about Y", "Describe the events of Z"), not as sentence \
fragments.

EXAMPLES OF GOOD DECOMPOSITION:
- "Tell me about Snape's story arc" →
  - "What is Snape's role in the early books and his relationship \
with Harry?"
  - "What did Snape's memories in The Prince's Tale reveal about his \
loyalty?"
  - "Why did Snape kill Dumbledore?"
  - "How did Harry's view of Snape change after the war?"
- "What is the prophecy about the chosen one?" →
  - "What is the exact wording of the prophecy?"
  - "What did Dumbledore explain about the prophecy's meaning?"
  - "Why did Voldemort believe Harry was the subject of the prophecy?"

EXAMPLES OF BAD DECOMPOSITION (do not do this):
- "Who is Snape?" → ["Who is Snape?"]  (no decomposition needed but \
template-padding into multiple sub-questions wastes effort)
- ["What is Snape's name?", "What is Snape's character?", \
"What is Snape's role?"]  (overlapping; all retrieve the same passages)

OUTPUT:
- `sub_questions`: list of strings, 1-5 items, each one a focused \
research sub-question.
- `rationale`: one sentence explaining how the sub-questions cover \
the original query. Used for the trace; not shown in the final answer.
"""


@lru_cache(maxsize=1)
def _planner_agent() -> Agent[None, Plan]:
    """Build the Haiku-driven planner agent. Cached per-process."""
    provider = OpenAIProvider(
        base_url=f"{settings.litellm.base_url}/v1",
        api_key="lab-not-a-real-key",
    )
    model = OpenAIModel(settings.litellm.haiku_alias, provider=provider)
    return Agent(model, output_type=Plan, system_prompt=_SYSTEM_PROMPT)


async def plan_query(query: str) -> Plan:
    """Decompose `query` into research sub-questions via the Haiku
    planner. Returns a typed `Plan`."""
    if not query:
        raise ValueError("plan_query requires a non-empty query")
    agent = _planner_agent()
    result = await agent.run(f"Research question: {query}")
    return result.output
