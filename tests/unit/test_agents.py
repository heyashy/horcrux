"""Synthesis agent tests — schema invariants + runtime citation check.

Mocks the agent via `pydantic_ai.models.test.TestModel` so we never hit
the LiteLLM proxy. The full end-to-end path (real model + real Qdrant) is
covered separately as a smoke target.
"""

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from horcrux import agents
from horcrux.agents import _format_context, synthesise
from horcrux.models import Finding, ScoredCandidate

pytestmark = pytest.mark.unit


def _candidate(cid: str = "id-a", text: str = "lorem") -> ScoredCandidate:
    return ScoredCandidate(
        id=cid,
        score=0.5,
        source="paragraph",
        text=text,
        book_num=1,
        chapter_num=1,
        chapter_title="t",
        page_start=1,
        characters=[],
    )


# ── Schema layer (the validator) ─────────────────────────────────


def test_finding_rejects_empty_source_ids():
    with pytest.raises(ValidationError):
        Finding(answer="x", source_ids=[], conviction=3)


def test_finding_rejects_conviction_above_5():
    with pytest.raises(ValidationError):
        Finding(answer="x", source_ids=["a"], conviction=6)


def test_finding_rejects_conviction_below_1():
    with pytest.raises(ValidationError):
        Finding(answer="x", source_ids=["a"], conviction=0)


def test_finding_gaps_default_empty():
    f = Finding(answer="x", source_ids=["a"], conviction=3)
    assert f.gaps == []


# ── _format_context ──────────────────────────────────────────────


def test_format_context_includes_each_id():
    cands = [_candidate("id-1"), _candidate("id-2"), _candidate("id-3")]
    rendered = _format_context(cands)
    assert "id=id-1" in rendered
    assert "id=id-2" in rendered
    assert "id=id-3" in rendered


def test_format_context_numbers_passages_in_order():
    cands = [_candidate("a"), _candidate("b")]
    rendered = _format_context(cands)
    # The numbering precedes the IDs, so [1] must come before [2].
    assert rendered.index("[1]") < rendered.index("[2]")


# ── synthesise() — runtime layer ─────────────────────────────────


async def test_synthesise_rejects_empty_candidates():
    with pytest.raises(ValueError, match="at least one candidate"):
        await synthesise("any question", [])


async def test_synthesise_rejects_fabricated_source_id():
    """The runtime layer: if the agent returns IDs not in the candidate
    set, we raise rather than trust the citation."""
    cands = [_candidate("real-id")]
    fake_finding = Finding(
        answer="x",
        source_ids=["real-id", "FABRICATED"],
        conviction=4,
    )
    test_model = TestModel(custom_output_args=fake_finding)

    agent = agents._synthesis_agent()
    with agent.override(model=test_model):  # noqa: SIM117
        with pytest.raises(ValueError, match="not in candidate set"):
            await synthesise("any question", cands)


async def test_synthesise_accepts_subset_of_candidates():
    """If the agent cites only some of the candidates, that's fine — the
    runtime check only flags fabrication, not under-citation."""
    cands = [_candidate("id-1"), _candidate("id-2"), _candidate("id-3")]
    finding = Finding(answer="x", source_ids=["id-2"], conviction=4)
    test_model = TestModel(custom_output_args=finding)

    agent = agents._synthesis_agent()
    with agent.override(model=test_model):
        out = await synthesise("any question", cands)
    assert out.source_ids == ["id-2"]


async def test_synthesise_passes_through_valid_finding():
    cands = [_candidate("id-1")]
    finding = Finding(
        answer="The wand chooses the wizard.",
        source_ids=["id-1"],
        conviction=5,
        gaps=[],
    )
    test_model = TestModel(custom_output_args=finding)

    agent = agents._synthesis_agent()
    with agent.override(model=test_model):
        out = await synthesise("Why does Ollivander say that?", cands)
    assert out.answer == "The wand chooses the wizard."
    assert out.conviction == 5
