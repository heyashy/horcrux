"""Synthesis agent tests — schema invariants + runtime citation check.

Mocks the agent via `pydantic_ai.models.test.TestModel` so we never hit
the LiteLLM proxy. The full end-to-end path (real model + real Qdrant) is
covered separately as a smoke target.
"""

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

from horcrux.agents import synthesis as _synthesis_module
from horcrux.agents.synthesis import (
    _CHAPTER_SNIPPET_WORDS,
    _format_context,
    _truncate_for_synthesis,
    synthesise,
)
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


# ── _truncate_for_synthesis ──────────────────────────────────────


def test_truncate_paragraph_passes_through():
    """Paragraph chunks carry the evidence — never truncate them."""
    long_text = " ".join(["word"] * 10_000)
    cand = _candidate(text=long_text)  # default source = paragraph
    assert _truncate_for_synthesis(cand) == long_text


def test_truncate_chapter_caps_at_snippet_words():
    """Chapter chunks store the whole chapter (3-5k tokens). Cap them so a
    top-10 candidate set fits under the per-call rate limit."""
    long_text = " ".join(["word"] * 10_000)
    cand = ScoredCandidate(
        id="c",
        score=0.5,
        source="chapter",
        text=long_text,
        book_num=1,
        chapter_num=1,
        chapter_title="t",
        page_start=1,
        characters=[],
    )
    out = _truncate_for_synthesis(cand)
    out_words = out.split()
    # _CHAPTER_SNIPPET_WORDS body + 2-word continuation marker
    # ("[…chapter" + "continues…]").
    assert len(out_words) == _CHAPTER_SNIPPET_WORDS + 2
    assert out.endswith("[…chapter continues…]")


def test_truncate_chapter_short_passes_through():
    """A chapter chunk under the cap shouldn't get a truncation marker."""
    short = "this chapter is only a handful of words"
    cand = ScoredCandidate(
        id="c", score=0.5, source="chapter", text=short,
        book_num=1, chapter_num=1, chapter_title="t",
        page_start=1, characters=[],
    )
    assert _truncate_for_synthesis(cand) == short


# ── _format_context ──────────────────────────────────────────────


def test_format_context_does_not_leak_uuids():
    """UUIDs are never shown to the model — citations are by passage
    number to avoid transcription errors on long alphanumeric strings."""
    cands = [_candidate("id-1"), _candidate("id-2"), _candidate("id-3")]
    rendered = _format_context(cands)
    assert "id-1" not in rendered
    assert "id-2" not in rendered
    assert "id-3" not in rendered


def test_format_context_uses_numbered_brackets():
    cands = [_candidate("a"), _candidate("b"), _candidate("c")]
    rendered = _format_context(cands)
    assert "[1]" in rendered
    assert "[2]" in rendered
    assert "[3]" in rendered


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
    """The runtime layer: if the agent returns a passage number that's
    out of range, we raise rather than trust the citation."""
    cands = [_candidate("real-id")]
    fake_finding = Finding(
        answer="x",
        source_ids=["1", "99"],  # 99 is out of range for 1-candidate set
        conviction=4,
    )
    test_model = TestModel(custom_output_args=fake_finding)

    agent = _synthesis_module._synthesis_agent()
    with agent.override(model=test_model):  # noqa: SIM117
        with pytest.raises(ValueError, match="not in"):
            await synthesise("any question", cands)


async def test_synthesise_rejects_non_numeric_source_id():
    """The runtime layer: catches the model returning a UUID-like string
    instead of a passage number — exactly the failure mode that motivated
    switching to numbered citations."""
    cands = [_candidate("real-id")]
    fake_finding = Finding(
        answer="x",
        source_ids=["real-id"],  # the model accidentally returned the UUID
        conviction=4,
    )
    test_model = TestModel(custom_output_args=fake_finding)

    agent = _synthesis_module._synthesis_agent()
    with agent.override(model=test_model):  # noqa: SIM117
        with pytest.raises(ValueError, match="not in"):
            await synthesise("any question", cands)


async def test_synthesise_translates_numbers_to_real_ids():
    """Agent returns passage numbers; wrapper translates them back to
    actual chunk IDs so downstream consumers (renderer, history) get
    the canonical IDs they need."""
    cands = [_candidate("id-1"), _candidate("id-2"), _candidate("id-3")]
    finding = Finding(answer="x", source_ids=["2"], conviction=4)
    test_model = TestModel(custom_output_args=finding)

    agent = _synthesis_module._synthesis_agent()
    with agent.override(model=test_model):
        out = await synthesise("any question", cands)
    assert out.source_ids == ["id-2"]


async def test_synthesise_handles_bracketed_citation_form():
    """Sonnet usually returns bare digits but occasionally wraps them in
    brackets (e.g. '[1]'). The wrapper strips brackets before parsing."""
    cands = [_candidate("only-id")]
    finding = Finding(answer="x", source_ids=["[1]"], conviction=4)
    test_model = TestModel(custom_output_args=finding)

    agent = _synthesis_module._synthesis_agent()
    with agent.override(model=test_model):
        out = await synthesise("any question", cands)
    assert out.source_ids == ["only-id"]


async def test_synthesise_passes_through_valid_finding():
    cands = [_candidate("id-1")]
    finding = Finding(
        answer="The wand chooses the wizard.",
        source_ids=["1"],
        conviction=5,
        gaps=[],
    )
    test_model = TestModel(custom_output_args=finding)

    agent = _synthesis_module._synthesis_agent()
    with agent.override(model=test_model):
        out = await synthesise("Why does Ollivander say that?", cands)
    assert out.answer == "The wand chooses the wizard."
    assert out.source_ids == ["id-1"]
    assert out.conviction == 5
