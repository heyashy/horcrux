"""ChatSession tests — pure data structure, no I/O."""

import uuid

import pytest

from horcrux.conversation import ChatSession, Turn
from horcrux.models import Finding, ScoredCandidate

pytestmark = pytest.mark.unit


def _candidate(cid: str = "id-a") -> ScoredCandidate:
    return ScoredCandidate(
        id=cid,
        score=0.5,
        source="paragraph",
        text=f"text-{cid}",
        book_num=1,
        chapter_num=1,
        chapter_title="t",
        page_start=1,
        characters=[],
    )


def _finding(answer: str = "x") -> Finding:
    return Finding(answer=answer, source_ids=["id-a"], conviction=4)


# ── Construction ────────────────────────────────────────────────


def test_default_thread_id_is_valid_uuid():
    session = ChatSession()
    # Round-trips through uuid.UUID without raising — definition of valid.
    uuid.UUID(session.thread_id)


def test_default_thread_id_is_unique_across_sessions():
    """Each session gets its own UUID by default."""
    a, b = ChatSession(), ChatSession()
    assert a.thread_id != b.thread_id


def test_explicit_thread_id_respected():
    """Tests can pin thread_id for reproducibility."""
    fixed = "00000000-0000-0000-0000-000000000000"
    session = ChatSession(thread_id=fixed)
    assert session.thread_id == fixed


def test_default_history_is_empty():
    assert len(ChatSession()) == 0


# ── record() ────────────────────────────────────────────────────


def test_record_appends_to_history():
    session = ChatSession()
    session.record("q", "q", _finding("first"), [_candidate()])
    session.record("q2", "q2", _finding("second"), [_candidate()])
    assert len(session) == 2
    assert session.history[0].finding.answer == "first"
    assert session.history[1].finding.answer == "second"


def test_record_returns_the_appended_turn():
    """record() returns the Turn so callers can chain rendering."""
    session = ChatSession()
    turn = session.record("q", "q-rewritten", _finding(), [_candidate()])
    assert isinstance(turn, Turn)
    assert turn.query == "q"
    assert turn.rewritten_query == "q-rewritten"


def test_record_preserves_original_vs_rewritten_query():
    """The session keeps both the user-typed query and the post-rewrite
    one — UIs need both for "did you mean" rendering."""
    session = ChatSession()
    turn = session.record(
        query="conjuncatvitus",
        rewritten_query="conjunctivitis",
        finding=_finding(),
        candidates=[_candidate()],
    )
    assert turn.query == "conjuncatvitus"
    assert turn.rewritten_query == "conjunctivitis"


# ── reset() ─────────────────────────────────────────────────────


def test_reset_clears_history():
    session = ChatSession()
    session.record("q", "q", _finding(), [_candidate()])
    session.record("q", "q", _finding(), [_candidate()])
    session.reset()
    assert len(session) == 0


def test_reset_keeps_thread_id():
    """reset() is a 'clear conversation' not 'new session' operation —
    thread_id stays so traces stay coherent across resets."""
    session = ChatSession()
    original_id = session.thread_id
    session.record("q", "q", _finding(), [_candidate()])
    session.reset()
    assert session.thread_id == original_id


# ── Turn payload ────────────────────────────────────────────────


def test_turn_holds_candidates_for_later_rendering():
    """The chat UI re-renders citations from past turns when the user
    types /history — needs the original ScoredCandidate list."""
    session = ChatSession()
    cands = [_candidate("a"), _candidate("b"), _candidate("c")]
    session.record("q", "q", _finding(), cands)
    assert len(session.history[0].candidates) == 3
