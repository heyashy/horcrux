"""Conversational chat session — multi-turn state.

    session.py   ChatSession + Turn dataclasses. Pure data structures,
                 no I/O. The REPL in scripts/chat.py owns orchestration;
                 these primitives just track the conversation.

Public API:

    from horcrux.chat import ChatSession, Turn
"""

from horcrux.chat.session import ChatSession, Turn

__all__ = ["ChatSession", "Turn"]
