"""Gateway adapter layer — shared session management and progress protocol.

Extracted from feishu_bot.py so that CLI, Desktop, and Feishu all share the
same session storage and conversation history.  Sessions are file-based (JSON)
via ConversationStore, so all surfaces read/write the same directory.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from graph.conversation import ConversationStore

logger = logging.getLogger(__name__)


# ── Progress protocol ─────────────────────────────────────────────

@dataclass
class ProgressEvent:
    """Unified progress event emitted by the ReAct loop."""
    round_num: int = 0
    action: str = ""       # "thinking" | "tool_call" | "tool_result" | "answer"
    detail: str = ""
    elapsed_ms: int = 0


class OutputAdapter(Protocol):
    """Any frontend (CLI, Feishu, Desktop) implements this to receive output."""

    async def send_progress(self, event: ProgressEvent) -> None: ...
    async def send_text(self, text: str) -> None: ...
    async def send_error(self, error: str) -> None: ...


# ── Session manager ───────────────────────────────────────────────

class SessionManager:
    """Manages user sessions across all gateway surfaces.

    Session keys are ``{user_id}:{context_id}`` where *context_id* is:
      - Feishu: chat_id
      - CLI: "cli"
      - Desktop: "desktop" or window ID

    The underlying ConversationStore writes JSON files to
    ``data/conversations/<user_id>/``, so sessions are automatically shared
    across surfaces that point at the same data directory.
    """

    def __init__(self, conv_store: ConversationStore | None = None):
        self._conv_store = conv_store or ConversationStore()
        # In-memory mapping: composite_key → session_id
        self._sessions: dict[str, str] = {}

    # ── Core operations ───────────────────────────────────────────

    def get_or_create(self, user_id: str, context_id: str) -> str:
        """Return current session_id for *user+context*, creating one if needed."""
        key = self._key(user_id, context_id)
        if key not in self._sessions:
            self._sessions[key] = self._make_id(user_id, context_id)
        return self._sessions[key]

    def new_session(self, user_id: str, context_id: str) -> str:
        """Force-create a fresh session and return its ID."""
        key = self._key(user_id, context_id)
        session_id = self._make_id(user_id, context_id)
        self._sessions[key] = session_id
        self._conv_store.create(session_id, user_id)
        return session_id

    def resume_session(self, user_id: str, context_id: str, session_id: str) -> bool:
        """Switch *user+context* to an existing *session_id*.

        Returns False if the session file does not exist.
        """
        if self._conv_store.load(session_id, user_id=user_id) is None:
            # Try without user_id (cross-surface resume)
            if self._conv_store.load(session_id) is None:
                return False
        key = self._key(user_id, context_id)
        self._sessions[key] = session_id
        return True

    def list_sessions(self, user_id: str, limit: int = 10) -> list[dict]:
        """List recent sessions for *user_id*."""
        return self._conv_store.list_sessions(user_id, limit=limit)

    def ensure_conversation(self, session_id: str, user_id: str) -> None:
        """Create conversation file if it doesn't exist yet."""
        if self._conv_store.load(session_id) is None:
            self._conv_store.create(session_id, user_id)

    # ── Convenience accessors ─────────────────────────────────────

    @property
    def conv_store(self) -> ConversationStore:
        return self._conv_store

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _key(user_id: str, context_id: str) -> str:
        return f"{user_id}:{context_id}"

    @staticmethod
    def _make_id(user_id: str, context_id: str) -> str:
        short_ctx = context_id[-8:] if len(context_id) > 8 else context_id
        return f"{context_id}_{user_id}_{short_ctx}_{uuid.uuid4().hex[:6]}"
