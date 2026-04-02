"""Gateway adapter layer — shared session management and progress protocol.

Extracted from feishu_bot.py so that CLI, Desktop, and Feishu all share the
same session storage and conversation history.  Sessions are file-based (JSON)
via ConversationStore, so all surfaces read/write the same directory.
"""

from __future__ import annotations

import uuid
import logging
from dataclasses import dataclass
from typing import Any

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
        """Return current session_id for *user+context*, creating one if needed.

        On cache miss (e.g. after service restart), tries to recover the most
        recent existing session from disk before creating a brand-new one.
        This ensures conversational continuity across restarts.
        """
        key = self._key(user_id, context_id)
        if key not in self._sessions:
            # Try to recover from disk: find the latest session for this user
            recovered = self._recover_session(user_id, context_id)
            if recovered:
                self._sessions[key] = recovered
                logger.info("Recovered session %s for %s (from disk)", recovered, key)
            else:
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

    def _recover_session(self, user_id: str, context_id: str) -> str | None:
        """Try to find the most recent session file for this user+context on disk.

        Session IDs have the format: _{user_suffix}_{context_id}_{random}
        We look for files matching _{user_suffix}_{context_id}_*.json,
        pick the most recently modified one, and return its session_id.
        """
        import fnmatch
        safe_uid = user_id[-12:] if len(user_id) > 12 else user_id
        safe_uid = "".join(c if c.isalnum() or c in "-_" else "_" for c in safe_uid)
        safe_ctx = "".join(c if c.isalnum() or c in "-_" else "_" for c in context_id)
        pattern = f"_{safe_uid}_{safe_ctx}_*.json"

        try:
            # Search in user-specific dir first, then base dir
            search_dirs = []
            user_dir = self._conv_store._user_dir(user_id)
            if user_dir.exists():
                search_dirs.append(user_dir)
            search_dirs.append(self._conv_store.data_dir)

            best_path = None
            best_mtime = 0.0
            for d in search_dirs:
                for p in d.iterdir():
                    if p.is_file() and fnmatch.fnmatch(p.name, pattern):
                        mt = p.stat().st_mtime
                        if mt > best_mtime:
                            best_mtime = mt
                            best_path = p

            if best_path:
                import json
                conv = json.loads(best_path.read_text())
                sid = conv.get("session_id", "")
                if sid:
                    return sid
        except Exception as e:
            logger.debug("Session recovery failed for %s:%s: %s", user_id, context_id, e)
        return None

    @staticmethod
    def _key(user_id: str, context_id: str) -> str:
        return f"{user_id}:{context_id}"

    @staticmethod
    def _make_id(user_id: str, context_id: str) -> str:
        safe_uid = user_id[-12:] if len(user_id) > 12 else user_id
        return f"_{safe_uid}_{context_id}_{uuid.uuid4().hex[:6]}"
