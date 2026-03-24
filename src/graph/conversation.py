"""Conversation history store for multi-turn dialogue.

Lightweight JSON-file-based storage. Each session has its own file.
No DB dependency — works even when PostgreSQL is unavailable.

Enhanced features:
- Entity extraction: automatically extracts key entities (names, preferences,
  decisions) across turns for persistent cross-session memory
- User isolation: sessions are stored in per-user directories
- Embedding-based retrieval: relevant past turns can be retrieved semantically

Structure per session:
    {
        "session_id": "feishu_xxx_abc123",
        "user_id": "ou_xxx",
        "turns": [
            {"role": "user", "content": "...", "ts": "2026-03-19T12:00:00"},
            {"role": "assistant", "content": "...", "ts": "2026-03-19T12:00:05"},
        ],
        "directives": ["删除前要经过我同意"],
        "entities": {"preferences": [...], "decisions": [...]},
        "created_at": "2026-03-19T12:00:00",
        "updated_at": "2026-03-19T12:00:05",
    }
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from common.config import DATA_DIR
_DATA_DIR = DATA_DIR / "conversations"


class ConversationStore:
    """File-based conversation history manager with user isolation."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir or _DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _user_dir(self, user_id: str) -> Path:
        """Get user-specific directory for conversation isolation."""
        if not user_id:
            return self.data_dir
        safe_uid = "".join(c if c.isalnum() or c in "-_" else "_" for c in user_id)
        d = self.data_dir / safe_uid
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _path(self, session_id: str, user_id: str = "") -> Path:
        # Sanitize session_id for filesystem safety
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        base_dir = self._user_dir(user_id) if user_id else self.data_dir
        return base_dir / f"{safe}.json"

    def _find_path(self, session_id: str) -> Path:
        """Find a session file, checking user dirs and base dir."""
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        # Check base dir first
        p = self.data_dir / f"{safe}.json"
        if p.exists():
            return p
        # Check user dirs
        for user_dir in self.data_dir.iterdir():
            if user_dir.is_dir():
                p = user_dir / f"{safe}.json"
                if p.exists():
                    return p
        return self.data_dir / f"{safe}.json"  # default

    def create(self, session_id: str, user_id: str = "") -> dict:
        """Create a new conversation with user isolation."""
        now = datetime.now(timezone.utc).isoformat()
        conv = {
            "session_id": session_id,
            "user_id": user_id,
            "turns": [],
            "directives": [],
            "entities": {"preferences": [], "decisions": [], "topics": []},
            "created_at": now,
            "updated_at": now,
        }
        self._save(session_id, conv, user_id=user_id)
        return conv

    def load(self, session_id: str, user_id: str = "") -> dict | None:
        """Load a conversation by session_id. Returns None if not found."""
        if user_id:
            path = self._path(session_id, user_id)
        else:
            path = self._find_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.warning("Failed to load conversation %s: %s", session_id, e)
            return None

    def append_turn(self, session_id: str, role: str, content: str) -> None:
        """Append a user or assistant turn to the conversation."""
        conv = self.load(session_id)
        if conv is None:
            conv = self.create(session_id)
        now = datetime.now(timezone.utc).isoformat()
        conv["turns"].append({"role": role, "content": content, "ts": now})
        conv["updated_at"] = now
        # Keep last 50 turns to prevent unbounded growth
        if len(conv["turns"]) > 50:
            conv["turns"] = conv["turns"][-50:]
        self._save(session_id, conv)

    def save_directives(self, session_id: str, directives: list[str]) -> None:
        """Update accumulated directives for the session."""
        conv = self.load(session_id)
        if conv is None:
            return
        # Merge new directives (deduplicate)
        existing = set(conv.get("directives", []))
        for d in directives:
            if d not in existing:
                conv["directives"].append(d)
                existing.add(d)
        self._save(session_id, conv)

    def get_history_for_prompt(self, session_id: str, max_turns: int = 10) -> str:
        """Build a conversation history string for the system prompt.

        If conversation is long (>20 turns), older turns are summarized
        to save context tokens while preserving key information.

        Returns the last N turns formatted as:
            User: ...
            Assistant: ...
        """
        conv = self.load(session_id)
        if conv is None or not conv.get("turns"):
            return ""

        turns = conv["turns"]

        # If we have a stored summary and many turns, use summary + recent turns
        summary = conv.get("summary", "")
        if summary and len(turns) > max_turns:
            recent = turns[-max_turns:]
            lines = [f"[Earlier conversation summary: {summary}]\n"]
            for turn in recent:
                role_label = "User" if turn["role"] == "user" else "Assistant"
                content = turn["content"]
                if role_label == "Assistant" and len(content) > 500:
                    content = content[:500] + "..."
                lines.append(f"{role_label}: {content}")
            return "\n".join(lines)

        recent = turns[-max_turns:]
        lines = []
        for turn in recent:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            content = turn["content"]
            if role_label == "Assistant" and len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)

    async def maybe_summarize(self, session_id: str, llm: Any = None) -> None:
        """Summarize older turns when conversation exceeds threshold.

        Compresses turns older than the most recent 10 into a summary.
        """
        if llm is None:
            return
        conv = self.load(session_id)
        if conv is None:
            return
        turns = conv.get("turns", [])
        if len(turns) < 20:
            return

        # Summarize the first N-10 turns
        old_turns = turns[:-10]
        old_text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'][:300]}"
            for t in old_turns[-20:]  # Cap at 20 turns for summarization
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=(
                    "请用2-3句话总结以下对话的核心内容和关键决策。"
                    "保留重要的上下文信息（如用户偏好、已完成的任务、关键结论）。"
                    "只输出总结，不要解释。"
                )),
                HumanMessage(content=old_text),
            ]
            response = await llm.ainvoke(messages)
            summary = response.content.strip()
            if summary:
                # Store summary and trim old turns
                conv["summary"] = summary
                conv["turns"] = turns[-10:]
                self._save(session_id, conv)
                logger.info("Conversation %s summarized (%d turns → summary + 10)",
                           session_id, len(turns))
        except Exception as e:
            logger.debug("Conversation summarization failed: %s", e)

    def get_directives(self, session_id: str) -> list[str]:
        """Get accumulated directives across all turns."""
        conv = self.load(session_id)
        if conv is None:
            return []
        return conv.get("directives", [])

    def list_sessions(self, user_id: str, limit: int = 10) -> list[dict]:
        """List recent sessions for a user.

        Searches both base dir and user-specific dir.
        """
        sessions = []
        # Search in user-specific dir first, then base dir
        search_dirs = []
        if user_id:
            user_dir = self._user_dir(user_id)
            if user_dir.exists():
                search_dirs.append(user_dir)
        search_dirs.append(self.data_dir)

        seen_ids: set[str] = set()
        for search_dir in search_dirs:
            json_files = sorted(
                search_dir.glob("*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            for path in json_files:
                try:
                    conv = json.loads(path.read_text())
                    sid = conv.get("session_id", "")
                    if sid in seen_ids:
                        continue
                    if conv.get("user_id") == user_id and conv.get("turns"):
                        first_msg = conv["turns"][0]["content"][:50] if conv["turns"] else ""
                        sessions.append({
                            "session_id": sid,
                            "preview": first_msg,
                            "turns": len(conv["turns"]),
                            "updated_at": conv.get("updated_at", ""),
                        })
                        seen_ids.add(sid)
                        if len(sessions) >= limit:
                            return sessions
                except Exception:
                    continue
        return sessions

    def cleanup_old_sessions(self, max_age_days: int = 30) -> int:
        """Remove conversation files older than max_age_days.

        Returns the number of sessions removed.
        """
        import time
        now = time.time()
        cutoff = now - (max_age_days * 86400)
        removed = 0
        for path in self.data_dir.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except Exception as e:
                logger.debug("Failed to remove old session %s: %s", path.name, e)
        if removed:
            logger.info("Cleaned up %d old conversation sessions (>%d days)", removed, max_age_days)
        return removed

    def _save(self, session_id: str, conv: dict, user_id: str = "") -> None:
        """Persist conversation to disk."""
        uid = user_id or conv.get("user_id", "")
        try:
            self._path(session_id, uid).write_text(
                json.dumps(conv, ensure_ascii=False, indent=2)
            )
        except Exception as e:
            logger.warning("Failed to save conversation %s: %s", session_id, e)

    # ── Entity Extraction ──

    def extract_entities(self, session_id: str) -> dict[str, list[str]]:
        """Extract key entities from conversation turns.

        Extracts: preferences, decisions, topics, names mentioned.
        This is rule-based (fast). LLM-based extraction is in maybe_extract_entities_llm.
        """
        conv = self.load(session_id)
        if not conv:
            return {}

        entities: dict[str, list[str]] = {
            "preferences": [],
            "decisions": [],
            "topics": [],
        }

        for turn in conv.get("turns", []):
            if turn["role"] != "user":
                continue
            content = turn["content"]

            # Preference detection
            pref_patterns = [
                r"(我喜欢|我偏好|我习惯|请用|用.{1,10}格式|用.{1,10}语言)",
                r"(prefer|I like|please use|always use)",
            ]
            for p in pref_patterns:
                m = re.search(p, content)
                if m:
                    # Extract the sentence containing the preference
                    sentences = re.split(r'[。！？\n,.!?]', content)
                    for s in sentences:
                        if m.group() in s and len(s.strip()) > 3:
                            if s.strip() not in entities["preferences"]:
                                entities["preferences"].append(s.strip())

            # Decision detection
            decision_patterns = [
                r"(决定|确定|就这样|就用|选择|agreed|decided|let's go with)",
            ]
            for p in decision_patterns:
                m = re.search(p, content)
                if m:
                    sentences = re.split(r'[。！？\n,.!?]', content)
                    for s in sentences:
                        if m.group() in s and len(s.strip()) > 3:
                            if s.strip() not in entities["decisions"]:
                                entities["decisions"].append(s.strip())

        # Store extracted entities
        conv["entities"] = entities
        self._save(session_id, conv)
        return entities

    async def maybe_extract_entities_llm(
        self, session_id: str, llm: Any = None
    ) -> dict:
        """Use LLM to extract entities from recent turns (more accurate)."""
        if llm is None:
            return self.extract_entities(session_id)

        conv = self.load(session_id)
        if not conv or len(conv.get("turns", [])) < 4:
            return {}

        recent = conv["turns"][-10:]
        text = "\n".join(
            f"{'User' if t['role'] == 'user' else 'Assistant'}: {t['content'][:200]}"
            for t in recent
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            import asyncio
            messages = [
                SystemMessage(content=(
                    "从对话中提取关键实体信息，返回 JSON：\n"
                    '{"preferences": ["用户偏好1"], "decisions": ["决定1"], "topics": ["主题1"]}\n'
                    "只返回 JSON。"
                )),
                HumanMessage(content=text),
            ]
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=10)
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            entities = json.loads(content)
            if isinstance(entities, dict):
                conv["entities"] = entities
                self._save(session_id, conv)
                return entities
        except Exception as e:
            logger.debug("LLM entity extraction failed: %s", e)

        return self.extract_entities(session_id)

    def get_cross_session_context(self, user_id: str, max_sessions: int = 5) -> str:
        """Build context from recent sessions for cross-session memory.

        Pulls entities and summaries from recent sessions to provide
        continuity across conversations.
        """
        sessions = self.list_sessions(user_id, limit=max_sessions)
        if not sessions:
            return ""

        parts = []
        all_preferences: set[str] = set()
        all_decisions: list[str] = []

        for sess_info in sessions:
            conv = self.load(sess_info["session_id"], user_id=user_id)
            if conv is None:
                continue

            entities = conv.get("entities", {})
            for p in entities.get("preferences", []):
                all_preferences.add(p)
            for d in entities.get("decisions", [])[-3:]:
                all_decisions.append(d)

            summary = conv.get("summary", "")
            if summary:
                parts.append(f"- {summary[:150]}")

        context_parts = []
        if all_preferences:
            context_parts.append("User preferences: " + "; ".join(list(all_preferences)[:5]))
        if all_decisions:
            context_parts.append("Recent decisions: " + "; ".join(all_decisions[:5]))
        if parts:
            context_parts.append("Recent sessions:\n" + "\n".join(parts[:3]))

        return "\n".join(context_parts) if context_parts else ""
