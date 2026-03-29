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
        """Append a user or assistant turn to the conversation.

        Automatically archives old topics when turns exceed threshold.
        """
        conv = self.load(session_id)
        if conv is None:
            conv = self.create(session_id)
        now = datetime.now(timezone.utc).isoformat()
        conv["turns"].append({"role": role, "content": content, "ts": now})
        conv["updated_at"] = now

        # Smart auto-archive: when turns grow large, archive cold topics
        from graph.context_compressor import ContextAssembler
        assembler = ContextAssembler()
        try:
            from common.config import get_settings
            _archive_th = get_settings().react.compress.archive_threshold
        except Exception:
            _archive_th = 30
        remaining, newly_archived = assembler.auto_archive(
            conv["turns"], archive_threshold=_archive_th,
        )
        if newly_archived:
            archive = conv.get("archive", {"topics": []})
            archive["topics"].extend(newly_archived)
            conv["archive"] = archive
            conv["turns"] = remaining
            logger.info("Auto-archived %d topics for session %s",
                        len(newly_archived), session_id)

        # Hard cap: keep last 50 turns as safety net
        if len(conv["turns"]) > 50:
            conv["turns"] = conv["turns"][-50:]
        self._save(session_id, conv)

    # ── Context CRUD (/modify support) ──────────────────────────

    def list_turns(self, session_id: str) -> list[dict]:
        """Return all turns with their indices."""
        conv = self.load(session_id)
        if conv is None:
            return []
        return conv.get("turns", [])

    def delete_turn(self, session_id: str, index: int) -> bool:
        """Delete a turn by index. Returns True on success."""
        conv = self.load(session_id)
        if conv is None:
            return False
        turns = conv.get("turns", [])
        if index < 0 or index >= len(turns):
            return False
        turns.pop(index)
        conv["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(session_id, conv)
        return True

    def delete_turns_range(self, session_id: str, start: int, end: int) -> int:
        """Delete turns in [start, end) range. Returns count deleted."""
        conv = self.load(session_id)
        if conv is None:
            return 0
        turns = conv.get("turns", [])
        start = max(0, start)
        end = min(len(turns), end)
        if start >= end:
            return 0
        count = end - start
        conv["turns"] = turns[:start] + turns[end:]
        conv["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(session_id, conv)
        return count

    def edit_turn(self, session_id: str, index: int, new_content: str) -> bool:
        """Edit a turn's content by index. Returns True on success."""
        conv = self.load(session_id)
        if conv is None:
            return False
        turns = conv.get("turns", [])
        if index < 0 or index >= len(turns):
            return False
        turns[index]["content"] = new_content
        turns[index]["ts"] = datetime.now(timezone.utc).isoformat()
        conv["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(session_id, conv)
        return True

    def clear_turns(self, session_id: str) -> bool:
        """Clear all turns (keep session metadata). Returns True on success."""
        conv = self.load(session_id)
        if conv is None:
            return False
        conv["turns"] = []
        conv["summary"] = ""
        conv["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save(session_id, conv)
        return True

    def get_summary(self, session_id: str) -> str:
        """Get the stored conversation summary."""
        conv = self.load(session_id)
        if conv is None:
            return ""
        return conv.get("summary", "")

    # ── Persistent directives (cross-session) ────────────────────

    def _persistent_directives_path(self, user_id: str) -> Path:
        """Path to user-level persistent directives file."""
        return self._user_dir(user_id) / "_directives.json"

    def load_persistent_directives(self, user_id: str) -> list[str]:
        """Load user-level directives that persist across sessions."""
        path = self._persistent_directives_path(user_id)
        if not path.exists():
            return []
        try:
            data = json.loads(path.read_text())
            return data.get("directives", [])
        except Exception:
            return []

    def save_persistent_directives(self, user_id: str, directives: list[str]) -> None:
        """Save user-level persistent directives."""
        path = self._persistent_directives_path(user_id)
        path.write_text(json.dumps(
            {"directives": directives},
            ensure_ascii=False, indent=2,
        ))

    def add_persistent_directive(self, user_id: str, directive: str) -> bool:
        """Add a single directive to persistent store. Returns False if duplicate."""
        existing = self.load_persistent_directives(user_id)
        if directive in existing:
            return False
        existing.append(directive)
        self.save_persistent_directives(user_id, existing)
        return True

    def remove_persistent_directive(self, user_id: str, index: int) -> bool:
        """Remove a persistent directive by index."""
        existing = self.load_persistent_directives(user_id)
        if index < 0 or index >= len(existing):
            return False
        existing.pop(index)
        self.save_persistent_directives(user_id, existing)
        return True

    def get_all_directives(self, session_id: str, user_id: str = "") -> list[str]:
        """Get merged directives: persistent (user-level) + session-level."""
        persistent = self.load_persistent_directives(user_id) if user_id else []
        session_dirs = self.get_directives(session_id)
        # Merge, persistent first, deduplicate
        seen = set()
        merged = []
        for d in persistent + session_dirs:
            if d not in seen:
                merged.append(d)
                seen.add(d)
        return merged

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

    def get_history_for_prompt(
        self,
        session_id: str,
        current_query: str = "",
        max_turns: int = 10,
        max_chars: int = 0,
    ) -> str:
        """Build a conversation history string for the system prompt.

        Uses three-dimensional intelligent compression:
        1. Semantic role classification (requirement/correction/result/...)
        2. Topic-based archiving (cold/hot layer)
        3. Relevance-driven dynamic compression (full/summary/title/hidden)

        Args:
            session_id: The conversation session ID.
            current_query: Current user query for relevance scoring.
            max_turns: Max recent turns (fallback if compressor unavailable).
            max_chars: Character budget. 0 = auto (max_tokens * 0.5 * 4).

        Returns:
            Assembled context string optimized for LLM consumption.
        """
        conv = self.load(session_id)
        if conv is None or not conv.get("turns"):
            return ""

        # Auto-compute max_chars from config: max_tokens * 0.5 (token budget) * 4 (chars/token)
        if max_chars <= 0:
            max_chars = self._auto_max_chars()

        turns = conv["turns"]
        summary = conv.get("summary", "")
        archived_topics = conv.get("archive", {}).get("topics", [])

        from graph.context_compressor import ContextAssembler
        assembler = ContextAssembler(max_chars=max_chars)

        return assembler.assemble(
            turns=turns,
            current_query=current_query,
            archived_topics=archived_topics,
            summary=summary,
        )

    @staticmethod
    def _auto_max_chars(default: int = 8000) -> int:
        """Compute max_chars from config: max_tokens * 0.5 * 4."""
        try:
            from common.config import get_settings
            settings = get_settings()
            model_cfg = settings.llm.get_model()
            if model_cfg and model_cfg.max_tokens:
                # 50% of max_tokens for context, ~4 chars per token
                return int(model_cfg.max_tokens * 0.5 * 4)
        except Exception:
            pass
        return default

    def smart_compress(self, session_id: str) -> str:
        """Force smart compression: archive old topics and return summary."""
        conv = self.load(session_id)
        if conv is None:
            return "(no conversation)"

        from graph.context_compressor import ContextAssembler
        assembler = ContextAssembler()
        try:
            from common.config import get_settings
            _manual_th = get_settings().react.compress.archive_manual_threshold
        except Exception:
            _manual_th = 6

        remaining, newly_archived = assembler.auto_archive(
            conv["turns"], archive_threshold=_manual_th,  # lower threshold for manual compress
        )
        if newly_archived:
            archive = conv.get("archive", {"topics": []})
            archive["topics"].extend(newly_archived)
            conv["archive"] = archive
            conv["turns"] = remaining
            self._save(session_id, conv)
            return f"Archived {len(newly_archived)} topic(s), {len(remaining)} turns remain"
        return "Nothing to archive (conversation is short or single-topic)"

    def recall_topic(self, session_id: str, query: str) -> list[dict]:
        """Recall archived topics matching a query.

        Returns list of matching topic summaries.
        """
        conv = self.load(session_id)
        if conv is None:
            return []

        archive = conv.get("archive", {})
        topics = archive.get("topics", [])
        if not topics:
            return []

        from graph.context_compressor import ContextAssembler
        assembler = ContextAssembler()
        updated = assembler.recall_topic(topics, query)
        archive["topics"] = updated
        conv["archive"] = archive
        self._save(session_id, conv)

        # Return recalled topics
        return [t for t in updated if t.get("status") == "recalled"]

    def list_topics(self, session_id: str) -> list[dict]:
        """List all topics (hot + archived) with status."""
        conv = self.load(session_id)
        if conv is None:
            return []

        archive = conv.get("archive", {})
        archived_topics = archive.get("topics", [])

        from graph.context_compressor import ContextAssembler
        assembler = ContextAssembler()
        return assembler.list_topics(conv.get("turns", []), archived_topics)

    def expand_topic(self, session_id: str, topic_id: str) -> dict | None:
        """Expand an archived topic back to full detail.

        Marks the topic as 'recalled' so it will be included in context.
        """
        conv = self.load(session_id)
        if conv is None:
            return None

        archive = conv.get("archive", {})
        for topic in archive.get("topics", []):
            if topic.get("id") == topic_id:
                topic["status"] = "recalled"
                self._save(session_id, conv)
                return topic
        return None

    def collapse_topic(self, session_id: str, topic_id: str) -> bool:
        """Collapse a recalled topic back to cold state."""
        conv = self.load(session_id)
        if conv is None:
            return False

        archive = conv.get("archive", {})
        for topic in archive.get("topics", []):
            if topic.get("id") == topic_id:
                topic["status"] = "cold"
                self._save(session_id, conv)
                return True
        return False

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

