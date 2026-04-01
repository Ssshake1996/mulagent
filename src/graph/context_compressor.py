"""Three-dimensional intelligent context compression.

Dimension 1 — Semantic Role Classification:
    Classifies each turn as requirement/correction/error_attempt/
    final_result/intermediate/directive/question.

Dimension 2 — Topic-based Archiving with Recall:
    Detects topic boundaries, groups related turns, archives cold topics
    while keeping hot topics in context. Users can /recall any topic.

Dimension 3 — Relevance-driven Dynamic Compression:
    Computes per-topic relevance to the current query using keyword overlap,
    recall-intent detection, and time decay.  Four compression levels:
        Full (≥0.7)  |  Summary (0.3–0.7)  |  Title (0.1–0.3)  |  Hidden (<0.1)
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# Dimension 1 — Semantic Turn Classification
# ══════════════════════════════════════════════════════════════════

# Semantic types ordered roughly by "keep priority"
SEM_TYPES = (
    "requirement",      # user's original request or goal
    "correction",       # user correcting / redirecting assistant
    "directive",        # persistent rule ("以后都用中文回答")
    "final_result",     # confirmed successful output
    "question",         # user asking for info (non-task)
    "error_attempt",    # failed attempt or error output
    "intermediate",     # thinking / partial progress
)

# Keywords / patterns for classification (Chinese + English)
_CLS_RULES: list[tuple[str, list[str]]] = [
    ("requirement", [
        r"帮我", r"请", r"我想", r"我需要", r"我要", r"能不能", r"能否",
        r"实现", r"开发", r"创建", r"写一[个段篇]",
        r"please", r"I want", r"I need", r"help me", r"could you",
        r"implement", r"create", r"build", r"write",
    ]),
    ("correction", [
        r"不对", r"不是这[个样]", r"错了", r"应该是", r"改成", r"换[一个]",
        r"不要", r"别", r"重新", r"修改",
        r"no[,.]?\s*(not|wrong)", r"that's wrong", r"instead",
        r"change it", r"fix", r"redo",
    ]),
    ("directive", [
        r"以后", r"记住", r"永远", r"所有.*都要", r"每次",
        r"always", r"never", r"from now on", r"remember",
    ]),
    ("question", [
        r"什么是", r"为什么", r"怎么", r"如何", r"是什么",
        r"what is", r"why", r"how", r"explain", r"tell me about",
    ]),
    ("error_attempt", [
        r"error", r"traceback", r"exception", r"failed",
        r"报错", r"失败", r"错误", r"异常",
    ]),
    ("final_result", [
        r"完成", r"搞定", r"成功", r"已[经完]",
        r"done", r"success", r"completed", r"here'?s the (result|output)",
    ]),
]


class TurnClassifier:
    """Classify a single turn by its semantic role."""

    def classify(self, role: str, content: str, prev_role: str = "",
                 prev_sem: str = "") -> str:
        """Return one of SEM_TYPES for this turn."""
        text = content[:500].lower()

        if role == "user":
            return self._classify_user(text)

        # Assistant turns: infer from content + preceding user sem_type
        if prev_sem == "requirement":
            # Check if this looks like a final result
            if _match_any(text, _CLS_RULES_MAP.get("final_result", [])):
                return "final_result"
        if _match_any(text, _CLS_RULES_MAP.get("error_attempt", [])):
            return "error_attempt"
        if prev_sem == "error_attempt":
            return "error_attempt"

        # Long assistant content with code blocks → likely final result
        if "```" in content and len(content) > 300:
            return "final_result"

        # Short assistant reply → intermediate
        if len(content) < 100:
            return "intermediate"

        return "final_result"

    def _classify_user(self, text: str) -> str:
        """Classify a user turn."""
        # Check in priority order
        for sem_type in ("directive", "correction", "requirement", "question"):
            patterns = _CLS_RULES_MAP.get(sem_type, [])
            if _match_any(text, patterns):
                return sem_type
        return "requirement"  # default for user turns


# Build lookup map
_CLS_RULES_MAP: dict[str, list[str]] = {t: pats for t, pats in _CLS_RULES}


def _match_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text) for p in patterns)


# ══════════════════════════════════════════════════════════════════
# Dimension 2 — Topic Grouping & Archiving
# ══════════════════════════════════════════════════════════════════

@dataclass
class Topic:
    """A group of related turns forming a coherent topic."""
    id: str
    title: str = ""
    keywords: list[str] = field(default_factory=list)
    summary: str = ""
    requirement: str = ""          # first user requirement in the topic
    final_result_preview: str = "" # last final_result snippet
    lessons: str = ""              # extracted from error_attempt → correction chains
    turns: list[dict] = field(default_factory=list)
    status: str = "hot"            # hot | cold | recalled
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id, "title": self.title, "keywords": self.keywords,
            "summary": self.summary, "requirement": self.requirement,
            "final_result_preview": self.final_result_preview,
            "lessons": self.lessons,
            "turns": self.turns, "status": self.status,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Topic":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# Signals that a new topic is starting
_TOPIC_BOUNDARY_PATTERNS = [
    r"另外", r"还有", r"换个", r"接下来", r"然后",
    r"新[的]?问题", r"下一[个步]",
    r"also", r"next", r"another", r"moving on", r"new topic",
    r"by the way", r"btw",
]


class TopicGrouper:
    """Detect topic boundaries and group turns into Topics."""

    def __init__(self, max_gap_turns: int = 6):
        self.max_gap_turns = max_gap_turns

    def group(self, turns: list[dict]) -> list[Topic]:
        """Group classified turns into topics."""
        if not turns:
            return []

        topics: list[Topic] = []
        current_turns: list[dict] = []

        for i, turn in enumerate(turns):
            if current_turns and self._is_boundary(turn, current_turns, i):
                topics.append(self._finalize_topic(current_turns))
                current_turns = []
            current_turns.append(turn)

        if current_turns:
            topics.append(self._finalize_topic(current_turns))

        return topics

    def _is_boundary(self, turn: dict, current: list[dict], idx: int) -> bool:
        """Detect if this turn starts a new topic."""
        if turn["role"] != "user":
            return False

        text = turn.get("content", "")[:200].lower()

        # Explicit boundary signals
        if _match_any(text, _TOPIC_BOUNDARY_PATTERNS):
            return True

        # New requirement after a final_result in the current group
        if turn.get("sem_type") == "requirement":
            last_assistant = None
            for t in reversed(current):
                if t["role"] == "assistant":
                    last_assistant = t
                    break
            if last_assistant and last_assistant.get("sem_type") == "final_result":
                return True

        # Long gap between current topic and this turn (by index count)
        if len(current) >= self.max_gap_turns:
            return True

        return False

    def _finalize_topic(self, turns: list[dict]) -> Topic:
        """Build a Topic from a group of turns."""
        now = datetime.now(timezone.utc).isoformat()
        topic_id = hashlib.md5(
            f"{turns[0].get('ts', now)}_{turns[0].get('content', '')[:50]}".encode()
        ).hexdigest()[:12]

        requirement = ""
        final_result = ""
        error_lessons: list[str] = []

        for t in turns:
            sem = t.get("sem_type", "")
            if sem == "requirement" and not requirement:
                requirement = t["content"][:200]
            if sem == "final_result":
                final_result = t["content"][:200]
            if sem == "error_attempt":
                error_lessons.append(t["content"][:100])

        # Extract keywords from requirement
        keywords = _extract_keywords(requirement)
        title = requirement[:60] if requirement else turns[0].get("content", "")[:60]

        return Topic(
            id=topic_id,
            title=title,
            keywords=keywords,
            requirement=requirement,
            final_result_preview=final_result,
            lessons="; ".join(error_lessons[:3]) if error_lessons else "",
            turns=turns,
            status="hot",
            created_at=turns[0].get("ts", now),
            updated_at=turns[-1].get("ts", now),
        )


def _extract_keywords(text: str, max_kw: int = 8) -> list[str]:
    """Extract keywords from text using simple word frequency."""
    if not text:
        return []
    # Remove common stop words (CN + EN)
    stops = {"的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
             "这", "中", "大", "为", "上", "个", "来", "也", "到", "说", "要", "与",
             "the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
             "to", "for", "of", "and", "or", "it", "this", "that", "with",
             "as", "by", "from", "be", "have", "has", "had", "do", "does",
             "i", "you", "he", "she", "we", "they", "me", "my", "your"}
    # Split by non-word chars, keep Chinese and ASCII
    words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z_]\w{2,}', text.lower())
    filtered = [w for w in words if w not in stops]
    # Frequency-based selection
    freq: dict[str, int] = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in ranked[:max_kw]]


# ══════════════════════════════════════════════════════════════════
# Dimension 3 — Relevance-driven Dynamic Compression
# ══════════════════════════════════════════════════════════════════

# Compression levels
LEVEL_FULL = "full"         # ≥0.7 — include all turns
LEVEL_SUMMARY = "summary"   # 0.3–0.7 — title + requirement + result preview
LEVEL_TITLE = "title"       # 0.1–0.3 — just topic title
LEVEL_HIDDEN = "hidden"     # <0.1 — omit entirely


def _get_compress_cfg():
    """Load compression config, fallback to defaults."""
    try:
        from common.config import get_settings
        return get_settings().react.compress
    except Exception:
        return None


def compute_relevance(topic: Topic, query: str, now_ts: float | None = None) -> float:
    """Compute relevance score of a topic to the current query.

    Three signals (weights from config react.compress):
    1. Keyword overlap (Jaccard similarity)
    2. Recall intent detection
    3. Time decay
    """
    cfg = _get_compress_cfg()
    w_kw = cfg.weight_keyword if cfg else 0.5
    w_recall = cfg.weight_recall if cfg else 0.3
    w_decay = cfg.weight_decay if cfg else 0.2
    half_life = cfg.decay_half_life_hours if cfg else 24.0

    if not query:
        return 0.3  # neutral

    query_kw = set(_extract_keywords(query))
    topic_kw = set(topic.keywords)

    # 1. Keyword overlap (Jaccard)
    if query_kw and topic_kw:
        intersection = query_kw & topic_kw
        union = query_kw | topic_kw
        kw_score = len(intersection) / len(union) if union else 0.0
    else:
        # Fallback: substring match on title/requirement
        q_lower = query.lower()
        kw_score = 0.0
        for kw in topic_kw:
            if kw in q_lower:
                kw_score = max(kw_score, 0.5)

    # 2. Recall intent (user referencing past topics)
    recall_score = 0.0
    if detect_recall_intent(query):
        # Check if any topic keywords appear in the recall query
        q_lower = query.lower()
        for kw in topic_kw:
            if kw in q_lower:
                recall_score = 1.0
                break
        if recall_score == 0:
            recall_score = 0.3  # general recall intent but no keyword match

    # 3. Time decay
    now = now_ts or time.time()
    topic_time = _parse_ts(topic.updated_at) if topic.updated_at else now
    age_hours = max(0, (now - topic_time) / 3600)
    decay_score = math.exp(-0.693 * age_hours / half_life)

    relevance = w_kw * kw_score + w_recall * recall_score + w_decay * decay_score
    return min(1.0, relevance)


def relevance_to_level(score: float) -> str:
    """Map relevance score to compression level."""
    cfg = _get_compress_cfg()
    t_full = cfg.level_full if cfg else 0.7
    t_summary = cfg.level_summary if cfg else 0.3
    t_title = cfg.level_title if cfg else 0.1

    if score >= t_full:
        return LEVEL_FULL
    if score >= t_summary:
        return LEVEL_SUMMARY
    if score >= t_title:
        return LEVEL_TITLE
    return LEVEL_HIDDEN


_RECALL_PATTERNS = [
    r"之前", r"上次", r"刚才", r"前面", r"那个", r"回顾",
    r"earlier", r"before", r"previous", r"last time", r"recall",
    r"go back", r"what was", r"remember when",
]


def detect_recall_intent(query: str) -> bool:
    """Detect if the user is referencing a past topic."""
    text = query.lower()
    return _match_any(text, _RECALL_PATTERNS)


def _parse_ts(ts_str: str) -> float:
    """Parse ISO timestamp to epoch seconds."""
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.timestamp()
    except Exception:
        return time.time()


# ══════════════════════════════════════════════════════════════════
# SmartCompressor — Produces compressed output per compression level
# ══════════════════════════════════════════════════════════════════

class SmartCompressor:
    """Compress a topic according to its compression level."""

    def compress(self, topic: Topic, level: str) -> str:
        """Return compressed text representation of the topic."""
        if level == LEVEL_HIDDEN:
            return ""

        if level == LEVEL_TITLE:
            return f"[Topic: {topic.title}]"

        if level == LEVEL_SUMMARY:
            parts = [f"[Topic: {topic.title}]"]
            if topic.requirement:
                parts.append(f"  Requirement: {topic.requirement[:150]}")
            if topic.final_result_preview:
                parts.append(f"  Result: {topic.final_result_preview[:150]}")
            if topic.lessons:
                parts.append(f"  Lessons: {topic.lessons[:100]}")
            return "\n".join(parts)

        # LEVEL_FULL — include all turns but still compress intermediates
        lines = []
        for t in topic.turns:
            sem = t.get("sem_type", "")
            role_label = "User" if t["role"] == "user" else "Assistant"
            content = t["content"]

            if sem == "intermediate":
                # Collapse intermediate turns
                content = content[:80] + "..." if len(content) > 80 else content
            elif sem == "error_attempt":
                # Keep first 150 chars + lesson marker
                content = f"[Error] {content[:150]}..."
            elif role_label == "Assistant" and len(content) > 500:
                content = content[:500] + "..."

            lines.append(f"{role_label}: {content}")

        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
# ContextAssembler — Assemble final context within token budget
# ══════════════════════════════════════════════════════════════════

class ContextAssembler:
    """Assemble conversation context with dynamic compression.

    Integrates all three dimensions:
    1. Classify turns by semantic role
    2. Group into topics
    3. Compress by relevance to current query
    """

    def __init__(self, max_tokens: int = 0):
        """
        Args:
            max_tokens: Token budget for the assembled context.
                        0 = auto-compute from context_window * context_compress_ratio.
                        Fallback: explicit context_max_chars / 2 → default 2000 tokens.
        """
        self.classifier = TurnClassifier()
        self.grouper = TopicGrouper()
        self.compressor = SmartCompressor()
        if max_tokens <= 0:
            max_tokens = self._compute_budget_tokens()
        self.max_tokens = max_tokens

    @staticmethod
    def _compute_budget_tokens() -> int:
        """Compute context budget in tokens from model context_window.

        Priority:
        1. Explicit context_max_chars > 0 in config → convert to tokens (÷2, conservative)
        2. context_window * context_compress_ratio (direct token budget)
        3. Fallback: 2000 tokens

        Ratio is clamped to [0.01, 0.5] to prevent misconfiguration.
        """
        try:
            from common.config import get_settings
            settings = get_settings()
            cfg = settings.react.compress

            # Backward compat: explicit char budget → convert to tokens conservatively
            if cfg.context_max_chars > 0:
                # Chinese ~1.5 tokens/char, English ~0.25 tokens/char
                # Use 0.5 as safe middle ground (overestimates → won't overflow)
                return max(500, int(cfg.context_max_chars * 0.5))

            # Dynamic: context_window * ratio (direct token budget)
            model_cfg = settings.llm.get_model()
            if model_cfg:
                ctx_window = model_cfg.get_context_window()
                ratio = max(0.01, min(0.5, cfg.context_compress_ratio))  # clamp
                return max(500, int(ctx_window * ratio))
        except Exception:
            pass
        return 2000

    def classify_turns(self, turns: list[dict]) -> list[dict]:
        """Add sem_type to each turn in-place and return them."""
        prev_role = ""
        prev_sem = ""
        for turn in turns:
            if "sem_type" not in turn:
                sem = self.classifier.classify(
                    turn["role"], turn["content"], prev_role, prev_sem,
                )
                turn["sem_type"] = sem
            prev_role = turn["role"]
            prev_sem = turn["sem_type"]
        return turns

    def assemble(
        self,
        turns: list[dict],
        current_query: str = "",
        archived_topics: list[dict] | None = None,
        summary: str = "",
    ) -> str:
        """Assemble compressed context for the LLM prompt.

        Args:
            turns: Current hot turns (with optional sem_type).
            current_query: The user's latest query (for relevance scoring).
            archived_topics: Cold/recalled topics from archive.
            summary: Legacy summary string (used as fallback).

        Returns:
            Assembled context string within token budget.
        """
        from common.tokenizer import estimate_tokens

        # Step 1: Classify turns
        turns = self.classify_turns(turns)

        # Step 2: Group into topics
        hot_topics = self.grouper.group(turns)

        # Step 3: Merge with archived topics
        all_topics: list[tuple[Topic, str]] = []  # (topic, source)

        # Archived topics (cold/recalled)
        if archived_topics:
            for td in archived_topics:
                topic = Topic.from_dict(td)
                all_topics.append((topic, "archive"))

        # Hot topics from current turns
        for topic in hot_topics:
            all_topics.append((topic, "hot"))

        if not all_topics:
            if summary:
                return f"[Earlier summary: {summary}]"
            return ""

        # Step 4: Compute relevance and assign compression levels
        now = time.time()
        scored: list[tuple[Topic, str, float, str]] = []
        for topic, source in all_topics:
            if source == "hot" and topic == hot_topics[-1]:
                # Most recent topic always gets full treatment
                score = 1.0
            else:
                score = compute_relevance(topic, current_query, now)
            level = relevance_to_level(score)
            scored.append((topic, source, score, level))

        # Sort: highest relevance first (but keep most recent hot topic last)
        last_hot = scored[-1] if scored and scored[-1][1] == "hot" else None
        rest = scored[:-1] if last_hot else scored
        rest.sort(key=lambda x: -x[2])

        # Step 5: Assemble within token budget
        parts: list[str] = []
        token_budget = self.max_tokens
        reserve_for_last = 0

        # Reserve space for the last (most recent) topic
        if last_hot:
            last_text = self.compressor.compress(last_hot[0], last_hot[3])
            reserve_for_last = estimate_tokens(last_text) + 20
            token_budget -= reserve_for_last

        # Add legacy summary if present
        if summary:
            summary_text = f"[Earlier summary: {summary[:300]}]"
            summary_cost = estimate_tokens(summary_text)
            if summary_cost < token_budget:
                parts.append(summary_text)
                token_budget -= summary_cost

        # Add topics by relevance (degrade if over budget)
        for topic, source, score, level in rest:
            text = self.compressor.compress(topic, level)
            if not text:
                continue

            text_cost = estimate_tokens(text)
            if text_cost <= token_budget:
                parts.append(text)
                token_budget -= text_cost
            else:
                # Degrade compression level
                for degraded in (LEVEL_SUMMARY, LEVEL_TITLE):
                    if degraded == level:
                        continue
                    degraded_text = self.compressor.compress(topic, degraded)
                    if degraded_text:
                        degraded_cost = estimate_tokens(degraded_text)
                        if degraded_cost <= token_budget:
                            parts.append(degraded_text)
                            token_budget -= degraded_cost
                            break

        # Add the most recent topic last
        if last_hot:
            last_text = self.compressor.compress(last_hot[0], last_hot[3])
            parts.append(last_text)

        return "\n\n".join(parts)

    def auto_archive(self, turns: list[dict], archive_threshold: int = 20
                     ) -> tuple[list[dict], list[dict]]:
        """Archive old topics when turns exceed threshold.

        Returns:
            (remaining_hot_turns, newly_archived_topic_dicts)
        """
        if len(turns) < archive_threshold:
            return turns, []

        turns = self.classify_turns(turns)
        topics = self.grouper.group(turns)

        if len(topics) <= 1:
            return turns, []

        # Archive all topics except the most recent one
        to_archive = topics[:-1]
        hot_topic = topics[-1]

        archived_dicts = []
        for topic in to_archive:
            topic.status = "cold"
            # Generate summary for archived topic
            if not topic.summary:
                topic.summary = self._make_summary(topic)
            archived_dicts.append(topic.to_dict())

        return hot_topic.turns, archived_dicts

    def _make_summary(self, topic: Topic) -> str:
        """Generate a rule-based summary for a topic."""
        parts = []
        if topic.requirement:
            parts.append(f"Task: {topic.requirement[:100]}")
        if topic.final_result_preview:
            parts.append(f"Result: {topic.final_result_preview[:100]}")
        if topic.lessons:
            parts.append(f"Issues: {topic.lessons[:80]}")
        return "; ".join(parts) if parts else topic.title[:100]

    def recall_topic(self, archived_topics: list[dict], query: str
                     ) -> list[dict]:
        """Find and mark archived topics that match a recall query.

        Returns the updated archived_topics list with matched topics
        set to status='recalled'.
        """
        if not archived_topics or not query:
            return archived_topics

        for td in archived_topics:
            topic = Topic.from_dict(td)
            score = compute_relevance(topic, query)
            if score >= 0.3:
                td["status"] = "recalled"
                logger.info("Recalled topic: %s (score=%.2f)", td.get("title", "?")[:40], score)

        return archived_topics

    def list_topics(self, turns: list[dict], archived: list[dict] | None = None
                    ) -> list[dict]:
        """List all topics (hot + archived) with their status."""
        turns = self.classify_turns(turns)
        hot_topics = self.grouper.group(turns)

        result = []
        if archived:
            for td in archived:
                result.append({
                    "id": td.get("id", "?"),
                    "title": td.get("title", "")[:60],
                    "status": td.get("status", "cold"),
                    "turns_count": len(td.get("turns", [])),
                    "requirement": td.get("requirement", "")[:80],
                })

        for topic in hot_topics:
            result.append({
                "id": topic.id,
                "title": topic.title[:60],
                "status": "hot",
                "turns_count": len(topic.turns),
                "requirement": topic.requirement[:80],
            })

        return result
