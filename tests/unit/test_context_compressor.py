"""Tests for three-dimensional intelligent context compression."""

import time
import tempfile
import pytest
from pathlib import Path
from datetime import datetime, timezone

from graph.context_compressor import (
    TurnClassifier,
    TopicGrouper,
    SmartCompressor,
    ContextAssembler,
    Topic,
    compute_relevance,
    detect_recall_intent,
    relevance_to_level,
    _extract_keywords,
    LEVEL_FULL,
    LEVEL_SUMMARY,
    LEVEL_TITLE,
    LEVEL_HIDDEN,
)


# ── TurnClassifier ──────────────────────────────────────────────

class TestTurnClassifier:
    def setup_method(self):
        self.cls = TurnClassifier()

    def test_classify_requirement_cn(self):
        assert self.cls.classify("user", "帮我写一段排序代码") == "requirement"

    def test_classify_requirement_en(self):
        assert self.cls.classify("user", "Please implement a sorting function") == "requirement"

    def test_classify_correction(self):
        assert self.cls.classify("user", "不对，应该是升序排列") == "correction"

    def test_classify_directive(self):
        assert self.cls.classify("user", "以后所有代码都用Python") == "directive"

    def test_classify_question(self):
        assert self.cls.classify("user", "什么是快速排序？") == "question"

    def test_classify_assistant_final_result(self):
        content = "```python\ndef sort(arr):\n    return sorted(arr)\n```\n这是排序函数的实现。" + "x" * 300
        assert self.cls.classify("assistant", content, prev_sem="requirement") == "final_result"

    def test_classify_assistant_error(self):
        assert self.cls.classify("assistant", "error: traceback in module X") == "error_attempt"

    def test_classify_assistant_short_intermediate(self):
        assert self.cls.classify("assistant", "OK, let me try.") == "intermediate"

    def test_default_user_is_requirement(self):
        assert self.cls.classify("user", "你好世界") == "requirement"


# ── TopicGrouper ────────────────────────────────────────────────

class TestTopicGrouper:
    def setup_method(self):
        self.grouper = TopicGrouper(max_gap_turns=6)

    def _make_turns(self, pairs):
        """Create turns from (role, content, sem_type) tuples."""
        turns = []
        ts = datetime.now(timezone.utc).isoformat()
        for role, content, sem in pairs:
            turns.append({"role": role, "content": content, "sem_type": sem, "ts": ts})
        return turns

    def test_single_topic(self):
        turns = self._make_turns([
            ("user", "帮我写排序", "requirement"),
            ("assistant", "sorted()", "final_result"),
        ])
        topics = self.grouper.group(turns)
        assert len(topics) == 1

    def test_explicit_boundary(self):
        turns = self._make_turns([
            ("user", "帮我写排序", "requirement"),
            ("assistant", "done", "final_result"),
            ("user", "另外，帮我写搜索", "requirement"),
            ("assistant", "search done", "final_result"),
        ])
        topics = self.grouper.group(turns)
        assert len(topics) == 2

    def test_boundary_after_final_result(self):
        turns = self._make_turns([
            ("user", "帮我写排序", "requirement"),
            ("assistant", "sorted()", "final_result"),
            ("user", "帮我写搜索", "requirement"),
            ("assistant", "search()", "final_result"),
        ])
        topics = self.grouper.group(turns)
        assert len(topics) == 2

    def test_empty(self):
        assert self.grouper.group([]) == []

    def test_max_gap_split(self):
        turns = self._make_turns(
            [("user", f"msg {i}", "requirement") if i % 2 == 0
             else ("assistant", f"reply {i}", "intermediate")
             for i in range(14)]
        )
        topics = self.grouper.group(turns)
        assert len(topics) >= 2


# ── Keyword Extraction ──────────────────────────────────────────

def test_extract_keywords_basic():
    kws = _extract_keywords("实现一个快速排序算法，使用Python语言")
    assert any("排序" in kw for kw in kws) or any("python" in kw for kw in kws)


def test_extract_keywords_empty():
    assert _extract_keywords("") == []


def test_extract_keywords_english():
    kws = _extract_keywords("implement a binary search tree in JavaScript")
    assert "binary" in kws or "search" in kws or "javascript" in kws


# ── Relevance Scoring ───────────────────────────────────────────

def test_compute_relevance_exact_match():
    topic = Topic(
        id="t1", title="sorting", keywords=["sorting", "algorithm", "python"],
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    score = compute_relevance(topic, "implement sorting algorithm")
    assert score > 0.3


def test_compute_relevance_no_match():
    topic = Topic(
        id="t1", title="sorting", keywords=["sorting", "algorithm"],
        updated_at=datetime.now(timezone.utc).isoformat(),
    )
    score = compute_relevance(topic, "weather forecast today")
    assert score < 0.5


def test_compute_relevance_time_decay():
    old_ts = "2024-01-01T00:00:00+00:00"
    new_ts = datetime.now(timezone.utc).isoformat()

    old_topic = Topic(id="t1", title="old", keywords=["test"], updated_at=old_ts)
    new_topic = Topic(id="t2", title="new", keywords=["test"], updated_at=new_ts)

    old_score = compute_relevance(old_topic, "test query")
    new_score = compute_relevance(new_topic, "test query")
    assert new_score > old_score


def test_relevance_to_level():
    assert relevance_to_level(0.8) == LEVEL_FULL
    assert relevance_to_level(0.5) == LEVEL_SUMMARY
    assert relevance_to_level(0.2) == LEVEL_TITLE
    assert relevance_to_level(0.05) == LEVEL_HIDDEN


# ── Recall Intent ───────────────────────────────────────────────

def test_detect_recall_cn():
    assert detect_recall_intent("之前那个排序怎么写的")
    assert detect_recall_intent("上次的结果是什么")


def test_detect_recall_en():
    assert detect_recall_intent("what was the previous result?")
    assert detect_recall_intent("go back to the earlier topic")


def test_no_recall_intent():
    assert not detect_recall_intent("帮我写一段新代码")
    assert not detect_recall_intent("implement a new feature")


# ── SmartCompressor ─────────────────────────────────────────────

class TestSmartCompressor:
    def setup_method(self):
        self.comp = SmartCompressor()

    def test_hidden(self):
        topic = Topic(id="t", title="test")
        assert self.comp.compress(topic, LEVEL_HIDDEN) == ""

    def test_title(self):
        topic = Topic(id="t", title="My Topic")
        result = self.comp.compress(topic, LEVEL_TITLE)
        assert "My Topic" in result

    def test_summary(self):
        topic = Topic(
            id="t", title="Sorting",
            requirement="implement quicksort",
            final_result_preview="def quicksort(arr): ...",
            lessons="off-by-one error",
        )
        result = self.comp.compress(topic, LEVEL_SUMMARY)
        assert "Sorting" in result
        assert "quicksort" in result
        assert "off-by-one" in result

    def test_full(self):
        topic = Topic(
            id="t", title="test",
            turns=[
                {"role": "user", "content": "write sort", "sem_type": "requirement"},
                {"role": "assistant", "content": "done: sorted()", "sem_type": "final_result"},
            ],
        )
        result = self.comp.compress(topic, LEVEL_FULL)
        assert "User: write sort" in result
        assert "sorted()" in result


# ── ContextAssembler ────────────────────────────────────────────

class TestContextAssembler:
    def setup_method(self):
        self.asm = ContextAssembler(max_chars=4000)

    def _make_turns(self, n=4):
        ts = datetime.now(timezone.utc).isoformat()
        turns = []
        for i in range(n):
            turns.append({"role": "user", "content": f"request {i}", "ts": ts})
            turns.append({"role": "assistant", "content": f"response {i}" + "x" * 200, "ts": ts})
        return turns

    def test_assemble_basic(self):
        turns = self._make_turns(2)
        result = self.asm.assemble(turns, current_query="request 1")
        assert "request" in result
        assert len(result) > 0

    def test_assemble_empty(self):
        assert self.asm.assemble([], current_query="test") == ""

    def test_assemble_with_summary(self):
        result = self.asm.assemble(
            turns=[{"role": "user", "content": "hello", "ts": "2026-01-01T00:00:00"}],
            summary="Earlier we discussed sorting",
        )
        assert "sorting" in result or "hello" in result

    def test_classify_turns(self):
        turns = [
            {"role": "user", "content": "帮我写排序代码", "ts": "2026-01-01T00:00:00"},
            {"role": "assistant", "content": "```python\ndef sort():\n    pass\n```" + "x" * 300, "ts": "2026-01-01T00:00:01"},
        ]
        classified = self.asm.classify_turns(turns)
        assert classified[0]["sem_type"] == "requirement"
        assert classified[1]["sem_type"] == "final_result"

    def test_auto_archive(self):
        turns = self._make_turns(20)
        remaining, archived = self.asm.auto_archive(turns, archive_threshold=10)
        assert len(remaining) < len(turns)
        assert len(archived) > 0
        assert all(t.get("status") == "cold" for t in archived)

    def test_auto_archive_short(self):
        turns = self._make_turns(2)
        remaining, archived = self.asm.auto_archive(turns, archive_threshold=10)
        assert remaining == turns
        assert archived == []

    def test_recall_topic(self):
        archived = [
            Topic(
                id="t1", title="sorting algorithm",
                keywords=["sorting", "algorithm", "quicksort"],
                status="cold",
                updated_at=datetime.now(timezone.utc).isoformat(),
            ).to_dict(),
        ]
        updated = self.asm.recall_topic(archived, "之前的sorting怎么做的")
        recalled = [t for t in updated if t["status"] == "recalled"]
        assert len(recalled) == 1

    def test_list_topics(self):
        ts = datetime.now(timezone.utc).isoformat()
        turns = [
            {"role": "user", "content": "帮我写排序", "ts": ts, "sem_type": "requirement"},
            {"role": "assistant", "content": "done", "ts": ts, "sem_type": "final_result"},
        ]
        archived = [
            {"id": "old1", "title": "old topic", "status": "cold", "turns": [], "requirement": "old req"},
        ]
        topics = self.asm.list_topics(turns, archived)
        assert len(topics) == 2
        statuses = {t["status"] for t in topics}
        assert "hot" in statuses
        assert "cold" in statuses

    def test_budget_enforcement(self):
        """Context should not exceed max_chars significantly."""
        asm = ContextAssembler(max_chars=500)
        turns = self._make_turns(10)
        result = asm.assemble(turns, current_query="test")
        # Allow some slack (headers, formatting)
        assert len(result) < 1500


# ── Integration with ConversationStore ──────────────────────────

def test_conversation_store_smart_compress():
    from graph.conversation import ConversationStore
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(data_dir=Path(tmp))
        store.create("s1", "u1")
        for i in range(15):
            store.append_turn("s1", "user", f"request {i}: 帮我实现功能{i}")
            store.append_turn("s1", "assistant", f"response {i}: " + "x" * 200)

        result = store.smart_compress("s1")
        assert "Archived" in result or "Nothing" in result


def test_conversation_store_list_topics():
    from graph.conversation import ConversationStore
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(data_dir=Path(tmp))
        store.create("s1", "u1")
        store.append_turn("s1", "user", "帮我写排序")
        store.append_turn("s1", "assistant", "done")

        topics = store.list_topics("s1")
        assert len(topics) >= 1
        assert topics[0]["status"] == "hot"


def test_conversation_store_get_history_with_query():
    from graph.conversation import ConversationStore
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(data_dir=Path(tmp))
        store.create("s1", "u1")
        store.append_turn("s1", "user", "帮我写排序代码")
        store.append_turn("s1", "assistant", "这是排序实现")

        history = store.get_history_for_prompt("s1", current_query="排序")
        assert len(history) > 0
        assert "排序" in history


def test_conversation_store_expand_collapse():
    from graph.conversation import ConversationStore
    with tempfile.TemporaryDirectory() as tmp:
        store = ConversationStore(data_dir=Path(tmp))
        store.create("s1", "u1")

        # Manually inject an archived topic
        conv = store.load("s1")
        conv["archive"] = {"topics": [
            {"id": "abc123", "title": "old topic", "status": "cold",
             "turns": [], "keywords": [], "requirement": "test",
             "summary": "", "final_result_preview": "", "lessons": "",
             "created_at": "", "updated_at": ""},
        ]}
        store._save("s1", conv)

        # Expand
        topic = store.expand_topic("s1", "abc123")
        assert topic is not None
        assert topic["status"] == "recalled"

        # Collapse
        assert store.collapse_topic("s1", "abc123")
        conv = store.load("s1")
        assert conv["archive"]["topics"][0]["status"] == "cold"
