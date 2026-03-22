"""Tests for the three-layer WorkingMemory system."""

import pytest

from graph.memory import (
    WorkingMemory,
    Fact,
    extract_directives_fast,
    compress_tool_result,
)


# ── WorkingMemory basics ──

def test_memory_empty():
    mem = WorkingMemory()
    assert mem.directives == []
    assert mem.state == {}
    assert mem.facts == []
    assert mem.build_context_message() == ""


def test_add_directive():
    mem = WorkingMemory()
    mem.add_directive("删除前需要确认")
    mem.add_directive("只处理30天前的邮件")
    assert len(mem.directives) == 2


def test_directive_dedup():
    mem = WorkingMemory()
    mem.add_directive("删除前需要确认")
    mem.add_directive("删除前需要确认")  # duplicate
    assert len(mem.directives) == 1


def test_update_state():
    mem = WorkingMemory()
    mem.update_state("processed", 10)
    mem.update_state("processed", 20)  # overwrite
    assert mem.state["processed"] == 20


def test_add_fact():
    mem = WorkingMemory()
    mem.add_fact("web_search", "Found 5 results", 1)
    mem.add_fact("execute_shell", "exit_code: 0", 2)
    assert len(mem.facts) == 2


# ── Context message building ──

def test_context_message_with_directives():
    mem = WorkingMemory()
    mem.add_directive("删除前需要确认")
    msg = mem.build_context_message()
    assert "RULES" in msg
    assert "删除前需要确认" in msg


def test_context_message_with_state():
    mem = WorkingMemory()
    mem.update_state("count", 42)
    msg = mem.build_context_message()
    assert "Current Progress" in msg
    assert "42" in msg


def test_context_message_with_facts():
    mem = WorkingMemory()
    mem.add_fact("web_search", "Found trending repos", 1)
    msg = mem.build_context_message()
    assert "Gathered Information" in msg
    assert "web_search" in msg


def test_context_message_directives_always_first():
    """Directives must always appear before state and facts."""
    mem = WorkingMemory()
    mem.add_fact("tool", "fact content", 1)
    mem.update_state("x", 1)
    mem.add_directive("important rule")
    msg = mem.build_context_message()

    rules_pos = msg.index("RULES")
    progress_pos = msg.index("Current Progress")
    info_pos = msg.index("Gathered Information")
    assert rules_pos < progress_pos < info_pos


# ── Fact compaction ──

def test_compact_facts_noop_when_few():
    mem = WorkingMemory()
    for i in range(3):
        mem.add_fact("tool", f"fact {i}", i)
    mem.compact_facts(keep_recent=5)
    assert len(mem.facts) == 3  # no change


def test_compact_facts_merges_old():
    mem = WorkingMemory()
    for i in range(20):
        mem.add_fact("web_search", f"search result {i}", i)

    mem.compact_facts(keep_recent=5)

    # Should have merged old + kept recent 5
    assert len(mem.facts) <= 6  # 1 merged + 5 recent
    # Recent facts preserved
    assert mem.facts[-1].content == "search result 19"


def test_compact_facts_preserves_directives():
    """Compaction must NEVER touch directives."""
    mem = WorkingMemory()
    mem.add_directive("do not delete without asking")
    for i in range(20):
        mem.add_fact("tool", f"fact {i}", i)

    mem.compact_facts(keep_recent=5)
    assert "do not delete without asking" in mem.directives
    assert len(mem.directives) == 1


# ── Directive Extraction (fast/rule-based) ──

def test_extract_approval_constraint():
    directives = extract_directives_fast("帮我清理邮箱，删除前要经过我同意")
    assert any("同意" in d for d in directives)


def test_extract_scope_constraint():
    directives = extract_directives_fast("只处理30天前的邮件，不要删除重要邮件")
    assert len(directives) >= 1


def test_extract_no_constraint():
    directives = extract_directives_fast("搜索一下GitHub热门项目")
    assert len(directives) == 0


def test_extract_multiple_constraints():
    text = "帮我清理邮箱。删除前先确认。只处理广告邮件。保留带附件的邮件。"
    directives = extract_directives_fast(text)
    assert len(directives) >= 2


def test_extract_ordering_constraint():
    directives = extract_directives_fast("先备份数据，再执行清理")
    assert len(directives) >= 1


# ── Tool result compression ──

def test_compress_short_result():
    result = "exit_code: 0\nhello"
    assert compress_tool_result(result, "execute_shell") == result


def test_compress_long_shell_output():
    long_output = "exit_code: 0\n" + "\n".join(f"line {i}" for i in range(200))
    compressed = compress_tool_result(long_output, "execute_shell", max_tokens=100)
    assert "exit_code: 0" in compressed
    assert "truncated" in compressed or "..." in compressed


def test_compress_long_search_result():
    long_result = "---".join(f"Result {i}: content here " * 20 for i in range(10))
    compressed = compress_tool_result(long_result, "web_search", max_tokens=100)
    assert len(compressed) < len(long_result)


# ── Pinned facts ──

def test_pinned_fact_survives_compaction():
    """Pinned facts must never be compacted."""
    mem = WorkingMemory()
    mem.pin_fact("critical", "this must survive", 0)
    for i in range(20):
        mem.add_fact("web_search", f"search result {i}", i + 1)

    mem.compact_facts(keep_recent=5)

    # Pinned fact must still be present
    pinned = [f for f in mem.facts if f.pinned]
    assert len(pinned) == 1
    assert pinned[0].content == "this must survive"


def test_pinned_fact_no_decay():
    """Pinned facts should not decay in relevance."""
    mem = WorkingMemory()
    mem.pin_fact("critical", "important", 0)
    for i in range(10):
        mem.add_fact("tool", f"fact {i}", i + 1)

    pinned = [f for f in mem.facts if f.pinned]
    assert pinned[0].relevance == 1.0  # Not decayed
