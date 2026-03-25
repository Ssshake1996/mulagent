"""Tests for the tool system."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from tools.base import ToolDef
from tools.registry import ToolRegistry, get_default_tools, ALL_TOOLS
from tools.generation import is_dangerous_command


# ── Tool Registry ──

def test_default_tools_count():
    """All 17 tools should be registered."""
    registry = get_default_tools()
    assert len(registry.names()) == 17


def test_default_tool_names():
    registry = get_default_tools()
    names = set(registry.names())
    expected = {
        "web_search", "knowledge_recall", "web_fetch", "read_file",
        "list_dir", "execute_shell", "code_run", "write_file",
        "edit_file", "delegate", "deep_research", "docs_lookup",
        "codemap", "browser_fetch", "sql_query", "git_ops", "github_ops",
    }
    assert names == expected


def test_tool_openai_schema():
    registry = get_default_tools()
    schemas = registry.to_openai_tools()
    assert len(schemas) == 17
    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "description" in schema["function"]
        assert "parameters" in schema["function"]


def test_tool_descriptions_text():
    registry = get_default_tools()
    text = registry.tool_descriptions_text()
    assert "web_search" in text
    assert "delegate" in text


def test_registry_get():
    registry = get_default_tools()
    tool = registry.get("web_search")
    assert tool is not None
    assert tool.name == "web_search"
    assert registry.get("nonexistent") is None


# ── Shell Safety ──

def test_dangerous_command_detection():
    assert is_dangerous_command("rm -rf /") is True
    assert is_dangerous_command("sudo apt install foo") is True
    assert is_dangerous_command("dd if=/dev/zero of=/dev/sda") is True
    assert is_dangerous_command("mkfs.ext4 /dev/sda1") is True
    assert is_dangerous_command("shutdown -h now") is True
    assert is_dangerous_command("reboot") is True
    assert is_dangerous_command("chmod -R 777 /") is True


def test_safe_command_allowed():
    assert is_dangerous_command("ls -la") is False
    assert is_dangerous_command("echo hello") is False
    assert is_dangerous_command("python3 script.py") is False
    assert is_dangerous_command("git status") is False
    assert is_dangerous_command("cat /etc/hosts") is False


def test_pipe_command_with_quoted_code_not_blocked():
    """curl | python3 -c '...' should NOT be blocked by dict literal in quotes."""
    cmd = '''curl -s "https://api.example.com/data" | python3 -c "import json; d = {'key': 'value'}"'''
    assert is_dangerous_command(cmd) is False


def test_dangerous_command_still_detected_outside_quotes():
    """Actual dangerous commands should still be caught even with quotes present."""
    assert is_dangerous_command('echo "hello" && sudo rm -rf /') is True
    assert is_dangerous_command("rm -rf / # 'safe'") is True


# ── Shell Execution ──

@pytest.mark.asyncio
async def test_execute_shell_basic():
    from tools.generation import _execute_shell
    result = await _execute_shell({"command": "echo hello"})
    assert "exit_code: 0" in result
    assert "hello" in result


@pytest.mark.asyncio
async def test_execute_shell_blocked():
    from tools.generation import _execute_shell
    result = await _execute_shell({"command": "sudo rm -rf /"})
    assert "BLOCKED" in result


@pytest.mark.asyncio
async def test_execute_shell_empty():
    from tools.generation import _execute_shell
    result = await _execute_shell({"command": ""})
    assert "Error" in result


# ── Code Run ──

@pytest.mark.asyncio
async def test_code_run_basic():
    from tools.generation import _code_run
    result = await _code_run({"code": "print(2 + 3)"})
    assert "exit_code: 0" in result
    assert "5" in result


@pytest.mark.asyncio
async def test_code_run_error():
    from tools.generation import _code_run
    result = await _code_run({"code": "raise ValueError('test')"})
    assert "exit_code: 1" in result or "ValueError" in result


# ── Read File ──

@pytest.mark.asyncio
async def test_read_file_basic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\n")

    from tools.injection import _read_file
    result = await _read_file({"path": str(f)})
    assert "line1" in result
    assert "line2" in result


@pytest.mark.asyncio
async def test_read_file_not_found():
    from tools.injection import _read_file
    result = await _read_file({"path": "/tmp/nonexistent_file_abc123.txt"})
    assert "not found" in result


# ── Write File ──

@pytest.mark.asyncio
async def test_write_file_basic(tmp_path):
    target = tmp_path / "output.txt"

    from tools.generation import _write_file
    result = await _write_file({"path": str(target), "content": "hello world"})
    assert "Written" in result
    assert target.read_text() == "hello world"


# ── Web Search (mocked) ──

@pytest.mark.asyncio
@patch("shutil.which", return_value=None)
async def test_web_search_no_mcporter_uses_fallback(mock_which):
    """When mcporter is unavailable, DuckDuckGo fallback should be used."""
    from tools.discovery import _web_search
    result = await _web_search({"query": "test"})
    # Should either return results via DuckDuckGo fallback or "No results"
    assert result is not None
    assert "Error" not in result[:20]


# ── Role & Knowledge Loading ──

def test_load_roles():
    from tools.isolation import _load_roles, reload_roles
    reload_roles()  # Clear cache first
    roles = _load_roles()
    assert len(roles) >= 12
    assert "planner" in roles
    assert "architect" in roles
    assert "code_reviewer" in roles
    assert "build_resolver" in roles
    assert "tdd_guide" in roles
    assert "security_auditor" in roles


def test_load_knowledge():
    from tools.isolation import _load_knowledge, _knowledge_cache
    _knowledge_cache.clear()
    content = _load_knowledge(["python"])
    assert "Security Patterns" in content
    assert "python" in _knowledge_cache  # Cached


def test_load_knowledge_multiple():
    from tools.isolation import _load_knowledge, _knowledge_cache
    _knowledge_cache.clear()
    content = _load_knowledge(["security", "code_review"])
    assert "OWASP" in content
    assert "Review Checklist" in content


def test_load_knowledge_missing():
    from tools.isolation import _load_knowledge, _knowledge_cache
    _knowledge_cache.clear()
    content = _load_knowledge(["nonexistent_knowledge_base"])
    assert content == ""


# ── Edit File ──

@pytest.mark.asyncio
async def test_edit_file_basic(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world\nsecond line\n")

    from tools.generation import _edit_file
    result = await _edit_file({
        "path": str(f),
        "old_text": "hello world",
        "new_text": "goodbye world",
    })
    assert "Edited" in result
    assert f.read_text() == "goodbye world\nsecond line\n"


@pytest.mark.asyncio
async def test_edit_file_not_found():
    from tools.generation import _edit_file
    result = await _edit_file({
        "path": "/tmp/nonexistent_edit_test.txt",
        "old_text": "x",
        "new_text": "y",
    })
    assert "not found" in result


@pytest.mark.asyncio
async def test_edit_file_old_text_not_found(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("hello world\n")

    from tools.generation import _edit_file
    result = await _edit_file({
        "path": str(f),
        "old_text": "nonexistent text",
        "new_text": "replacement",
    })
    assert "not found" in result


# ── List Dir ──

@pytest.mark.asyncio
async def test_list_dir_basic(tmp_path):
    (tmp_path / "file1.txt").write_text("a")
    (tmp_path / "file2.py").write_text("b")
    (tmp_path / "subdir").mkdir()

    from tools.injection import _list_dir
    result = await _list_dir({"path": str(tmp_path)})
    assert "subdir/" in result
    assert "file1.txt" in result
    assert "file2.py" in result


@pytest.mark.asyncio
async def test_list_dir_recursive(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "deep.py").write_text("c")

    from tools.injection import _list_dir
    result = await _list_dir({"path": str(tmp_path), "recursive": True})
    assert "deep.py" in result


# ── Codemap ──

@pytest.mark.asyncio
async def test_codemap_python(tmp_path):
    f = tmp_path / "sample.py"
    f.write_text(
        "class Foo:\n"
        "    def bar(self, x):\n"
        "        pass\n\n"
        "def baz(a, b):\n"
        "    return a + b\n"
    )

    from tools.codemap import _codemap
    result = await _codemap({"path": str(f)})
    assert "class Foo" in result
    assert "baz" in result


@pytest.mark.asyncio
async def test_codemap_directory(tmp_path):
    (tmp_path / "a.py").write_text("def hello(): pass\n")
    (tmp_path / "b.py").write_text("class World: pass\n")

    from tools.codemap import _codemap
    result = await _codemap({"path": str(tmp_path)})
    assert "hello" in result
    assert "World" in result
    assert "Files scanned: 2" in result


# ── Smart Knowledge Selection ──

def test_smart_knowledge_selection_python():
    from tools.isolation import _select_knowledge
    all_kbs = ["python", "typescript", "go", "java", "rust", "cpp", "kotlin", "flutter", "pytorch", "refactor"]
    selected = _select_knowledge(all_kbs, "Fix the Python Django view that returns 500 error")
    assert "python" in selected
    assert len(selected) <= 4


def test_smart_knowledge_selection_typescript():
    from tools.isolation import _select_knowledge
    all_kbs = ["python", "typescript", "go", "java", "rust", "cpp", "kotlin", "refactor"]
    selected = _select_knowledge(all_kbs, "React component with useEffect hook not working in Next.js")
    assert "typescript" in selected
    assert "python" not in selected


def test_smart_knowledge_selection_no_signal():
    from tools.isolation import _select_knowledge
    all_kbs = ["python", "typescript", "go", "java", "rust", "cpp", "refactor"]
    selected = _select_knowledge(all_kbs, "Fix the bug in the login page")
    assert len(selected) <= 4


# ── Multi-language Code Run ──

@pytest.mark.asyncio
async def test_code_run_javascript():
    """Test code_run with JavaScript if node is available."""
    import shutil
    if not shutil.which("node"):
        pytest.skip("node not available")

    from tools.generation import _code_run
    result = await _code_run({"code": "console.log(2 + 3)", "language": "javascript"})
    assert "5" in result


@pytest.mark.asyncio
async def test_code_run_unsupported_language():
    from tools.generation import _code_run
    result = await _code_run({"code": "print('hello')", "language": "cobol"})
    assert "unsupported language" in result.lower()


def test_role_has_knowledge_refs():
    from tools.isolation import _load_roles, reload_roles
    reload_roles()
    roles = _load_roles()
    # architect should reference architect knowledge
    assert "architect" in roles["architect"].get("knowledge", [])
    # code_reviewer should reference code_review knowledge
    assert "code_review" in roles["code_reviewer"].get("knowledge", [])
    # security_auditor should reference security knowledge
    assert "security" in roles["security_auditor"].get("knowledge", [])


# ── Tool Learning (UCB1) ──

def test_tool_learning_ucb1():
    from evolution.tool_learning import ToolLearner
    learner = ToolLearner(exploration_c=1.4)

    # Record some outcomes
    learner.record_outcome("web_search", {"query": "test"}, success=True, latency_s=2.0)
    learner.record_outcome("web_search", {"query": "test2"}, success=True, latency_s=1.5)
    learner.record_outcome("web_search", {"query": "test3"}, success=False, latency_s=3.0)
    learner.record_outcome("code_run", {"code": "print(1)"}, success=True, latency_s=0.5)

    # Recommend tools
    recs = learner.recommend_tools("search for info", ["web_search", "code_run", "delegate"])
    assert len(recs) == 3
    # delegate should have high exploration bonus (never used)
    delegate_rec = next(r for r in recs if r["tool"] == "delegate")
    assert delegate_rec["total_calls"] == 0
    assert delegate_rec["exploration_bonus"] > 1.0  # High exploration bonus


def test_tool_learning_anti_matthew():
    """UCB1 should give exploration bonus to underused tools."""
    from evolution.tool_learning import ToolLearner
    learner = ToolLearner(exploration_c=1.4)

    # Use tool_a heavily, tool_b never
    for _ in range(50):
        learner.record_outcome("tool_a", {}, success=True, latency_s=1.0)

    recs = learner.recommend_tools("any task", ["tool_a", "tool_b"])
    # tool_b should rank higher due to exploration bonus (anti-Matthew)
    assert recs[0]["tool"] == "tool_b"


def test_tool_learning_serialization():
    from evolution.tool_learning import ToolLearner
    learner = ToolLearner()
    learner.record_outcome("web_search", {"q": "test"}, success=True)
    learner.record_outcome("code_run", {"code": "1+1"}, success=False)

    data = learner.to_dict()
    restored = ToolLearner.from_dict(data)
    assert restored._total_trials == 2
    assert "web_search" in restored._stats
    assert "code_run" in restored._stats


def test_tool_learning_suggest_params():
    from evolution.tool_learning import ToolLearner
    learner = ToolLearner()

    # Record enough outcomes for a pattern
    for i in range(5):
        learner.record_outcome("web_search", {"query": "python docs"}, success=True)
    for i in range(5):
        learner.record_outcome("web_search", {"query": "bad query"}, success=False)

    suggestion = learner.suggest_params("web_search", {"query": "test"})
    assert suggestion is not None
    assert suggestion["success_rate"] > 0.5


# ── Knowledge RAG ──

def test_chunk_markdown():
    from tools.knowledge_rag import _chunk_markdown

    text = """## Section 1
This is the first section with some content about Python.

## Section 2
This is the second section about TypeScript.

### Subsection 2.1
More details here about React components.
"""
    chunks = _chunk_markdown(text)
    assert len(chunks) >= 2
    assert any("Python" in c for c in chunks)
    assert any("TypeScript" in c for c in chunks)


def test_chunk_markdown_oversized():
    from tools.knowledge_rag import _chunk_markdown
    # Create an oversized section
    text = "## Big Section\n" + "A" * 2000 + "\n\n" + "B" * 2000
    chunks = _chunk_markdown(text, max_chunk=1500)
    assert all(len(c) <= 1500 for c in chunks)


# ── SQL Query Safety ──

def test_sql_write_detection():
    from tools.sql_query import _is_write_query
    assert _is_write_query("INSERT INTO users VALUES (1, 'test')") is True
    assert _is_write_query("UPDATE users SET name='x'") is True
    assert _is_write_query("DELETE FROM users") is True
    assert _is_write_query("DROP TABLE users") is True
    assert _is_write_query("SELECT * FROM users") is False
    assert _is_write_query("SELECT count(*) FROM orders WHERE status = 'active'") is False


def test_sql_write_detection_with_comments():
    from tools.sql_query import _is_write_query
    # Write in comment should not trigger (comment stripped)
    assert _is_write_query("SELECT * FROM users -- DELETE") is False
    # Actual write after comment strip
    assert _is_write_query("SELECT 1; DROP TABLE users") is True


# ── Git Tools ──

@pytest.mark.asyncio
async def test_git_status():
    from tools.git_tools import _git_ops
    result = await _git_ops({"action": "status"})
    # Should return something (we're in a git repo)
    assert result is not None
    assert "Error" not in result[:20]


@pytest.mark.asyncio
async def test_git_log():
    from tools.git_tools import _git_ops
    result = await _git_ops({"action": "log", "count": 5})
    assert result is not None


@pytest.mark.asyncio
async def test_git_unknown_action():
    from tools.git_tools import _git_ops
    result = await _git_ops({"action": "nonsense"})
    assert "Unknown" in result


# ── Observability ──

def test_metrics_counter():
    from common.observability import MetricsRegistry
    m = MetricsRegistry()
    m.inc("test_counter")
    m.inc("test_counter")
    assert m.get_counter("test_counter") == 2.0


def test_metrics_histogram():
    from common.observability import MetricsRegistry
    m = MetricsRegistry()
    for v in [1.0, 2.0, 3.0, 4.0, 5.0]:
        m.observe("latency", v)
    stats = m.get_histogram_stats("latency")
    assert stats["count"] == 5
    assert stats["mean"] == 3.0


def test_metrics_prometheus_format():
    from common.observability import MetricsRegistry
    m = MetricsRegistry()
    m.inc("requests_total", tool="web_search")
    m.inc("requests_total", tool="web_search")
    text = m.to_prometheus()
    assert "requests_total" in text
    assert "2" in text


def test_tracer_span():
    from common.observability import Tracer
    t = Tracer()
    with t.span("test_op") as s:
        s.attributes["custom"] = "value"
    traces = t.get_recent_traces()
    assert len(traces) == 1
    assert traces[0]["operation"] == "test_op"
    assert traces[0]["status"] == "ok"
    assert traces[0]["duration_ms"] >= 0


def test_alert_rule():
    from common.observability import MetricsRegistry, AlertRule
    m = MetricsRegistry()
    rule = AlertRule(metric_name="errors", threshold=5, severity="critical", message="too many errors")
    m.add_alert_rule(rule)
    for _ in range(4):
        m.inc("errors")
    assert not rule.fired
    m.inc("errors")  # Now total = 5
    assert rule.fired


# ── Checkpoint ──

def test_checkpoint_serialize_memory():
    from graph.checkpoint import _serialize_memory, _deserialize_memory
    from graph.memory import WorkingMemory

    memory = WorkingMemory()
    memory.add_directive("no delete")
    memory.update_state("round", 3)
    memory.add_fact("web_search", "found results", 1)

    data = _serialize_memory(memory)
    restored = _deserialize_memory(data)

    assert "no delete" in restored.directives
    assert restored.state["round"] == 3
    assert len(restored.facts) == 1
    assert restored.facts[0].source == "web_search"


def test_checkpoint_resume_context():
    from graph.checkpoint import build_resume_context

    checkpoint = {
        "user_input": "分析一下市场数据",
        "round_num": 5,
        "strategies_tried": [
            {"tool": "web_search", "args_summary": "query=市场", "outcome": "ok"},
            {"tool": "code_run", "args_summary": "code=...", "outcome": "fail"},
        ],
        "memory": {
            "directives": [],
            "state": {},
            "facts": [
                {"source": "web_search", "content": "GDP增长5%", "round_num": 2},
            ],
        },
    }
    context = build_resume_context(checkpoint)
    assert "resumed" in context.lower()
    assert "round 5" in context
    assert "web_search" in context


# ── Conversation Enhancement ──

def test_conversation_entity_extraction(tmp_path):
    from graph.conversation import ConversationStore

    store = ConversationStore(data_dir=tmp_path)
    store.create("test_session", "user1")
    store.append_turn("test_session", "user", "我喜欢用Python写代码")
    store.append_turn("test_session", "user", "决定使用FastAPI框架")

    entities = store.extract_entities("test_session")
    assert "preferences" in entities
    assert "decisions" in entities


def test_conversation_user_isolation(tmp_path):
    from graph.conversation import ConversationStore

    store = ConversationStore(data_dir=tmp_path)
    store.create("session_a", "user1")
    store.create("session_b", "user2")
    store.append_turn("session_a", "user", "hello from user1")
    store.append_turn("session_b", "user", "hello from user2")

    # Each user's data is in their own directory
    conv1 = store.load("session_a", user_id="user1")
    conv2 = store.load("session_b", user_id="user2")
    assert conv1 is not None
    assert conv2 is not None
    assert conv1["user_id"] == "user1"
    assert conv2["user_id"] == "user2"


# ── Progressive Output ──

def test_auto_complete_detection():
    from graph.react_orchestrator import _should_auto_complete
    assert _should_auto_complete("请自行判断并完成") is True
    assert _should_auto_complete("自动完成所有任务") is True
    assert _should_auto_complete("帮我查一下天气") is False
    assert _should_auto_complete("just do it all") is True


# ── Analyst role has sql_query tool ──

def test_analyst_has_sql_query():
    from tools.isolation import _load_roles, reload_roles
    reload_roles()
    roles = _load_roles()
    assert "sql_query" in roles["analyst"].get("tools", [])


# ── Coder role has git_ops tool ──

def test_coder_has_git_ops():
    from tools.isolation import _load_roles, reload_roles
    reload_roles()
    roles = _load_roles()
    assert "git_ops" in roles["coder"].get("tools", [])


# ── Delegate Depth Control ──

@pytest.mark.asyncio
async def test_delegate_depth_0_allows_redelegate():
    """At depth 0, sub-agent should keep delegate tool (depth < MAX)."""
    from tools.isolation import _delegate, MAX_DELEGATE_DEPTH

    with patch("graph.react_orchestrator.react_loop", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = "Sub result"

        mock_llm = MagicMock()
        mock_delegate_tool = MagicMock()
        mock_other_tool = MagicMock()
        mock_tools = {"delegate": mock_delegate_tool, "web_search": mock_other_tool}

        result = await _delegate(
            {"task": "research something"},
            llm=mock_llm,
            tools=mock_tools,
            parent_directives=[],
            delegate_depth=0,
        )

        # Verify sub-agent was called with delegate still in tools
        call_kwargs = mock_react.call_args
        sub_tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert "delegate" in sub_tools
        # Verify depth was incremented in deps
        sub_deps = call_kwargs.kwargs.get("deps") or call_kwargs[1].get("deps")
        assert sub_deps["delegate_depth"] == 1


@pytest.mark.asyncio
async def test_delegate_at_max_depth_excludes_delegate():
    """At max depth, delegate tool should be excluded from sub-agent."""
    from tools.isolation import _delegate, MAX_DELEGATE_DEPTH

    with patch("graph.react_orchestrator.react_loop", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = "Deep result"

        mock_llm = MagicMock()
        mock_tools = {"delegate": MagicMock(), "web_search": MagicMock()}

        result = await _delegate(
            {"task": "deep task"},
            llm=mock_llm,
            tools=mock_tools,
            parent_directives=[],
            delegate_depth=MAX_DELEGATE_DEPTH - 1,
        )

        call_kwargs = mock_react.call_args
        sub_tools = call_kwargs.kwargs.get("tools") or call_kwargs[1].get("tools")
        assert "delegate" not in sub_tools
        assert "web_search" in sub_tools


@pytest.mark.asyncio
async def test_delegate_default_depth_is_zero():
    """When delegate_depth is not in deps, it should default to 0."""
    from tools.isolation import _delegate

    with patch("graph.react_orchestrator.react_loop", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = "Result"

        mock_llm = MagicMock()
        mock_tools = {"delegate": MagicMock(), "web_search": MagicMock()}

        # No delegate_depth in deps
        result = await _delegate(
            {"task": "test task"},
            llm=mock_llm,
            tools=mock_tools,
            parent_directives=[],
        )

        call_kwargs = mock_react.call_args
        sub_deps = call_kwargs.kwargs.get("deps") or call_kwargs[1].get("deps")
        assert sub_deps["delegate_depth"] == 1
