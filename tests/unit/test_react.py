"""Tests for the ReAct orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graph.react_orchestrator import react_loop, _force_conclude_fallback, _brief_args
from graph.memory import WorkingMemory, Fact


# ── Helper tests ──

def test_brief_args():
    assert _brief_args({"query": "hello"}) == "query=hello"
    assert "..." in _brief_args({"code": "x" * 100})


def test_force_conclude_empty():
    mem = WorkingMemory()
    result = _force_conclude_fallback(mem, "test task")
    assert "抱歉" in result or "sorry" in result.lower()


def test_force_conclude_with_facts():
    mem = WorkingMemory()
    mem.add_fact("web_search", "Found 3 trending repos", 1)
    mem.add_fact("web_fetch", "Repo details here", 2)
    result = _force_conclude_fallback(mem, "test task")
    assert "web_search" in result
    assert "web_fetch" in result


# ── ReAct loop with mock LLM ──

@pytest.mark.asyncio
async def test_react_loop_direct_answer():
    """LLM that gives a direct answer without tool calls should return immediately."""
    mock_response = MagicMock()
    mock_response.content = "The answer is 42."
    mock_response.tool_calls = []

    mock_llm = MagicMock()
    mock_llm_with_tools = AsyncMock(return_value=mock_response)
    mock_llm.bind_tools = MagicMock(return_value=MagicMock(ainvoke=mock_llm_with_tools))

    result = await react_loop(
        user_input="what is 6 * 7?",
        tools={},
        llm=mock_llm,
        max_rounds=5,
        timeout=10,
    )
    assert result == "The answer is 42."


@pytest.mark.asyncio
async def test_react_loop_with_tool_call():
    """LLM calls a tool, then gives final answer."""
    # Round 1: LLM calls a tool
    tool_response = MagicMock()
    tool_response.content = ""
    tool_response.tool_calls = [{"name": "execute_shell", "args": {"command": "echo hi"}, "id": "call_1"}]

    # Round 2: LLM gives final answer
    final_response = MagicMock()
    final_response.content = "The command returned: hi"
    final_response.tool_calls = []

    call_count = 0
    async def mock_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return tool_response
        return final_response

    mock_llm = MagicMock()
    mock_llm_bound = MagicMock()
    mock_llm_bound.ainvoke = mock_ainvoke
    mock_llm.bind_tools = MagicMock(return_value=mock_llm_bound)

    # Mock tool
    from tools.base import ToolDef
    mock_tool = ToolDef(
        name="execute_shell",
        description="test",
        parameters={"type": "object", "properties": {}},
        fn=AsyncMock(return_value="exit_code: 0\nstdout:\nhi"),
    )

    result = await react_loop(
        user_input="run echo hi",
        tools={"execute_shell": mock_tool},
        llm=mock_llm,
        max_rounds=5,
        timeout=10,
        is_sub_agent=True,  # skip directive extraction
    )
    assert "hi" in result.lower() or "command" in result.lower()


@pytest.mark.asyncio
async def test_react_loop_max_rounds():
    """Should stop after max_rounds and force conclude."""
    # LLM always calls a tool, never gives final answer
    tool_response = MagicMock()
    tool_response.content = ""
    tool_response.tool_calls = [{"name": "web_search", "args": {"query": "test"}, "id": "call_1"}]

    # For _force_conclude_llm: LLM synthesizes an answer
    synthesis_response = MagicMock()
    synthesis_response.content = "综合结果：搜索未找到相关内容。"

    mock_llm = MagicMock()
    mock_llm_bound = MagicMock()
    mock_llm_bound.ainvoke = AsyncMock(return_value=tool_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm_bound)
    # Direct ainvoke (used by _force_conclude_llm)
    mock_llm.ainvoke = AsyncMock(return_value=synthesis_response)

    from tools.base import ToolDef
    mock_tool = ToolDef(
        name="web_search",
        description="test",
        parameters={"type": "object", "properties": {}},
        fn=AsyncMock(return_value="No results"),
    )

    result = await react_loop(
        user_input="search forever",
        tools={"web_search": mock_tool},
        llm=mock_llm,
        max_rounds=3,
        timeout=10,
        is_sub_agent=True,
    )
    # Should have some output (force concluded via LLM synthesis)
    assert len(result) > 0


@pytest.mark.asyncio
async def test_react_loop_timeout():
    """Should handle overall timeout gracefully."""
    import asyncio

    async def slow_tool(params, **deps):
        await asyncio.sleep(100)
        return "never reached"

    tool_response = MagicMock()
    tool_response.content = ""
    tool_response.tool_calls = [{"name": "slow_tool", "args": {}, "id": "call_1"}]

    # For _force_conclude_llm after timeout
    synthesis_response = MagicMock()
    synthesis_response.content = "任务超时，未能完成。"

    mock_llm = MagicMock()
    mock_llm_bound = MagicMock()
    mock_llm_bound.ainvoke = AsyncMock(return_value=tool_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm_bound)
    mock_llm.ainvoke = AsyncMock(return_value=synthesis_response)

    from tools.base import ToolDef
    mock_tool = ToolDef(
        name="slow_tool",
        description="test",
        parameters={"type": "object", "properties": {}},
        fn=slow_tool,
    )

    result = await react_loop(
        user_input="slow task",
        tools={"slow_tool": mock_tool},
        llm=mock_llm,
        max_rounds=3,
        timeout=2,  # Very short timeout
        tool_timeout=1,  # Short tool timeout too
        is_sub_agent=True,
    )
    # Should return something (force concluded due to timeout)
    assert len(result) > 0


# ── Integration: run_react without LLM ──

@pytest.mark.asyncio
async def test_run_react_no_llm():
    """When llm=None, should return error status."""
    from graph.orchestrator import run_react
    result = await run_react(user_input="write hello world", llm=None)
    assert result["status"] == "failed"
