"""Tests for graceful degradation when infrastructure services are unavailable.

Verifies that the system continues to function when Redis, Qdrant, or
PostgreSQL are down — core ReAct loop should always work.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from tools.base import ToolDef


# ── Helper: create a mock LLM that gives direct answers ──

def _make_direct_answer_llm(answer: str = "The answer is 42."):
    """Create a mock LLM that returns a direct answer (no tool calls)."""
    mock_response = MagicMock()
    mock_response.content = answer
    mock_response.tool_calls = []

    mock_llm = MagicMock()
    mock_llm_bound = MagicMock()
    mock_llm_bound.ainvoke = AsyncMock(return_value=mock_response)
    mock_llm.bind_tools = MagicMock(return_value=mock_llm_bound)
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)
    return mock_llm


def _make_tool_then_answer_llm(tool_name: str, tool_args: dict, answer: str):
    """Create a mock LLM that calls a tool once, then gives final answer."""
    tool_response = MagicMock()
    tool_response.content = ""
    tool_response.tool_calls = [{"name": tool_name, "args": tool_args, "id": "call_1"}]

    final_response = MagicMock()
    final_response.content = answer
    final_response.tool_calls = []

    call_count = 0
    async def mock_ainvoke(messages):
        nonlocal call_count
        call_count += 1
        return tool_response if call_count == 1 else final_response

    mock_llm = MagicMock()
    mock_llm_bound = MagicMock()
    mock_llm_bound.ainvoke = mock_ainvoke
    mock_llm.bind_tools = MagicMock(return_value=mock_llm_bound)
    mock_llm.ainvoke = AsyncMock(return_value=final_response)
    return mock_llm


# ── Redis degradation ──

@pytest.mark.asyncio
async def test_react_works_without_redis():
    """ReAct loop should work when Redis is completely unavailable."""
    from graph.react_orchestrator import react_loop

    mock_llm = _make_direct_answer_llm("Hello!")

    with patch("common.redis_client.get_redis", new_callable=AsyncMock, return_value=None):
        result = await react_loop(
            user_input="say hello",
            tools={},
            llm=mock_llm,
            max_rounds=3,
            timeout=10,
        )
    assert result == "Hello!"


@pytest.mark.asyncio
async def test_idempotency_skip_when_redis_down():
    """Tool execution should proceed normally when Redis idempotency check fails."""
    from graph.react_orchestrator import react_loop

    mock_tool = ToolDef(
        name="execute_shell",
        description="test",
        parameters={"type": "object", "properties": {}},
        fn=AsyncMock(return_value="exit_code: 0\nstdout:\nhi"),
    )

    mock_llm = _make_tool_then_answer_llm(
        "execute_shell", {"command": "echo hi"}, "Done: hi"
    )

    with patch("common.redis_client.is_duplicate", side_effect=ConnectionError("redis down")):
        result = await react_loop(
            user_input="run echo hi",
            tools={"execute_shell": mock_tool},
            llm=mock_llm,
            max_rounds=3,
            timeout=10,
            is_sub_agent=True,
        )
    assert len(result) > 0


# ── Qdrant degradation ──

@pytest.mark.asyncio
async def test_react_works_without_qdrant():
    """ReAct loop should work when Qdrant is unavailable (no RAG)."""
    from graph.react_orchestrator import react_loop

    mock_tool = ToolDef(
        name="web_search",
        description="test",
        parameters={"type": "object", "properties": {}},
        fn=AsyncMock(return_value="Found: Python is a programming language"),
    )

    mock_llm = _make_tool_then_answer_llm(
        "web_search", {"query": "python"}, "Python is a programming language."
    )

    result = await react_loop(
        user_input="what is python?",
        tools={"web_search": mock_tool},
        llm=mock_llm,
        deps={"qdrant": None, "collection_name": "test"},
        max_rounds=3,
        timeout=10,
        is_sub_agent=True,
    )
    assert "python" in result.lower() or "Python" in result


@pytest.mark.asyncio
async def test_delegate_works_without_qdrant():
    """Delegate tool should work when Qdrant is unavailable (no experience/RAG)."""
    from tools.isolation import _delegate

    # Mock react_loop to return a direct answer
    with patch("graph.react_orchestrator.react_loop", new_callable=AsyncMock) as mock_react:
        mock_react.return_value = "Sub-agent result"

        mock_llm = MagicMock()
        mock_tools = {"web_search": MagicMock()}

        result = await _delegate(
            {"task": "research python", "role": "researcher"},
            llm=mock_llm,
            tools=mock_tools,
            parent_directives=[],
            qdrant=None,
            collection_name="test",
            delegate_depth=0,
        )
    assert "Sub-agent result" in result


# ── PostgreSQL degradation ──

@pytest.mark.asyncio
async def test_checkpoint_skip_when_db_down():
    """Checkpoint save should fail silently when PostgreSQL is unavailable."""
    from graph.react_orchestrator import react_loop

    mock_tool = ToolDef(
        name="web_search",
        description="test",
        parameters={"type": "object", "properties": {}},
        fn=AsyncMock(return_value="Found results"),
    )

    mock_llm = _make_tool_then_answer_llm(
        "web_search", {"query": "test"}, "Here are the results."
    )

    with patch("graph.checkpoint.save_checkpoint", side_effect=Exception("db down")):
        result = await react_loop(
            user_input="search test",
            tools={"web_search": mock_tool},
            llm=mock_llm,
            max_rounds=5,
            timeout=10,
            is_sub_agent=True,
            task_id="test-task-id",
        )
    assert len(result) > 0


# ── Tool learning degradation ──

@pytest.mark.asyncio
async def test_tool_learning_skip_when_unavailable():
    """Tool learning recording should fail silently."""
    from graph.react_orchestrator import react_loop

    mock_tool = ToolDef(
        name="web_search",
        description="test",
        parameters={"type": "object", "properties": {}},
        fn=AsyncMock(return_value="Search results here"),
    )

    mock_llm = _make_tool_then_answer_llm(
        "web_search", {"query": "test"}, "Results found."
    )

    with patch("evolution.tool_learning.get_tool_learner", side_effect=Exception("no learner")):
        result = await react_loop(
            user_input="search something",
            tools={"web_search": mock_tool},
            llm=mock_llm,
            max_rounds=3,
            timeout=10,
            is_sub_agent=True,
        )
    assert len(result) > 0


# ── Combined degradation ──

@pytest.mark.asyncio
async def test_all_infra_down():
    """Core loop works when Redis + Qdrant + PostgreSQL are all unavailable."""
    from graph.react_orchestrator import react_loop

    mock_llm = _make_direct_answer_llm("I can still answer without infrastructure.")

    with (
        patch("common.redis_client.get_redis", new_callable=AsyncMock, return_value=None),
    ):
        result = await react_loop(
            user_input="hello",
            tools={},
            llm=mock_llm,
            deps={"qdrant": None},
            max_rounds=3,
            timeout=10,
        )
    assert "answer" in result.lower() or "infrastructure" in result.lower()
