"""Tests for the orchestrator graph."""

import pytest
from graph.orchestrator import build_graph, execute_node, should_continue_executing
from agents.registry import load_registry, AgentRegistry, AgentMeta
from agents.adapter import AdapterFactory
from common.config import CONFIG_DIR


@pytest.mark.asyncio
async def test_execute_node_mock():
    """Execute node should work without registry (mock mode)."""
    state = {
        "subtasks": [
            {"id": "t1", "name": "test", "description": "do something",
             "agent_type": "code", "dependencies": [], "status": "pending"}
        ],
        "subtask_results": {},
        "current_subtask_index": 0,
    }
    result = await execute_node(state)
    assert "final_output" in result
    assert result["subtask_results"]["t1"] is not None


@pytest.mark.asyncio
async def test_execute_node_with_registry():
    """Execute node should select agent from registry."""
    registry = load_registry(CONFIG_DIR / "agents.yaml")
    factory = AdapterFactory()  # mock mode (no LLM)
    state = {
        "subtasks": [
            {"id": "t1", "name": "test", "description": "write a function",
             "agent_type": "code", "dependencies": [], "status": "pending"}
        ],
        "subtask_results": {},
        "current_subtask_index": 0,
    }
    result = await execute_node(state, registry=registry, adapter_factory=factory)
    assert "final_output" in result
    assert "Thinker" in result["subtask_results"]["t1"]


@pytest.mark.asyncio
async def test_execute_multi_subtask():
    """Execute should handle multi-step plans."""
    registry = load_registry(CONFIG_DIR / "agents.yaml")
    factory = AdapterFactory()
    state = {
        "subtasks": [
            {"id": "t1", "name": "research", "description": "find info",
             "agent_type": "research", "dependencies": [], "status": "pending"},
            {"id": "t2", "name": "write", "description": "write report",
             "agent_type": "writing", "dependencies": ["t1"], "status": "pending"},
        ],
        "subtask_results": {},
        "current_subtask_index": 0,
    }
    # First call: execute t1 (t2 depends on t1)
    result = await execute_node(state, registry=registry, adapter_factory=factory)
    assert "t1" in result["subtask_results"]

    # If t2 is still pending, run again
    if result.get("status") == "executing":
        result = await execute_node(result, registry=registry, adapter_factory=factory)
    assert "t2" in result["subtask_results"]
    assert "final_output" in result


def test_should_continue_executing():
    assert should_continue_executing({"status": "executing"}) == "execute"
    assert should_continue_executing({"status": "quality_check"}) == "quality_check"


@pytest.mark.asyncio
async def test_full_graph_mock():
    """Run the complete graph in mock mode (no LLM)."""
    graph = build_graph()
    compiled = graph.compile()

    initial_state = {
        "user_input": "write a Python function to sort a list",
        "session_id": "test-001",
    }

    result = await compiled.ainvoke(initial_state)
    assert result["intent"] == "code"
    assert result["final_output"] != ""
    assert result["quality_passed"] is True
    assert result["status"] == "completed"


@pytest.mark.asyncio
async def test_full_graph_with_registry():
    """Run graph with agent registry."""
    registry = load_registry(CONFIG_DIR / "agents.yaml")
    factory = AdapterFactory()
    graph = build_graph(registry=registry, adapter_factory=factory)
    compiled = graph.compile()

    result = await compiled.ainvoke({
        "user_input": "搜索最新的AI新闻",
        "session_id": "test-002",
    })
    assert result["intent"] == "research"
    assert "Retriever" in result["final_output"]
    assert result["status"] == "completed"
