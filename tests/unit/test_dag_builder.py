"""Tests for DAG builder."""

import pytest
from graph.dag_builder import plan_node, get_ready_subtasks, all_subtasks_done, _plan_simple


def test_plan_simple():
    subtasks = _plan_simple("write hello world", "code")
    assert len(subtasks) == 1
    assert subtasks[0]["id"] == "t1"
    assert subtasks[0]["agent_type"] == "code"


@pytest.mark.asyncio
async def test_plan_node_simple():
    state = {"user_input": "hello", "intent": "general", "complexity": "simple"}
    result = await plan_node(state, llm=None)
    assert "subtasks" in result
    assert len(result["subtasks"]) == 1
    assert result["status"] == "planning"


def test_get_ready_subtasks_no_deps():
    subtasks = [
        {"id": "t1", "status": "pending", "dependencies": []},
        {"id": "t2", "status": "pending", "dependencies": []},
    ]
    ready = get_ready_subtasks(subtasks, {})
    assert len(ready) == 2


def test_get_ready_subtasks_with_deps():
    subtasks = [
        {"id": "t1", "status": "pending", "dependencies": []},
        {"id": "t2", "status": "pending", "dependencies": ["t1"]},
    ]
    # t1 not done yet
    ready = get_ready_subtasks(subtasks, {})
    assert len(ready) == 1
    assert ready[0]["id"] == "t1"

    # t1 done — mark it completed before checking again
    subtasks[0]["status"] = "completed"
    ready = get_ready_subtasks(subtasks, {"t1": "result"})
    assert len(ready) == 1
    assert ready[0]["id"] == "t2"


def test_get_ready_subtasks_skips_completed():
    subtasks = [
        {"id": "t1", "status": "completed", "dependencies": []},
        {"id": "t2", "status": "pending", "dependencies": ["t1"]},
    ]
    ready = get_ready_subtasks(subtasks, {"t1": "done"})
    assert len(ready) == 1
    assert ready[0]["id"] == "t2"


def test_all_subtasks_done_true():
    subtasks = [
        {"id": "t1", "status": "completed"},
        {"id": "t2", "status": "completed"},
    ]
    assert all_subtasks_done(subtasks) is True


def test_all_subtasks_done_false():
    subtasks = [
        {"id": "t1", "status": "completed"},
        {"id": "t2", "status": "pending"},
    ]
    assert all_subtasks_done(subtasks) is False
