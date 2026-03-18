"""Tests for feedback-driven evolution loop."""

import uuid

import pytest
from unittest.mock import AsyncMock, MagicMock

from agents.registry import AgentMeta, AgentRegistry
from common.vector import get_qdrant_client, ensure_collection, text_to_embedding
from evolution.experience import store_experience
from evolution.feedback_loop import (
    _boost_experience,
    _update_agent_stats,
    process_feedback,
)


# --- Agent stats update tests ---

@pytest.mark.asyncio
async def test_update_agent_stats_positive():
    """Rating 4-5 should increase agent success rate."""
    registry = AgentRegistry()
    registry.register(AgentMeta(id="code_agent", name="Code", description="", skills=["code"], success_rate=0.5, total_runs=10))

    # Mock db query returning agent_id
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [("code_agent",)]
    mock_db.execute.return_value = mock_result

    updates = await _update_agent_stats(mock_db, uuid.uuid4(), rating=5, registry=registry)

    assert len(updates) == 1
    assert updates[0]["agent_id"] == "code_agent"
    assert updates[0]["new_success_rate"] > 0.5  # should increase


@pytest.mark.asyncio
async def test_update_agent_stats_negative():
    """Rating 1-2 should decrease agent success rate."""
    registry = AgentRegistry()
    registry.register(AgentMeta(id="code_agent", name="Code", description="", skills=["code"], success_rate=0.8, total_runs=10))

    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [("code_agent",)]
    mock_db.execute.return_value = mock_result

    updates = await _update_agent_stats(mock_db, uuid.uuid4(), rating=1, registry=registry)

    assert len(updates) == 1
    assert updates[0]["new_success_rate"] < 0.8  # should decrease


@pytest.mark.asyncio
async def test_update_agent_stats_neutral():
    """Rating 3 should not update stats."""
    registry = AgentRegistry()
    registry.register(AgentMeta(id="code_agent", name="Code", description="", skills=["code"]))

    mock_db = AsyncMock()
    updates = await _update_agent_stats(mock_db, uuid.uuid4(), rating=3, registry=registry)
    assert updates == []


# --- Experience boost tests ---

def test_boost_experience_found():
    client = get_qdrant_client(in_memory=True)
    collection = "test_boost"
    ensure_collection(client, collection)

    # Store initial experience
    from qdrant_client.models import PointStruct
    client.upsert(
        collection_name=collection,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=text_to_embedding("test"),
            payload={"task_id": "task-123", "quality_score": 1.0, "problem_pattern": "test"},
        )],
    )

    result = _boost_experience(client, collection, "task-123", rating=5)
    assert result is True

    # Verify the quality_score was updated
    points, _ = client.scroll(collection_name=collection, limit=1)
    assert points[0].payload["quality_score"] == 2.0  # 1.0 + (5-3)*0.5


def test_boost_experience_not_found():
    client = get_qdrant_client(in_memory=True)
    collection = "test_boost_empty"
    ensure_collection(client, collection)

    result = _boost_experience(client, collection, "nonexistent", rating=5)
    assert result is False


# --- Full process_feedback tests ---

@pytest.mark.asyncio
async def test_process_feedback_unknown_task():
    """Feedback for unknown task should return error."""
    mock_db = AsyncMock()
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db.execute.return_value = mock_result

    result = await process_feedback(
        rating=5, task_id=uuid.uuid4(), comment=None, db=mock_db,
    )
    assert result["error"] == "task_trace_not_found"


@pytest.mark.asyncio
async def test_process_feedback_high_rating_boosts():
    """High rating should trigger experience boost."""
    mock_db = AsyncMock()

    # Mock trace lookup
    mock_trace = MagicMock()
    mock_trace.id = uuid.uuid4()
    mock_trace.user_input = "test"
    mock_trace.intent_category = "code"
    mock_trace.status = "completed"
    mock_trace.final_output = "result"

    mock_result_trace = MagicMock()
    mock_result_trace.scalar_one_or_none.return_value = mock_trace

    # Mock agent lookup (empty)
    mock_result_agents = MagicMock()
    mock_result_agents.all.return_value = []

    mock_db.execute.side_effect = [mock_result_trace, mock_result_agents]

    client = get_qdrant_client(in_memory=True)
    collection = "test_loop"
    ensure_collection(client, collection)

    # Pre-store experience for this task
    exp = {"problem_pattern": "test", "recommended_strategy": "direct", "recommended_agents": [], "tips": "none"}
    await store_experience(client, collection, exp, text_to_embedding("test"), str(mock_trace.id))

    result = await process_feedback(
        rating=5, task_id=mock_trace.id, comment="great",
        db=mock_db, qdrant=client, collection_name=collection,
    )

    boost_actions = [a for a in result["actions"] if a["type"] == "experience_boosted"]
    assert len(boost_actions) == 1
