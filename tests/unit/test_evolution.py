"""Tests for evolution layer: experience extraction, storage, and retrieval."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from common.vector import VECTOR_DIM, ensure_collection, get_qdrant_client, text_to_embedding
from evolution.experience import extract_experience, search_similar_experiences, store_experience


# --- text_to_embedding tests ---

def test_embedding_dimension():
    vec = text_to_embedding("hello world")
    assert len(vec) == VECTOR_DIM


def test_embedding_deterministic():
    v1 = text_to_embedding("test input")
    v2 = text_to_embedding("test input")
    assert v1 == v2


def test_embedding_different_texts():
    v1 = text_to_embedding("hello")
    v2 = text_to_embedding("goodbye")
    assert v1 != v2


def test_embedding_normalized():
    vec = text_to_embedding("normalize me")
    norm = sum(v * v for v in vec) ** 0.5
    assert abs(norm - 1.0) < 1e-6


# --- Qdrant in-memory tests ---

def test_qdrant_in_memory():
    client = get_qdrant_client(in_memory=True)
    collections = client.get_collections().collections
    assert isinstance(collections, list)


def test_ensure_collection_creates():
    client = get_qdrant_client(in_memory=True)
    ensure_collection(client, "test_collection")
    names = [c.name for c in client.get_collections().collections]
    assert "test_collection" in names


def test_ensure_collection_idempotent():
    client = get_qdrant_client(in_memory=True)
    ensure_collection(client, "test_col")
    ensure_collection(client, "test_col")  # should not raise
    names = [c.name for c in client.get_collections().collections]
    assert names.count("test_col") == 1


# --- store + search round-trip ---

@pytest.mark.asyncio
async def test_store_and_search_experience():
    client = get_qdrant_client(in_memory=True)
    collection = "test_experiences"
    ensure_collection(client, collection)

    experience = {
        "problem_pattern": "sorting algorithm implementation",
        "recommended_strategy": "single agent, direct code generation",
        "recommended_agents": ["code_agent"],
        "tips": "use built-in sorted() for simple cases",
    }
    embedding = text_to_embedding("write a sorting function in Python")

    point_id = await store_experience(client, collection, experience, embedding, "task-001")
    assert point_id  # non-empty string

    # Search with the same embedding → should find the experience
    results = await search_similar_experiences(client, collection, embedding, top_k=1)
    assert len(results) == 1
    assert results[0]["problem_pattern"] == "sorting algorithm implementation"
    assert results[0]["score"] > 0.9  # near-exact match


@pytest.mark.asyncio
async def test_search_empty_collection():
    client = get_qdrant_client(in_memory=True)
    collection = "empty_col"
    ensure_collection(client, collection)

    results = await search_similar_experiences(
        client, collection, text_to_embedding("anything"), top_k=3,
    )
    assert results == []


# --- extract_experience tests ---

@pytest.mark.asyncio
async def test_extract_experience_no_llm():
    result = await extract_experience({"user_input": "test"}, llm=None)
    assert result is None


@pytest.mark.asyncio
async def test_extract_experience_with_mock_llm():
    mock_llm = AsyncMock()
    mock_llm.ainvoke.return_value = MagicMock(
        content='{"problem_pattern": "test", "recommended_strategy": "direct", "recommended_agents": ["code_agent"], "tips": "none"}'
    )

    result = await extract_experience(
        {"user_input": "write code", "intent": "code", "subtasks": [], "status": "completed"},
        llm=mock_llm,
    )
    assert result is not None
    assert result["problem_pattern"] == "test"
    assert result["recommended_agents"] == ["code_agent"]
