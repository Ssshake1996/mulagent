"""Tests for API gateway."""

import pytest
from httpx import AsyncClient, ASGITransport

from main import create_app


@pytest.fixture
async def client():
    app = create_app()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        from agents.adapter import AdapterFactory
        from agents.registry import load_registry
        from gateway.routes import init_dependencies
        from gateway.streaming import init_stream_dependencies

        registry = load_registry()
        factory = AdapterFactory()
        init_dependencies(registry, factory, llm=None, llm_manager=None, db_session_factory=None, qdrant=None)
        init_stream_dependencies(registry, factory, llm=None, llm_manager=None, db_session_factory=None, qdrant=None)

        yield ac


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("ok", "degraded")
    assert data["agents_count"] == 3


@pytest.mark.asyncio
async def test_list_agents(client):
    resp = await client.get("/api/v1/agents")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["agents"]) == 3


@pytest.mark.asyncio
async def test_list_models(client):
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "default" in data
    assert "models" in data


@pytest.mark.asyncio
async def test_create_task_code(client):
    resp = await client.post("/api/v1/tasks", json={"input": "write a Python sort function"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["final_output"] != ""


@pytest.mark.asyncio
async def test_create_task_with_invalid_model(client):
    resp = await client.post("/api/v1/tasks", json={"input": "hello", "model": "nonexistent"})
    # Without llm_manager, model param is ignored (no error)
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_create_task_empty_input(client):
    resp = await client.post("/api/v1/tasks", json={"input": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_task_with_session_id(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "calculate 2+2",
        "session_id": "my-session-123",
    })
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "my-session-123"


@pytest.mark.asyncio
async def test_feedback_no_db(client):
    resp = await client.post("/api/v1/feedback", json={
        "task_id": "00000000-0000-0000-0000-000000000001",
        "rating": 5,
    })
    assert resp.status_code == 503
    assert "Database not available" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_health_db_and_qdrant_disconnected(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json()["db_connected"] is False
    assert resp.json()["qdrant_connected"] is False


@pytest.mark.asyncio
async def test_stream_task(client):
    resp = await client.post("/api/v1/tasks/stream", json={"input": "write hello world"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
