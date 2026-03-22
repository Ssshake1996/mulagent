"""End-to-end tests: full flow from HTTP request to response.

Tests verify the complete pipeline via the ReAct orchestrator.
Without LLM configured, tasks return a "failed" status (no mock fallback).
"""

import pytest
from httpx import AsyncClient, ASGITransport

from gateway.routes import init_dependencies
from gateway.streaming import init_stream_dependencies
from main import create_app


@pytest.fixture(scope="module")
def setup():
    """Initialize shared dependencies once for all e2e tests."""
    init_dependencies(llm=None, llm_manager=None)
    init_stream_dependencies(llm=None, llm_manager=None)


@pytest.fixture
async def client(setup):
    app = create_app()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ─── Flow tests ───


@pytest.mark.asyncio
async def test_e2e_code_task(client):
    """Task submission should return a response (failed without LLM)."""
    resp = await client.post("/api/v1/tasks", json={
        "input": "write a Python function to check if a number is prime",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("completed", "failed")
    assert data["final_output"] != ""


@pytest.mark.asyncio
async def test_e2e_research_task(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "search for the best practices in microservices architecture",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("completed", "failed")


@pytest.mark.asyncio
async def test_e2e_general_task(client):
    """General/unclassified task should still return a response."""
    resp = await client.post("/api/v1/tasks", json={
        "input": "hello, how are you today?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("completed", "failed")


# ─── Session continuity ───


@pytest.mark.asyncio
async def test_e2e_session_continuity(client):
    """Same session_id should be preserved across requests."""
    sid = "e2e-session-001"
    r1 = await client.post("/api/v1/tasks", json={"input": "write hello world", "session_id": sid})
    r2 = await client.post("/api/v1/tasks", json={"input": "debug my code", "session_id": sid})
    assert r1.json()["session_id"] == sid
    assert r2.json()["session_id"] == sid


# ─── SSE streaming ───


@pytest.mark.asyncio
async def test_e2e_stream_produces_events(client):
    """Stream endpoint should return SSE content type with events."""
    resp = await client.post("/api/v1/tasks/stream", json={"input": "write a sort function"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
