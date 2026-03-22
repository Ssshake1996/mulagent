"""End-to-end tests: full flow from HTTP request to response.

Tests verify the complete pipeline in mock mode (no LLM):
  User input → API → Legacy pipeline (dispatch → plan → execute → quality) → Response

When LLM is available, ReAct mode is used instead.
"""

import pytest
from httpx import AsyncClient, ASGITransport

from agents.adapter import AdapterFactory
from agents.registry import load_registry
from gateway.routes import init_dependencies
from gateway.streaming import init_stream_dependencies
from main import create_app


@pytest.fixture(scope="module")
def setup():
    """Initialize shared dependencies once for all e2e tests."""
    registry = load_registry()
    factory = AdapterFactory()
    init_dependencies(registry, factory, llm=None, llm_manager=None)
    init_stream_dependencies(registry, factory, llm=None, llm_manager=None)
    return registry, factory


@pytest.fixture
async def client(setup):
    app = create_app()
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ─── Flow tests (legacy mock mode) ───


@pytest.mark.asyncio
async def test_e2e_code_task(client):
    """Code task: dispatched and completed via legacy pipeline."""
    resp = await client.post("/api/v1/tasks", json={
        "input": "write a Python function to check if a number is prime",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["final_output"] != ""


@pytest.mark.asyncio
async def test_e2e_research_task(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "search for the best practices in microservices architecture",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_e2e_general_task(client):
    """General/unclassified task should still complete."""
    resp = await client.post("/api/v1/tasks", json={
        "input": "hello, how are you today?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"


# ─── Session continuity ───


@pytest.mark.asyncio
async def test_e2e_session_continuity(client):
    """Same session_id should be preserved across requests."""
    sid = "e2e-session-001"
    r1 = await client.post("/api/v1/tasks", json={"input": "write hello world", "session_id": sid})
    r2 = await client.post("/api/v1/tasks", json={"input": "debug my code", "session_id": sid})
    assert r1.json()["session_id"] == sid
    assert r2.json()["session_id"] == sid


# ─── Agent registry verification ───


@pytest.mark.asyncio
async def test_e2e_agents_endpoint(client):
    """Verify 3 agents are registered (thinker, retriever, executor)."""
    resp = await client.get("/api/v1/agents")
    data = resp.json()
    assert len(data["agents"]) == 3
    ids = {a["id"] for a in data["agents"]}
    assert ids == {"thinker", "retriever", "executor"}


# ─── SSE streaming ───


@pytest.mark.asyncio
async def test_e2e_stream_produces_events(client):
    """Stream endpoint should return SSE content type with events."""
    resp = await client.post("/api/v1/tasks/stream", json={"input": "write a sort function"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "dispatch" in body or "event" in body
