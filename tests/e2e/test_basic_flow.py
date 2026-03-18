"""End-to-end tests: full flow from HTTP request to response.

These tests verify the complete pipeline:
  User input → API → Dispatcher → DAG Planner → Agent Execution → Quality Gate → Response
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


# ─── Flow tests per intent category ───


@pytest.mark.asyncio
async def test_e2e_code_task(client):
    """Code task: dispatched to code agent, executed, quality checked."""
    resp = await client.post("/api/v1/tasks", json={
        "input": "write a Python function to check if a number is prime",
    })
    assert resp.status_code == 200
    data = resp.json()

    assert data["intent"] == "code"
    assert data["status"] == "completed"
    assert data["quality_passed"] is True
    assert len(data["subtasks"]) >= 1
    assert data["subtasks"][0]["status"] == "completed"
    assert "Code Agent" in data["final_output"]


@pytest.mark.asyncio
async def test_e2e_research_task(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "search for the best practices in microservices architecture",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "research"
    assert data["status"] == "completed"
    assert "Research Agent" in data["final_output"]


@pytest.mark.asyncio
async def test_e2e_data_task(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "analyze the sales data and create a chart",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "data"
    assert data["status"] == "completed"
    assert "Data" in data["final_output"]


@pytest.mark.asyncio
async def test_e2e_writing_task(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "write a blog article about artificial intelligence trends",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "writing"
    assert data["status"] == "completed"
    assert "Writing Agent" in data["final_output"]


@pytest.mark.asyncio
async def test_e2e_reasoning_task(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "calculate the area of a circle with radius 5",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "reasoning"
    assert data["status"] == "completed"
    assert "Reasoning Agent" in data["final_output"]


@pytest.mark.asyncio
async def test_e2e_chinese_task(client):
    resp = await client.post("/api/v1/tasks", json={
        "input": "帮我写一段Python代码实现快速排序",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "code"
    assert data["status"] == "completed"


@pytest.mark.asyncio
async def test_e2e_general_task(client):
    """General/unclassified task should still complete."""
    resp = await client.post("/api/v1/tasks", json={
        "input": "hello, how are you today?",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "general"
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
    """Verify all 5 agents are registered and accessible."""
    resp = await client.get("/api/v1/agents")
    data = resp.json()
    assert len(data["agents"]) == 5
    ids = {a["id"] for a in data["agents"]}
    assert ids == {"code_agent", "research_agent", "data_agent", "writing_agent", "reasoning_agent"}


# ─── SSE streaming ───


@pytest.mark.asyncio
async def test_e2e_stream_produces_events(client):
    """Stream endpoint should return SSE content type with events."""
    resp = await client.post("/api/v1/tasks/stream", json={"input": "write a sort function"})
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    # Body should contain event data
    body = resp.text
    assert "dispatch" in body or "event" in body
