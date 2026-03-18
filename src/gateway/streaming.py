"""Server-Sent Events (SSE) streaming for real-time task progress."""

from __future__ import annotations

import json
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agents.adapter import AdapterFactory
from agents.registry import AgentRegistry
from common.llm import LLMManager
from graph.orchestrator import build_graph

stream_router = APIRouter()

from sqlalchemy.ext.asyncio import async_sessionmaker

_registry: AgentRegistry | None = None
_factory: AdapterFactory | None = None
_llm = None
_llm_manager: LLMManager | None = None
_db_session_factory: async_sessionmaker | None = None
_qdrant = None
_collection_name: str = "case_library"


def init_stream_dependencies(
    registry: AgentRegistry, factory: AdapterFactory, llm=None, llm_manager=None,
    db_session_factory=None, qdrant=None, collection_name: str = "case_library",
):
    global _registry, _factory, _llm, _llm_manager, _db_session_factory, _qdrant, _collection_name
    _registry = registry
    _factory = factory
    _llm = llm
    _llm_manager = llm_manager
    _db_session_factory = db_session_factory
    _qdrant = qdrant
    _collection_name = collection_name


class StreamRequest(BaseModel):
    input: str = Field(..., min_length=1)
    session_id: str | None = None
    model: str | None = None


async def _task_event_generator(user_input: str, session_id: str, model_id: str | None) -> AsyncGenerator[dict, None]:
    """Generate SSE events as the graph executes."""
    # Resolve LLM
    llm = _llm
    if model_id and _llm_manager:
        llm = _llm_manager.get(model_id) or _llm

    factory = AdapterFactory(llm_client=llm) if llm != _llm else _factory
    llm_light = _llm_manager.get_light(model_id) if _llm_manager else None
    graph = build_graph(
        registry=_registry, adapter_factory=factory, llm=llm, llm_light=llm_light,
        qdrant=_qdrant, collection_name=_collection_name,
    )
    compiled = graph.compile()

    async for event in compiled.astream(
        {"user_input": user_input, "session_id": session_id},
        stream_mode="updates",
    ):
        for node_name, node_output in event.items():
            yield {
                "event": node_name,
                "data": json.dumps({
                    "node": node_name,
                    "status": node_output.get("status", ""),
                    "intent": node_output.get("intent", ""),
                    "final_output": node_output.get("final_output", ""),
                    "quality_passed": node_output.get("quality_passed", None),
                }, ensure_ascii=False),
            }

    yield {"event": "done", "data": json.dumps({"status": "completed"})}


@stream_router.post("/tasks/stream")
async def stream_task(req: StreamRequest):
    """Submit a task and receive SSE stream of progress events."""
    session_id = req.session_id or str(uuid.uuid4())
    return EventSourceResponse(_task_event_generator(req.input, session_id, req.model))
