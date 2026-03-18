"""FastAPI routes — the HTTP entry point for the system."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import async_sessionmaker

from agents.adapter import AdapterFactory
from agents.registry import AgentRegistry
from common.llm import LLMManager
from evolution.feedback import record_feedback
from evolution.trace import record_task_trace
from graph.orchestrator import build_graph

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons (initialized in lifespan)
_registry: AgentRegistry | None = None
_factory: AdapterFactory | None = None
_llm = None
_llm_manager: LLMManager | None = None
_db_session_factory: async_sessionmaker | None = None
_qdrant = None
_collection_name: str = "case_library"


def init_dependencies(
    registry: AgentRegistry,
    factory: AdapterFactory,
    llm=None,
    llm_manager=None,
    db_session_factory=None,
    qdrant=None,
    collection_name: str = "case_library",
):
    global _registry, _factory, _llm, _llm_manager, _db_session_factory, _qdrant, _collection_name
    _registry = registry
    _factory = factory
    _llm = llm
    _llm_manager = llm_manager
    _db_session_factory = db_session_factory
    _qdrant = qdrant
    _collection_name = collection_name


class TaskRequest(BaseModel):
    input: str = Field(..., min_length=1, description="User's task description")
    session_id: str | None = Field(default=None, description="Optional session ID for continuity")
    model: str | None = Field(default=None, description="Model ID to use (from config)")


class TaskResponse(BaseModel):
    task_id: str
    session_id: str
    status: str
    intent: str
    model_used: str
    final_output: str
    quality_passed: bool
    subtasks: list[dict[str, Any]]
    timing: dict[str, int] | None = None


class FeedbackRequest(BaseModel):
    task_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None


class FeedbackResponse(BaseModel):
    id: str
    task_id: str
    rating: int
    status: str


class HealthResponse(BaseModel):
    status: str
    version: str
    agents_count: int
    llm_default: str
    llm_models: list[dict[str, str]]
    db_connected: bool
    qdrant_connected: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    agent_count = len(_registry.list_all()) if _registry else 0
    return HealthResponse(
        status="ok",
        version="0.1.0",
        agents_count=agent_count,
        llm_default=_llm_manager.default_id if _llm_manager else "",
        llm_models=_llm_manager.list_models() if _llm_manager else [],
        db_connected=_db_session_factory is not None,
        qdrant_connected=_qdrant is not None,
    )


@router.post("/tasks", response_model=TaskResponse)
async def create_task(req: TaskRequest):
    """Submit a task for multi-agent processing."""
    session_id = req.session_id or str(uuid.uuid4())
    task_id = uuid.uuid4()

    # Resolve which LLM to use
    llm = _llm
    model_used = _llm_manager.default_id if _llm_manager else "mock"
    if req.model and _llm_manager:
        selected = _llm_manager.get(req.model)
        if selected is None:
            available = [m["id"] for m in _llm_manager.list_models()]
            raise HTTPException(
                status_code=400,
                detail=f"Model '{req.model}' not found. Available: {available}",
            )
        llm = selected
        model_used = req.model

    # Build graph
    if llm != _llm:
        factory = AdapterFactory(
            llm_client=llm,
            use_openclaw=_factory._use_openclaw if _factory else False,
            openclaw_timeout=_factory._openclaw_timeout if _factory else 120,
        )
    else:
        factory = _factory
    llm_light = _llm_manager.get_light(req.model) if _llm_manager else None
    graph = build_graph(
        registry=_registry, adapter_factory=factory, llm=llm, llm_light=llm_light,
        qdrant=_qdrant, collection_name=_collection_name,
    )
    compiled = graph.compile()

    try:
        result = await compiled.ainvoke({
            "user_input": req.input,
            "session_id": session_id,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Task execution failed: {e}")

    # Persist trace to PostgreSQL (fire-and-forget, don't block response)
    if _db_session_factory is not None:
        try:
            async with _db_session_factory() as db:
                await record_task_trace(
                    db,
                    session_id=session_id,
                    user_input=req.input,
                    intent=result.get("intent", "general"),
                    dag_plan=result.get("subtasks"),
                    subtask_results=result.get("subtask_results", {}),
                    final_output=result.get("final_output", ""),
                    status=result.get("status", "unknown"),
                    subtasks=result.get("subtasks", []),
                )
        except Exception as e:
            logger.warning("Failed to record trace: %s", e)

    # Extract and store experience in Qdrant (fire-and-forget)
    if _qdrant is not None and result.get("status") == "completed":
        try:
            from common.vector import text_to_embedding
            from evolution.experience import extract_experience, store_experience

            experience = await extract_experience(result, llm=llm)
            if experience:
                embedding = text_to_embedding(req.input)
                await store_experience(
                    _qdrant, _collection_name, experience, embedding, str(task_id),
                )
                logger.info("Experience stored for task %s", task_id)
        except Exception as e:
            logger.warning("Failed to extract/store experience: %s", e)

    return TaskResponse(
        task_id=str(task_id),
        session_id=session_id,
        status=result.get("status", "unknown"),
        intent=result.get("intent", "general"),
        model_used=model_used,
        final_output=result.get("final_output", ""),
        quality_passed=result.get("quality_passed", False),
        subtasks=result.get("subtasks", []),
        timing=result.get("_timing"),
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(req: FeedbackRequest):
    """Submit feedback for a completed task and trigger evolution."""
    if _db_session_factory is None:
        raise HTTPException(status_code=503, detail="Database not available")

    try:
        task_uuid = uuid.UUID(req.task_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid task_id format")

    async with _db_session_factory() as db:
        fb = await record_feedback(
            db,
            task_id=task_uuid,
            rating=req.rating,
            comment=req.comment,
        )

    # Trigger feedback-driven evolution (fire-and-forget)
    try:
        from evolution.feedback_loop import process_feedback

        llm = _llm_manager.get() if _llm_manager else _llm
        async with _db_session_factory() as db:
            await process_feedback(
                rating=req.rating,
                task_id=task_uuid,
                comment=req.comment,
                db=db,
                registry=_registry,
                qdrant=_qdrant,
                collection_name=_collection_name,
                llm=llm,
            )
    except Exception as e:
        logger.warning("Feedback evolution failed: %s", e)

    return FeedbackResponse(
        id=str(fb.id),
        task_id=req.task_id,
        rating=req.rating,
        status="recorded",
    )


@router.get("/agents")
async def list_agents():
    if _registry is None:
        return {"agents": []}
    return {
        "agents": [
            {
                "id": a.id, "name": a.name, "description": a.description,
                "skills": a.skills, "success_rate": a.success_rate, "total_runs": a.total_runs,
            }
            for a in _registry.list_all()
        ]
    }


@router.get("/models")
async def list_models():
    if _llm_manager is None:
        return {"default": "", "models": []}
    return {"default": _llm_manager.default_id, "models": _llm_manager.list_models()}
