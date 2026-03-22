"""FastAPI routes — the HTTP entry point for the system."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import async_sessionmaker

from common.llm import LLMManager
from evolution.feedback import record_feedback
from evolution.trace import record_task_trace
from graph.orchestrator import run_react

logger = logging.getLogger(__name__)
router = APIRouter()

# Module-level singletons (initialized in lifespan)
_llm = None
_llm_manager: LLMManager | None = None
_db_session_factory: async_sessionmaker | None = None
_qdrant = None
_collection_name: str = "case_library"


def init_dependencies(
    llm=None,
    llm_manager=None,
    db_session_factory=None,
    qdrant=None,
    collection_name: str = "case_library",
):
    global _llm, _llm_manager, _db_session_factory, _qdrant, _collection_name
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
    llm_default: str
    llm_models: list[dict[str, str]]
    db_connected: bool
    qdrant_connected: bool
    redis_connected: bool = False
    uptime_s: float = 0
    tools_count: int = 0

_start_time: float = 0


def set_start_time():
    global _start_time
    import time
    _start_time = time.monotonic()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check: LLM, DB, Qdrant, Redis, tools."""
    import time

    # Check Redis connectivity
    redis_ok = False
    try:
        from common.redis_client import get_redis
        r = await get_redis()
        redis_ok = r is not None
    except Exception:
        pass

    # Check Qdrant connectivity
    qdrant_ok = False
    if _qdrant is not None:
        try:
            _qdrant.get_collections()
            qdrant_ok = True
        except Exception:
            pass

    # Count available tools
    tools_count = 0
    try:
        from tools.registry import get_default_tools
        tools_count = len(get_default_tools().as_dict())
    except Exception:
        pass

    uptime = time.monotonic() - _start_time if _start_time else 0

    # Overall status: degraded if any component is down
    status = "ok"
    if not qdrant_ok:
        status = "degraded"

    return HealthResponse(
        status=status,
        version="0.1.0",
        llm_default=_llm_manager.default_id if _llm_manager else "",
        llm_models=_llm_manager.list_models() if _llm_manager else [],
        db_connected=_db_session_factory is not None,
        qdrant_connected=qdrant_ok,
        redis_connected=redis_ok,
        uptime_s=round(uptime, 1),
        tools_count=tools_count,
    )


@router.get("/health/liveness")
async def liveness():
    """Kubernetes liveness probe — always returns 200 if process is running."""
    return {"status": "alive"}


@router.get("/health/readiness")
async def readiness():
    """Kubernetes readiness probe — checks critical dependencies."""
    checks = {"llm": _llm is not None}

    if _qdrant is not None:
        try:
            _qdrant.get_collections()
            checks["qdrant"] = True
        except Exception:
            checks["qdrant"] = False

    all_ready = all(checks.values())
    if not all_ready:
        from fastapi.responses import JSONResponse
        return JSONResponse({"ready": False, "checks": checks}, status_code=503)
    return {"ready": True, "checks": checks}


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

    # Run via ReAct orchestrator
    try:
        result = await run_react(
            user_input=req.input,
            llm=llm,
            qdrant=_qdrant,
            collection_name=_collection_name,
        )
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
            from common.vector import text_to_embedding_async
            from evolution.experience import extract_experience, store_experience

            experience = await extract_experience(result, llm=llm)
            if experience:
                embedding = await text_to_embedding_async(req.input)
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


@router.get("/models")
async def list_models():
    if _llm_manager is None:
        return {"default": "", "models": []}
    return {"default": _llm_manager.default_id, "models": _llm_manager.list_models()}


@router.get("/metrics")
async def get_metrics():
    """Return system metrics summary (JSON)."""
    from common.logging_config import metrics
    summary = metrics.get_summary(last_n_minutes=60)
    # Merge with observability metrics
    try:
        from common.observability import metrics as obs_metrics
        summary["observability"] = obs_metrics.summary()
    except Exception:
        pass
    return summary


@router.get("/metrics/prometheus")
async def prometheus_metrics():
    """Return metrics in Prometheus text format."""
    from fastapi.responses import PlainTextResponse
    try:
        from common.observability import metrics as obs_metrics
        return PlainTextResponse(
            content=obs_metrics.to_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )
    except Exception as e:
        return PlainTextResponse(content=f"# Error: {e}", media_type="text/plain")


@router.get("/traces")
async def get_traces(limit: int = 20):
    """Return recent distributed traces for debugging."""
    try:
        from common.observability import tracer
        return {"traces": tracer.get_recent_traces(limit=limit)}
    except Exception as e:
        return {"traces": [], "error": str(e)}


@router.get("/checkpoints")
async def list_checkpoints(session_id: str = ""):
    """List available task checkpoints for resumption."""
    try:
        from graph.checkpoint import list_checkpoints
        return {"checkpoints": await list_checkpoints(session_id)}
    except Exception as e:
        return {"checkpoints": [], "error": str(e)}


@router.post("/config/reload")
async def reload_config():
    """Hot-reload settings from config/settings.yaml."""
    try:
        from common.config import reload_settings
        from tools.isolation import reload_roles
        new_settings = reload_settings()
        reload_roles()
        return {
            "status": "reloaded",
            "debug": new_settings.debug,
            "react_timeout": new_settings.react.timeout,
            "react_max_rounds": new_settings.react.max_rounds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}")
