"""Server-Sent Events (SSE) streaming for real-time task progress.

Uses the ReAct orchestrator with per-round progress events.

Events emitted:
  - "progress": {round, action, detail} — each tool call or thinking step
  - "result":   {status, final_output, tools_used} — final answer
  - "error":    {message} — on failure
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncGenerator

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from common.llm import LLMManager
from graph.orchestrator import run_react

logger = logging.getLogger(__name__)
stream_router = APIRouter()

from sqlalchemy.ext.asyncio import async_sessionmaker

_llm = None
_llm_manager: LLMManager | None = None
_db_session_factory: async_sessionmaker | None = None
_qdrant = None
_collection_name: str = "case_library"


def init_stream_dependencies(
    llm=None, llm_manager=None,
    db_session_factory=None, qdrant=None, collection_name: str = "case_library",
):
    global _llm, _llm_manager, _db_session_factory, _qdrant, _collection_name
    _llm = llm
    _llm_manager = llm_manager
    _db_session_factory = db_session_factory
    _qdrant = qdrant
    _collection_name = collection_name


class StreamRequest(BaseModel):
    input: str = Field(..., min_length=1)
    session_id: str | None = None
    model: str | None = None


async def _task_event_generator(
    user_input: str, session_id: str, model_id: str | None,
) -> AsyncGenerator[dict, None]:
    """Generate SSE events as the ReAct loop executes."""
    from common.trace_context import trace_ctx

    trace_id = trace_ctx.new_trace()
    logger.info("Stream task started (trace=%s, session=%s)", trace_id, session_id)

    # Resolve LLM
    llm = _llm
    if model_id and _llm_manager:
        llm = _llm_manager.get(model_id) or _llm

    # Progress event queue — filled by the callback, consumed by this generator
    progress_queue: asyncio.Queue[dict] = asyncio.Queue()
    start_time = time.monotonic()

    async def on_progress(round_num: int, action: str, detail: str):
        """Callback from react_loop, pushes events into the queue."""
        elapsed = time.monotonic() - start_time
        await progress_queue.put({
            "event": "progress",
            "data": json.dumps({
                "round": round_num,
                "action": action,
                "detail": detail,
                "elapsed_s": round(elapsed, 1),
            }, ensure_ascii=False),
        })

    # Run the task in background, collect result
    result_holder: dict[str, Any] = {}
    error_holder: dict[str, str] = {}

    async def _run():
        try:
            result = await run_react(
                user_input=user_input,
                llm=llm,
                qdrant=_qdrant,
                collection_name=_collection_name,
                on_progress=on_progress,
            )
            result_holder.update(result)
        except Exception as e:
            logger.exception("Stream task failed")
            error_holder["message"] = str(e)
        finally:
            await progress_queue.put(None)  # sentinel

    # Start task
    task = asyncio.create_task(_run())

    # Yield progress events as they arrive
    while True:
        event = await progress_queue.get()
        if event is None:
            break
        yield event

    # Wait for task to fully complete
    await task

    # Yield final result or error
    elapsed = time.monotonic() - start_time
    if error_holder:
        yield {
            "event": "error",
            "data": json.dumps({
                "message": error_holder["message"],
                "elapsed_s": round(elapsed, 1),
            }, ensure_ascii=False),
        }
    else:
        result = result_holder
        yield {
            "event": "result",
            "data": json.dumps({
                "trace_id": trace_id,
                "status": result.get("status", "unknown"),
                "final_output": result.get("final_output", ""),
                "intent": result.get("intent", "react"),
                "tools_used": result.get("tools_used", []),
                "elapsed_s": round(elapsed, 1),
            }, ensure_ascii=False),
        }

    # Record metrics
    from common.logging_config import metrics
    metrics.record_task(
        "react_stream", elapsed,
        result_holder.get("status", "error"),
        tools_used=result_holder.get("tools_used", []),
    )


@stream_router.post("/tasks/stream")
async def stream_task(req: StreamRequest):
    """Submit a task and receive SSE stream of progress events.

    Events:
    - progress: {round, action, detail, elapsed_s}
    - result: {status, final_output, intent, tools_used, elapsed_s}
    - error: {message, elapsed_s}
    """
    session_id = req.session_id or str(uuid.uuid4())
    return EventSourceResponse(_task_event_generator(req.input, session_id, req.model))
