"""Trace recorder — persists execution traces to PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from models.trace import SubtaskTrace, TaskTrace


async def record_task_trace(
    session: AsyncSession,
    *,
    session_id: str,
    user_input: str,
    intent: str,
    dag_plan: list[dict] | None,
    subtask_results: dict[str, str],
    final_output: str,
    status: str,
    subtasks: list[dict[str, Any]],
    trace_id: str = "",
) -> TaskTrace:
    """Record a complete task execution trace."""
    # Auto-detect trace_id from context if not provided
    if not trace_id:
        try:
            from common.trace_context import get_trace_id
            trace_id = get_trace_id()
        except Exception:
            pass

    task = TaskTrace(
        id=uuid.uuid4(),
        trace_id=trace_id or None,
        session_id=session_id,
        user_input=user_input,
        intent_category=intent,
        dag_plan=dag_plan,
        final_output=final_output,
        status=status,
        completed_at=datetime.now(timezone.utc) if status == "completed" else None,
    )
    session.add(task)

    for st in subtasks:
        sub = SubtaskTrace(
            id=uuid.uuid4(),
            task_id=task.id,
            agent_id=st.get("agent_type", "unknown"),
            subtask_name=st.get("name", ""),
            input_data={"description": st.get("description", "")},
            output_data={"result": subtask_results.get(st.get("id", ""), "")},
            status=st.get("status", "unknown"),
            duration_ms=st.get("duration_ms"),
        )
        session.add(sub)

    await session.commit()
    return task
