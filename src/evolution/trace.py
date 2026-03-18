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
) -> TaskTrace:
    """Record a complete task execution trace."""
    task = TaskTrace(
        id=uuid.uuid4(),
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
