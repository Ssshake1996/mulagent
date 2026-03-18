"""Integration test — verify trace persistence to PostgreSQL."""

import uuid

import pytest
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from common.db import Base, create_engine, create_session_factory
from evolution.feedback import record_feedback
from evolution.trace import record_task_trace
from models.feedback import Feedback
from models.trace import SubtaskTrace, TaskTrace


@pytest.fixture
async def db_session():
    """Create a real async session against local PostgreSQL."""
    engine = create_engine()
    session_factory = create_session_factory(engine)
    async with session_factory() as session:
        yield session
    await engine.dispose()


@pytest.mark.asyncio
async def test_record_task_trace(db_session: AsyncSession):
    """Write a task trace and verify it persists."""
    session_id = f"test-{uuid.uuid4().hex[:8]}"
    trace = await record_task_trace(
        db_session,
        session_id=session_id,
        user_input="integration test input",
        intent="code",
        dag_plan=[{"id": "t1", "name": "test subtask"}],
        subtask_results={"t1": "done"},
        final_output="test output",
        status="completed",
        subtasks=[
            {"id": "t1", "name": "test subtask", "agent_type": "code_agent", "description": "do stuff", "status": "completed", "duration_ms": 100}
        ],
    )
    assert trace.id is not None
    assert trace.session_id == session_id

    # Verify subtask was also written
    result = await db_session.execute(
        select(SubtaskTrace).where(SubtaskTrace.task_id == trace.id)
    )
    subtasks = result.scalars().all()
    assert len(subtasks) == 1
    assert subtasks[0].agent_id == "code_agent"

    # Clean up
    for st in subtasks:
        await db_session.delete(st)
    await db_session.delete(trace)
    await db_session.commit()


@pytest.mark.asyncio
async def test_record_feedback(db_session: AsyncSession):
    """Write feedback and verify it persists."""
    task_id = uuid.uuid4()
    fb = await record_feedback(
        db_session,
        task_id=task_id,
        rating=5,
        comment="great job",
    )
    assert fb.id is not None
    assert fb.rating == 5

    # Clean up
    await db_session.delete(fb)
    await db_session.commit()
