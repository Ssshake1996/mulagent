"""Feedback collector — captures explicit and implicit user feedback."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from models.feedback import Feedback


async def record_feedback(
    session: AsyncSession,
    *,
    task_id: uuid.UUID,
    rating: int | None = None,
    comment: str | None = None,
    feedback_type: str = "explicit",
) -> Feedback:
    """Record user feedback for a task."""
    fb = Feedback(
        id=uuid.uuid4(),
        task_id=task_id,
        rating=rating,
        comment=comment,
        feedback_type=feedback_type,
    )
    session.add(fb)
    await session.commit()
    return fb
