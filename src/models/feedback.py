"""Feedback models."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from common.db import Base


class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), index=True)
    rating: Mapped[int | None] = mapped_column(Integer)  # 1-5
    comment: Mapped[str | None] = mapped_column(Text)
    feedback_type: Mapped[str] = mapped_column(String(32), default="explicit")  # explicit/implicit
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
