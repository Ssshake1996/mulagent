"""Tests for SQLAlchemy model definitions."""

import uuid

from models.trace import TaskTrace, SubtaskTrace
from models.feedback import Feedback
from common.db import Base


def test_task_trace_table_name():
    assert TaskTrace.__tablename__ == "task_traces"


def test_subtask_trace_table_name():
    assert SubtaskTrace.__tablename__ == "subtask_traces"


def test_feedback_table_name():
    assert Feedback.__tablename__ == "feedbacks"


def test_base_has_all_tables():
    table_names = set(Base.metadata.tables.keys())
    assert "task_traces" in table_names
    assert "subtask_traces" in table_names
    assert "feedbacks" in table_names


def test_task_trace_columns():
    cols = {c.name for c in TaskTrace.__table__.columns}
    assert "id" in cols
    assert "session_id" in cols
    assert "user_input" in cols
    assert "dag_plan" in cols
    assert "status" in cols


def test_subtask_trace_has_fk():
    fks = [fk.target_fullname for fk in SubtaskTrace.__table__.foreign_keys]
    assert "task_traces.id" in fks
