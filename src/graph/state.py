"""Global LangGraph state definition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Annotated, Any

from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage


def merge_timing(existing: dict[str, int] | None, new: dict[str, int] | None) -> dict[str, int]:
    """Reducer: merge timing dicts from different nodes."""
    result = dict(existing or {})
    result.update(new or {})
    return result


class TaskStatus(str, Enum):
    PENDING = "pending"
    DISPATCHING = "dispatching"
    PLANNING = "planning"
    EXECUTING = "executing"
    QUALITY_CHECK = "quality_check"
    COMPLETED = "completed"
    FAILED = "failed"


class IntentCategory(str, Enum):
    CODE = "code"
    RESEARCH = "research"
    DATA = "data"
    WRITING = "writing"
    REASONING = "reasoning"
    EXECUTE = "execute"
    GENERAL = "general"


# Legacy skill map — kept for backward compatibility with tests
INTENT_SKILL_MAP: dict[IntentCategory, list[str]] = {
    IntentCategory.CODE: ["code_gen", "code_review", "debug"],
    IntentCategory.RESEARCH: ["web_search", "summarize"],
    IntentCategory.DATA: ["data_analysis", "visualization", "statistics"],
    IntentCategory.WRITING: ["copywriting", "translation"],
    IntentCategory.REASONING: ["math", "logic", "planning"],
    IntentCategory.EXECUTE: ["shell_exec", "api_call", "file_ops"],
    IntentCategory.GENERAL: [],
}

# Intent → agent type routing (the real routing table)
INTENT_AGENT_MAP: dict[IntentCategory, str] = {
    IntentCategory.CODE: "thinker",
    IntentCategory.RESEARCH: "retriever",
    IntentCategory.DATA: "thinker",
    IntentCategory.WRITING: "thinker",
    IntentCategory.REASONING: "thinker",
    IntentCategory.EXECUTE: "executor",
    IntentCategory.GENERAL: "thinker",
}


@dataclass
class SubtaskPlan:
    id: str
    name: str
    description: str
    agent_type: str  # maps to intent category
    dependencies: list[str] = field(default_factory=list)
    result: str | None = None
    status: str = "pending"


class GraphState:
    """TypedDict-style state for LangGraph.

    Using Annotated types for reducer support where needed.
    """
    pass


# LangGraph state as a TypedDict (required by StateGraph)
from typing import TypedDict


class AgentState(TypedDict, total=False):
    """The shared state flowing through the entire LangGraph."""
    # User input
    user_input: str
    session_id: str

    # Dispatcher output
    intent: str  # IntentCategory value
    complexity: str  # "simple" | "complex"

    # DAG plan
    subtasks: list[dict[str, Any]]  # list of SubtaskPlan dicts
    current_subtask_index: int

    # Execution
    subtask_results: dict[str, str]  # subtask_id → result
    agent_outputs: Annotated[list[BaseMessage], add_messages]

    # Quality gate
    quality_passed: bool
    quality_feedback: str

    # Final
    final_output: str
    status: str  # TaskStatus value
    error: str

    # Performance monitoring
    _timing: Annotated[dict[str, int], merge_timing]
