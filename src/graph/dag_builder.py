"""Dynamic DAG builder: converts task plans into executable sequences.

Phase 1: Supports serial pipeline execution.
Phase 2: Will generate full LangGraph subgraphs with parallel branches.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from graph.state import AgentState, IntentCategory

logger = logging.getLogger(__name__)


async def plan_node(state: AgentState, llm=None) -> dict[str, Any]:
    """LangGraph node: generate a task execution plan.

    For simple tasks: single subtask directly mapped to intent.
    For complex tasks: LLM-based decomposition (or fallback to single).
    """
    user_input = state.get("user_input", "")
    intent = state.get("intent", "general")
    complexity = state.get("complexity", "simple")

    if complexity == "simple" or llm is None:
        subtasks = _plan_simple(user_input, intent)
    else:
        subtasks = await _plan_complex(user_input, intent, llm)

    return {
        "subtasks": subtasks,
        "current_subtask_index": 0,
        "subtask_results": {},
        "status": "planning",
    }


def _plan_simple(user_input: str, intent: str) -> list[dict[str, Any]]:
    """Single-subtask plan for simple tasks."""
    return [
        {
            "id": "t1",
            "name": f"Execute {intent} task",
            "description": user_input,
            "agent_type": intent,
            "dependencies": [],
            "status": "pending",
        }
    ]


async def _plan_complex(user_input: str, intent: str, llm) -> list[dict[str, Any]]:
    """LLM-based task decomposition for complex tasks."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from graph.dispatcher import load_prompt

    prompt = load_prompt("dag_planner")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["user"].format(user_input=user_input, intent=intent)),
    ]

    response = await llm.ainvoke(messages)
    try:
        result = json.loads(response.content)
        subtasks = result.get("subtasks", [])
        for st in subtasks:
            st.setdefault("status", "pending")
        return subtasks if subtasks else _plan_simple(user_input, intent)
    except json.JSONDecodeError:
        logger.warning("Failed to parse DAG planner response, using simple plan")
        return _plan_simple(user_input, intent)


def get_ready_subtasks(subtasks: list[dict], results: dict[str, str]) -> list[dict]:
    """Find subtasks whose dependencies are all satisfied."""
    ready = []
    for st in subtasks:
        if st.get("status") != "pending":
            continue
        deps = st.get("dependencies", [])
        if all(dep_id in results for dep_id in deps):
            ready.append(st)
    return ready


def all_subtasks_done(subtasks: list[dict]) -> bool:
    return all(st.get("status") in ("completed", "failed") for st in subtasks)
