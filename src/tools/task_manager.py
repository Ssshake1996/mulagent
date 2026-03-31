"""Task management and planning tools.

- todo_manage: Create, track, and complete tasks within the ReAct loop
- plan_submit: Submit an execution plan for user approval before proceeding

These replace the pure-text todolist approach with structured tools.

Replaces the pure-text todolist approach with a structured tool that the LLM
can call to explicitly manage task progress. Tasks are stored in WorkingMemory.state
and rendered in the context message.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

_TASKS_KEY = "_tasks"


def _get_tasks(deps: dict[str, Any]) -> list[dict[str, Any]]:
    """Get task list from WorkingMemory state."""
    memory = deps.get("memory")
    if memory is None:
        return []
    return memory.state.get(_TASKS_KEY, [])


def _set_tasks(deps: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
    """Save task list to WorkingMemory state."""
    memory = deps.get("memory")
    if memory is not None:
        memory.update_state(_TASKS_KEY, tasks)


async def _todo_manage(params: dict[str, Any], **deps: Any) -> str:
    """Task management: create, update, list, or check off tasks."""
    action = params.get("action", "list")
    tasks = _get_tasks(deps)

    if action == "create":
        items = params.get("items", [])
        if not items:
            return "Error: items is required (list of task descriptions)"

        # If tasks already exist, keep completed ones and replace pending/in_progress
        # This prevents duplicate task lists when LLM re-plans after loop detection
        done_tasks = [t for t in tasks if t.get("status") == "done"]
        if done_tasks:
            # Preserve completed tasks, start new IDs after the highest existing ID
            max_id = max(t["id"] for t in done_tasks)
        else:
            max_id = 0

        new_tasks = list(done_tasks)  # keep completed tasks
        for i, item in enumerate(items, start=max_id + 1):
            task = {
                "id": i,
                "text": item if isinstance(item, str) else str(item),
                "status": "pending",
                "created_at": time.time(),
            }
            new_tasks.append(task)
        tasks = new_tasks
        _set_tasks(deps, tasks)
        replaced = " (replaced pending tasks)" if done_tasks else ""
        return f"Created {len(items)} tasks{replaced}. Total: {len(tasks)}\n" + _format_tasks(tasks)

    elif action == "done":
        task_id = params.get("task_id")
        if task_id is None:
            return "Error: task_id is required"
        for t in tasks:
            if t["id"] == task_id:
                t["status"] = "done"
                t["done_at"] = time.time()
                _set_tasks(deps, tasks)
                return f"✓ Task #{task_id} marked done.\n" + _format_tasks(tasks)
        return f"Error: task #{task_id} not found"

    elif action == "update":
        task_id = params.get("task_id")
        new_text = params.get("text", "")
        new_status = params.get("status", "")
        if task_id is None:
            return "Error: task_id is required"
        for t in tasks:
            if t["id"] == task_id:
                if new_text:
                    t["text"] = new_text
                if new_status:
                    t["status"] = new_status
                _set_tasks(deps, tasks)
                return f"Updated task #{task_id}.\n" + _format_tasks(tasks)
        return f"Error: task #{task_id} not found"

    elif action == "list":
        if not tasks:
            return "No tasks yet. Use action='create' with items=[...] to plan your work."
        return _format_tasks(tasks)

    else:
        return f"Unknown action: {action}. Use: create, done, update, list"


def _format_tasks(tasks: list[dict]) -> str:
    """Format task list for display."""
    lines = []
    done = sum(1 for t in tasks if t["status"] == "done")
    lines.append(f"[Progress: {done}/{len(tasks)}]")
    for t in tasks:
        icon = "✓" if t["status"] == "done" else "○"
        lines.append(f"  {icon} #{t['id']} {t['text']}")
    return "\n".join(lines)


TODO_MANAGE = ToolDef(
    name="todo_manage",
    description=(
        "Manage your task list. Use this to plan multi-step work: "
        "create a task list first, then mark tasks done as you complete them. "
        "Actions: create (with items list), done (with task_id), update, list."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "create | done | update | list",
                "enum": ["create", "done", "update", "list"],
            },
            "items": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Task descriptions (for action=create)",
            },
            "task_id": {
                "type": "integer",
                "description": "Task ID (for action=done or action=update)",
            },
            "text": {
                "type": "string",
                "description": "New text (for action=update)",
            },
            "status": {
                "type": "string",
                "description": "New status (for action=update): pending | in_progress | done | skipped",
            },
        },
        "required": ["action"],
    },
    fn=_todo_manage,
)


# ── Plan submission tool ──��─────────────────────────────────────

# Sentinel value: when the react_loop sees this in a tool result,
# it pauses and returns the plan to the caller for user approval.
PLAN_PENDING_MARKER = "##PLAN_PENDING##"


async def _plan_submit(params: dict[str, Any], **deps: Any) -> str:
    """Submit an execution plan for user review before proceeding.

    Use this for high-impact tasks where the user should review the approach first.
    The plan will be shown to the user who can approve, reject, or modify it.
    """
    plan = params.get("plan", "")
    if not plan:
        return "Error: plan is required"

    risk_level = params.get("risk_level", "medium")
    estimated_steps = params.get("estimated_steps", 0)

    # Store plan in memory state for the orchestrator to detect
    memory = deps.get("memory")
    if memory is not None:
        memory.update_state("_pending_plan", {
            "plan": plan,
            "risk_level": risk_level,
            "estimated_steps": estimated_steps,
        })

    return (
        f"{PLAN_PENDING_MARKER}\n"
        f"**执行计划** (风险: {risk_level})\n\n"
        f"{plan}\n\n"
        f"预计步骤: {estimated_steps or '未指定'}\n\n"
        f"等待用户确认..."
    )


PLAN_SUBMIT = ToolDef(
    name="plan_submit",
    description=(
        "Submit an execution plan for user approval before proceeding. "
        "Use this for high-risk tasks (bulk changes, external system operations, complex refactoring). "
        "The system will pause and show the plan to the user. "
        "Do NOT use this for simple/low-risk tasks — just execute directly."
    ),
    parameters={
        "type": "object",
        "properties": {
            "plan": {
                "type": "string",
                "description": "The execution plan in structured text (numbered steps)",
            },
            "risk_level": {
                "type": "string",
                "description": "low | medium | high",
                "enum": ["low", "medium", "high"],
            },
            "estimated_steps": {
                "type": "integer",
                "description": "Estimated number of execution steps",
            },
        },
        "required": ["plan"],
    },
    fn=_plan_submit,
)
