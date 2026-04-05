"""ProjectPilot: iterative DAG orchestrator for large multi-step projects.

Sits above the existing ReAct loop. Decomposes a project into sub-tasks,
manages their dependency DAG, executes them via run_react(), then reviews
results and re-plans if needed — forming a feedback loop.

Execution model:
    Plan → Execute → Review → (Re-plan if needed) → Execute → ...

Convergence guards (three lines of defense):
    1. Max iteration rounds (configurable, default 3)
    2. Score stagnation detection (no improvement over N rounds → stop)
    3. Escalate to user decision when a sub-task fails repeatedly
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)

# ── Data Models ──────────────────────────────────────────────────────

DECISION_MARKER = re.compile(r"\[DECISION_NEEDED:\s*(.+?)\]", re.DOTALL)


@dataclass
class SubTask:
    """A single unit of work within a project."""
    id: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "pending"          # pending | running | completed | failed | decision_needed
    result: str | None = None
    retry_count: int = 0
    decision_prompt: str | None = None
    decision_response: str | None = None


@dataclass
class ProjectState:
    """Full persisted state for a project."""
    project_id: str
    session_id: str
    original_input: str
    subtasks: list[SubTask] = field(default_factory=list)
    iteration: int = 0
    scores: list[int] = field(default_factory=list)   # review score per iteration
    status: str = "running"          # running | paused_for_decision | completed | failed
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def completed_count(self) -> int:
        return sum(1 for t in self.subtasks if t.status == "completed")

    def total_count(self) -> int:
        return len(self.subtasks)

    def progress_pct(self) -> int:
        if not self.subtasks:
            return 0
        return int(self.completed_count() / self.total_count() * 100)


# Type alias for the event callback used by Feishu integration
OnEventCallback = Callable[[str, ProjectState, dict], Awaitable[Any]]


# ── Decomposition ────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a project planner. Break down the following project into ordered sub-tasks.

Rules:
- Each sub-task should be independently executable by an AI agent with tool access.
- Identify dependencies: which tasks must complete before others can start.
- Keep the number of tasks reasonable (3-10 for most projects).
- Return ONLY valid JSON, no explanation.

Format:
[
  {"id": "t1", "description": "...", "depends_on": []},
  {"id": "t2", "description": "...", "depends_on": ["t1"]},
  ...
]

Project:
{user_input}
"""


async def decompose_project(user_input: str, llm: Any) -> list[SubTask]:
    """Use LLM to decompose a project into ordered sub-tasks with dependencies."""
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are a project planning assistant. Return only valid JSON."),
        HumanMessage(content=_DECOMPOSE_PROMPT.format(user_input=user_input[:3000])),
    ]

    response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30)
    content = response.content.strip()

    # Strip markdown code fences if present
    if content.startswith("```"):
        lines = content.split("\n")
        content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        raw_tasks = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Decomposition returned invalid JSON, treating as single task")
        return [SubTask(id="t1", description=user_input)]

    if not isinstance(raw_tasks, list) or not raw_tasks:
        return [SubTask(id="t1", description=user_input)]

    subtasks = []
    valid_ids = {t.get("id", f"t{i+1}") for i, t in enumerate(raw_tasks)}
    for i, t in enumerate(raw_tasks):
        tid = t.get("id", f"t{i+1}")
        desc = t.get("description", "")
        deps = [d for d in t.get("depends_on", []) if d in valid_ids and d != tid]
        subtasks.append(SubTask(id=tid, description=desc, depends_on=deps))

    # Validate DAG (check for cycles)
    if _has_cycle(subtasks):
        logger.warning("Cycle detected in decomposition, removing all dependencies")
        for t in subtasks:
            t.depends_on = []

    logger.info("Project decomposed into %d sub-tasks", len(subtasks))
    return subtasks


def _has_cycle(subtasks: list[SubTask]) -> bool:
    """Check for cycles in the dependency graph via topological sort."""
    graph: dict[str, list[str]] = {t.id: list(t.depends_on) for t in subtasks}
    visited: set[str] = set()
    in_stack: set[str] = set()

    def dfs(node: str) -> bool:
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        for dep in graph.get(node, []):
            if dfs(dep):
                return True
        in_stack.discard(node)
        return False

    return any(dfs(t.id) for t in subtasks if t.id not in visited)


# ── DAG Scheduling ───────────────────────────────────────────────────

def get_ready_tasks(subtasks: list[SubTask]) -> list[SubTask]:
    """Return sub-tasks whose dependencies are all completed and status is pending."""
    completed_ids = {t.id for t in subtasks if t.status == "completed"}
    return [
        t for t in subtasks
        if t.status == "pending" and all(d in completed_ids for d in t.depends_on)
    ]


def build_subtask_context(
    subtask: SubTask,
    all_subtasks: list[SubTask],
    original_input: str,
) -> str:
    """Build the user_input string for run_react() with inter-task context."""
    parts = [f"## 项目背景\n{original_input[:1000]}"]

    # Include results from dependency tasks
    dep_results = []
    for dep_id in subtask.depends_on:
        dep = next((t for t in all_subtasks if t.id == dep_id), None)
        if dep and dep.result:
            # Truncate long results to avoid context explosion
            result_text = dep.result[:2000]
            dep_results.append(f"### 前置任务 [{dep.id}] {dep.description[:80]}\n{result_text}")

    if dep_results:
        parts.append("## 前置任务结果\n" + "\n\n".join(dep_results))

    parts.append(f"## 当前任务\n{subtask.description}")

    # If this is a retry with a decision response, include it
    if subtask.decision_response:
        parts.append(f"## 用户决策\n{subtask.decision_response}")

    return "\n\n".join(parts)


# ── Review & Re-plan ─────────────────────────────────────────────────

_REVIEW_PROMPT = """\
You are reviewing a project's progress. Analyze the completed sub-tasks and their results.

Original project: {original_input}

Completed tasks and results:
{completed_summary}

Remaining tasks:
{remaining_summary}

Evaluate:
1. Score the current progress (1-5): 1=major issues, 3=acceptable, 5=excellent
2. Are there any completed tasks that need correction?
3. Do remaining tasks need adjustment based on what was learned?

Return JSON only:
{{
  "score": 3,
  "analysis": "brief analysis",
  "corrections": [
    {{"task_id": "t1", "action": "redo", "reason": "..."}}
  ],
  "adjustments": [
    {{"task_id": "t3", "new_description": "...", "reason": "..."}}
  ],
  "new_tasks": [
    {{"id": "t_fix1", "description": "...", "depends_on": ["t1"], "insert_before": "t3"}}
  ]
}}

If everything looks good, return corrections/adjustments/new_tasks as empty arrays.
"""


async def review_iteration(
    state: ProjectState, llm: Any,
) -> dict[str, Any]:
    """LLM reviews completed work and suggests corrections/adjustments."""
    from langchain_core.messages import HumanMessage, SystemMessage

    completed = [t for t in state.subtasks if t.status == "completed"]
    remaining = [t for t in state.subtasks if t.status == "pending"]

    completed_summary = "\n".join(
        f"[{t.id}] {t.description[:100]}\nResult: {(t.result or '')[:500]}"
        for t in completed
    ) or "(none)"

    remaining_summary = "\n".join(
        f"[{t.id}] {t.description[:100]}" for t in remaining
    ) or "(none — all tasks completed)"

    prompt = _REVIEW_PROMPT.format(
        original_input=state.original_input[:1000],
        completed_summary=completed_summary,
        remaining_summary=remaining_summary,
    )

    messages = [
        SystemMessage(content="You are a project review assistant. Return only valid JSON."),
        HumanMessage(content=prompt),
    ]

    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30)
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(content)
    except Exception as e:
        logger.warning("Review failed: %s, skipping corrections", e)
        return {"score": 3, "analysis": "review skipped", "corrections": [], "adjustments": [], "new_tasks": []}


def apply_review(state: ProjectState, review: dict[str, Any], max_retries: int) -> None:
    """Apply review results: redo tasks, adjust descriptions, insert new tasks."""
    task_map = {t.id: t for t in state.subtasks}

    # Apply corrections (redo tasks)
    for corr in review.get("corrections", []):
        tid = corr.get("task_id", "")
        action = corr.get("action", "")
        if tid in task_map and action == "redo":
            task = task_map[tid]
            if task.retry_count < max_retries:
                task.status = "pending"
                task.result = None
                task.retry_count += 1
                logger.info("Task %s marked for redo (attempt %d)", tid, task.retry_count)
            else:
                logger.info("Task %s exceeded max retries (%d), skipping redo", tid, max_retries)

    # Apply adjustments (update descriptions)
    for adj in review.get("adjustments", []):
        tid = adj.get("task_id", "")
        new_desc = adj.get("new_description", "")
        if tid in task_map and new_desc and task_map[tid].status == "pending":
            task_map[tid].description = new_desc
            logger.info("Task %s description updated", tid)

    # Insert new tasks
    for new_task in review.get("new_tasks", []):
        new_id = new_task.get("id", f"t_new_{uuid.uuid4().hex[:4]}")
        if new_id not in task_map:
            deps = [d for d in new_task.get("depends_on", []) if d in task_map]
            st = SubTask(id=new_id, description=new_task.get("description", ""), depends_on=deps)
            state.subtasks.append(st)
            logger.info("New correction task inserted: %s", new_id)


# ── Checkpoint ───────────────────────────────────────────────────────

_PROJECT_PREFIX = "project:"
_PROJECT_TTL = 86400  # 24 hours


async def save_project_checkpoint(state: ProjectState) -> bool:
    """Persist project state to Redis."""
    from common.redis_client import cache_set

    state.updated_at = time.time()
    data = json.dumps({
        "project_id": state.project_id,
        "session_id": state.session_id,
        "original_input": state.original_input,
        "subtasks": [asdict(t) for t in state.subtasks],
        "iteration": state.iteration,
        "scores": state.scores,
        "status": state.status,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }, ensure_ascii=False)

    key = f"{_PROJECT_PREFIX}{state.project_id}"
    result = await cache_set(key, data, ttl=_PROJECT_TTL)
    if result:
        logger.debug("Project checkpoint saved: %s (iteration %d)", state.project_id, state.iteration)
    return result


async def load_project_checkpoint(project_id: str) -> ProjectState | None:
    """Load project state from Redis."""
    from common.redis_client import cache_get

    data = await cache_get(f"{_PROJECT_PREFIX}{project_id}")
    if not data:
        return None

    try:
        raw = json.loads(data)
        subtasks = [SubTask(**t) for t in raw.get("subtasks", [])]
        return ProjectState(
            project_id=raw["project_id"],
            session_id=raw.get("session_id", ""),
            original_input=raw["original_input"],
            subtasks=subtasks,
            iteration=raw.get("iteration", 0),
            scores=raw.get("scores", []),
            status=raw.get("status", "running"),
            created_at=raw.get("created_at", 0),
            updated_at=raw.get("updated_at", 0),
        )
    except Exception as e:
        logger.warning("Failed to load project checkpoint: %s", e)
        return None


# ── Main Execution Loop ──────────────────────────────────────────────

async def run_project(
    user_input: str,
    llm: Any,
    on_event: OnEventCallback | None = None,
    session_id: str = "",
    **react_params: Any,
) -> ProjectState:
    """Execute a project with iterative DAG + feedback loop.

    Args:
        user_input: The project description from the user.
        llm: LangChain ChatModel.
        on_event: Async callback for progress/decision events.
        session_id: Session ID for context.
        **react_params: Passed through to run_react() (qdrant, timeout, etc.).

    Returns:
        Final ProjectState with all results.
    """
    from common.config import get_settings
    from graph.orchestrator import run_react

    cfg = get_settings().project_pilot
    project_id = uuid.uuid4().hex[:12]

    # ── Step 1: Decompose ──
    if on_event:
        await on_event("project_start", ProjectState(
            project_id=project_id, session_id=session_id,
            original_input=user_input,
        ), {"phase": "decomposing"})

    subtasks = await decompose_project(user_input, llm)

    state = ProjectState(
        project_id=project_id,
        session_id=session_id,
        original_input=user_input,
        subtasks=subtasks,
    )

    logger.info("Project %s started: %d sub-tasks, max %d iterations",
                project_id, len(subtasks), cfg.max_iterations)

    if on_event:
        await on_event("project_progress", state, {"phase": "decomposed"})

    # ── Step 2: Iterative execute + review loop ──
    for iteration in range(cfg.max_iterations):
        state.iteration = iteration + 1
        await save_project_checkpoint(state)

        # Execute all ready tasks in waves until none are left
        made_progress = await _execute_wave(
            state, llm, on_event, cfg.max_parallel_subtasks, react_params,
        )

        if not made_progress:
            # No tasks could run — either all done or stuck
            break

        # Check if paused for decision
        if state.status == "paused_for_decision":
            await save_project_checkpoint(state)
            return state

        # Check if all tasks completed
        if all(t.status == "completed" for t in state.subtasks):
            break

        # ── Review phase ──
        if on_event:
            await on_event("project_progress", state, {"phase": "reviewing"})

        review = await review_iteration(state, llm)
        score = review.get("score", 3)
        state.scores.append(score)

        logger.info("Project %s iteration %d: score=%d, corrections=%d",
                    project_id, state.iteration, score,
                    len(review.get("corrections", [])))

        # Convergence check: no improvement over threshold rounds
        if len(state.scores) >= cfg.convergence_threshold:
            recent = state.scores[-cfg.convergence_threshold:]
            if all(s <= recent[0] for s in recent[1:]):
                logger.info("Project %s converged (scores stagnant: %s)", project_id, recent)
                break

        # Apply review corrections
        has_changes = (
            review.get("corrections") or
            review.get("adjustments") or
            review.get("new_tasks")
        )
        if has_changes:
            apply_review(state, review, cfg.max_subtask_retries)
            if on_event:
                await on_event("project_progress", state, {"phase": "replanned"})
        else:
            # No corrections needed — if all complete, we're done
            if all(t.status == "completed" for t in state.subtasks):
                break

    # ── Final ──
    if all(t.status == "completed" for t in state.subtasks):
        state.status = "completed"
    elif any(t.status == "failed" for t in state.subtasks):
        state.status = "failed"
    else:
        state.status = "completed"  # max iterations reached, output best effort

    await save_project_checkpoint(state)

    if on_event:
        await on_event("project_complete", state, {})

    logger.info("Project %s finished: status=%s, iterations=%d, completed=%d/%d",
                project_id, state.status, state.iteration,
                state.completed_count(), state.total_count())

    return state


async def _execute_wave(
    state: ProjectState,
    llm: Any,
    on_event: OnEventCallback | None,
    max_parallel: int,
    react_params: dict[str, Any],
) -> bool:
    """Execute waves of ready tasks until no more can proceed.

    Returns True if at least one task was executed.
    """
    from graph.orchestrator import run_react

    made_progress = False

    while True:
        ready = get_ready_tasks(state.subtasks)
        if not ready:
            break

        # Limit parallelism
        batch = ready[:max_parallel]

        for t in batch:
            t.status = "running"
        if on_event:
            await on_event("project_progress", state, {
                "phase": "executing",
                "running": [t.id for t in batch],
            })

        # Execute batch concurrently
        results = await asyncio.gather(
            *[_execute_subtask(t, state, llm, react_params) for t in batch],
            return_exceptions=True,
        )

        for task, result in zip(batch, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = f"Error: {result}"
                logger.error("Sub-task %s failed: %s", task.id, result)
            else:
                output = result.get("final_output", "")
                # Check for decision marker
                decision_match = DECISION_MARKER.search(output)
                if decision_match:
                    task.status = "decision_needed"
                    task.decision_prompt = decision_match.group(1)
                    task.result = output
                    state.status = "paused_for_decision"
                    logger.info("Sub-task %s needs user decision: %s", task.id, task.decision_prompt[:80])
                elif result.get("status") == "completed":
                    task.status = "completed"
                    task.result = output
                else:
                    task.status = "failed"
                    task.result = output

            made_progress = True

        if on_event:
            await on_event("project_progress", state, {"phase": "batch_done"})

        # If paused for decision, stop executing
        if state.status == "paused_for_decision":
            break

        await save_project_checkpoint(state)

    return made_progress


async def _execute_subtask(
    subtask: SubTask,
    state: ProjectState,
    llm: Any,
    react_params: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single sub-task via run_react()."""
    from graph.orchestrator import run_react

    user_input = build_subtask_context(subtask, state.subtasks, state.original_input)

    result = await run_react(
        user_input=user_input,
        llm=react_params.get("llm", llm),
        qdrant=react_params.get("qdrant"),
        collection_name=react_params.get("collection_name", "case_library"),
        timeout=react_params.get("timeout", 0),
    )

    return result


# ── Resume ───────────────────────────────────────────────────────────

async def resume_project(
    project_id: str,
    decision_response: str | None = None,
    llm: Any = None,
    on_event: OnEventCallback | None = None,
    **react_params: Any,
) -> ProjectState | None:
    """Resume a paused/interrupted project from checkpoint.

    Args:
        project_id: The project to resume.
        decision_response: User's answer if paused for decision.
        llm: LangChain ChatModel.
        on_event: Progress callback.
        **react_params: Passed through to run_react().

    Returns:
        Updated ProjectState, or None if project not found.
    """
    from common.config import get_settings

    state = await load_project_checkpoint(project_id)
    if not state:
        logger.warning("Project %s not found in checkpoint", project_id)
        return None

    cfg = get_settings().project_pilot

    # Apply decision response
    if decision_response and state.status == "paused_for_decision":
        for t in state.subtasks:
            if t.status == "decision_needed":
                t.decision_response = decision_response
                t.status = "pending"  # retry with user's input
                break
        state.status = "running"

    if on_event:
        await on_event("project_progress", state, {"phase": "resumed"})

    # Continue the iteration loop from where we left off
    remaining_iterations = cfg.max_iterations - state.iteration
    for iteration in range(max(1, remaining_iterations)):
        state.iteration += 1
        await save_project_checkpoint(state)

        made_progress = await _execute_wave(
            state, llm, on_event, cfg.max_parallel_subtasks, react_params,
        )

        if not made_progress:
            break

        if state.status == "paused_for_decision":
            await save_project_checkpoint(state)
            return state

        if all(t.status == "completed" for t in state.subtasks):
            break

        review = await review_iteration(state, llm)
        score = review.get("score", 3)
        state.scores.append(score)

        if len(state.scores) >= cfg.convergence_threshold:
            recent = state.scores[-cfg.convergence_threshold:]
            if all(s <= recent[0] for s in recent[1:]):
                break

        has_changes = (
            review.get("corrections") or
            review.get("adjustments") or
            review.get("new_tasks")
        )
        if has_changes:
            apply_review(state, review, cfg.max_subtask_retries)

    # Final status
    if all(t.status == "completed" for t in state.subtasks):
        state.status = "completed"
    elif any(t.status == "failed" for t in state.subtasks):
        state.status = "failed"
    else:
        state.status = "completed"

    await save_project_checkpoint(state)
    if on_event:
        await on_event("project_complete", state, {})

    return state


# ── Result Formatting ────────────────────────────────────────────────

def format_project_result(state: ProjectState) -> str:
    """Format the final project output as a structured summary."""
    parts = [f"## 执行概要\n"]
    parts.append(f"- 项目状态: {state.status}")
    parts.append(f"- 总任务数: {state.total_count()}, 完成: {state.completed_count()}")
    parts.append(f"- 迭代轮次: {state.iteration}")
    if state.scores:
        parts.append(f"- 审查评分: {' → '.join(str(s) for s in state.scores)}")

    parts.append("\n## 各任务结果\n")
    for t in state.subtasks:
        icon = {"completed": "✅", "failed": "❌", "pending": "⬜", "running": "🔄"}.get(t.status, "❓")
        parts.append(f"### {icon} [{t.id}] {t.description[:80]}")
        if t.result:
            parts.append(t.result[:1500])
        if t.retry_count > 0:
            parts.append(f"_(重试 {t.retry_count} 次)_")
        parts.append("")

    return "\n".join(parts)
