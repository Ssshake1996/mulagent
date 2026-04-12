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


def _llm_timeout() -> int:
    """Derive LLM call timeout from react.timeout. Floor 30s."""
    try:
        from common.config import get_settings
        return max(get_settings().react.timeout // 20, 30)
    except Exception:
        return 30


# ── Data Models ────────────────────────────────────────────────────���─

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
    complexity: str = "medium"       # light | medium | heavy — controls rounds budget


@dataclass
class ProjectMemory:
    """Cross-stage structured memory — distilled knowledge, not raw results."""
    decisions: list[str] = field(default_factory=list)       # key decisions made
    constraints: list[str] = field(default_factory=list)     # discovered constraints
    artifacts: dict[str, str] = field(default_factory=dict)  # path/name → brief desc
    lessons: list[str] = field(default_factory=list)         # failure lessons


@dataclass
class ProjectState:
    """Full persisted state for a project."""
    project_id: str
    session_id: str
    original_input: str
    acceptance_criteria: str = ""     # clarified goal + measurable acceptance criteria
    subtasks: list[SubTask] = field(default_factory=list)
    iteration: int = 0
    scores: list[int] = field(default_factory=list)   # review score per iteration
    status: str = "running"          # running | plan_review | paused_for_decision | completed | failed
    project_memory: ProjectMemory = field(default_factory=ProjectMemory)
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


# ── Goal Clarification ───────────────────────────────────────────────

_CLARIFY_GOAL_PROMPT = """\
The user has described a project goal below. It may be vague or incomplete.

Your job:
1. Interpret the user's intent and produce a clear, structured set of acceptance criteria.
2. Assess how confident you are that your interpretation is correct (confidence 1-5).

Rules:
- Infer what the user most likely wants, even if they didn't state it explicitly.
- Each criterion should be concrete and verifiable (not "make it good", but "API returns 200 on /health").
- Cover: functional requirements, key constraints, and definition of done.
- Use the same language as the user's input.
- confidence 5 = user input is unambiguous, has clear steps/specs, no room for misinterpretation.
- confidence 4 = mostly clear, minor assumptions needed.
- confidence 3 = some ambiguity, multiple valid interpretations exist.
- confidence 1-2 = very vague, significant guesswork required.
- Return ONLY valid JSON, no explanation.

Format:
{{
  "clarified_goal": "one paragraph restating what the user actually wants to achieve",
  "acceptance_criteria": [
    "criterion 1: ...",
    "criterion 2: ...",
    ...
  ],
  "confidence": 4,
  "ambiguities": ["optional: list of unclear points that could affect execution"]
}}

User's input:
{user_input}
"""


async def clarify_goal(user_input: str, llm: Any) -> tuple[str, list[str], int, list[str]]:
    """Clarify a potentially vague user goal into measurable acceptance criteria.

    Returns (clarified_goal_text, list_of_criteria, confidence_1_to_5, ambiguities).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content="You are a requirements analyst. Return only valid JSON."),
        HumanMessage(content=_CLARIFY_GOAL_PROMPT.format(user_input=user_input[:3000])),
    ]

    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=_llm_timeout())
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(content)
        goal = result.get("clarified_goal", user_input)
        criteria = result.get("acceptance_criteria", [])
        confidence = int(result.get("confidence", 3))
        ambiguities = result.get("ambiguities", [])
        if not criteria:
            return goal, [f"完成用户要求: {user_input[:200]}"], confidence, ambiguities
        logger.info("Goal clarified: %d criteria, confidence=%d, ambiguities=%d",
                     len(criteria), confidence, len(ambiguities))
        return goal, criteria, confidence, ambiguities
    except Exception as e:
        logger.warning("Goal clarification failed: %s, using original input", e)
        return user_input, [f"完成用户要求: {user_input[:200]}"], 3, []


def _format_plan_for_review(state: "ProjectState") -> str:
    """Format project plan as a human-readable summary for user review."""
    parts = [f"## 目标理解\n{state.acceptance_criteria}"]
    parts.append(f"\n## 执行计划（共 {len(state.subtasks)} 个子任务）\n")
    for i, t in enumerate(state.subtasks, 1):
        deps = f" (依赖: {', '.join(t.depends_on)})" if t.depends_on else ""
        complexity = getattr(t, "complexity", "medium")
        parts.append(f"{i}. **[{t.id}]** {t.description}{deps} `[{complexity}]`")
    return "\n".join(parts)


# ── Decomposition ────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """\
You are a project planner. Break down the following project into ordered sub-tasks.

## Clarified Goal
{clarified_goal}

## Acceptance Criteria (every sub-task must contribute to meeting these)
{acceptance_criteria}

Rules:
- Every sub-task must directly serve one or more acceptance criteria above. Do NOT add "nice to have" tasks.
- Each sub-task should be independently executable by an AI agent with tool access.
- Each sub-task description must clearly state what outcome is expected.
- Identify dependencies: which tasks must complete before others can start.
- Keep the number of tasks reasonable (3-10 for most projects).
- Assess each task's complexity: "light" (simple lookup/read), "medium" (moderate work), "heavy" (multi-step, lots of code/files).
- Return ONLY valid JSON, no explanation.

Format:
[
  {{"id": "t1", "description": "...", "depends_on": [], "complexity": "medium"}},
  {{"id": "t2", "description": "...", "depends_on": ["t1"], "complexity": "heavy"}},
  ...
]
"""


async def decompose_project(
    user_input: str, llm: Any,
    clarified_goal: str = "", acceptance_criteria: list[str] | None = None,
) -> list[SubTask]:
    """Use LLM to decompose a project into ordered sub-tasks with dependencies."""
    from langchain_core.messages import HumanMessage, SystemMessage

    goal_text = clarified_goal or user_input
    criteria_text = "\n".join(f"- {c}" for c in (acceptance_criteria or [])) or "(none specified)"

    messages = [
        SystemMessage(content="You are a project planning assistant. Return only valid JSON."),
        HumanMessage(content=_DECOMPOSE_PROMPT.format(
            clarified_goal=goal_text[:2000],
            acceptance_criteria=criteria_text,
        )),
    ]

    response = await asyncio.wait_for(llm.ainvoke(messages), timeout=_llm_timeout())
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
    _valid_complexity = {"light", "medium", "heavy"}
    for i, t in enumerate(raw_tasks):
        tid = t.get("id", f"t{i+1}")
        desc = t.get("description", "")
        deps = [d for d in t.get("depends_on", []) if d in valid_ids and d != tid]
        complexity = t.get("complexity", "medium")
        if complexity not in _valid_complexity:
            complexity = "medium"
        subtasks.append(SubTask(id=tid, description=desc, depends_on=deps,
                                complexity=complexity))

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


async def _extract_subtask_knowledge(
    result: str, subtask: SubTask, llm: Any,
) -> dict[str, Any]:
    """Extract cross-stage knowledge from a completed sub-task result.

    Returns dict with keys: decisions, constraints, artifacts, lessons.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "从以下任务执行结果中提取跨阶段关键信息。返回 JSON:\n"
            '{"decisions": ["做出的关键决策"], '
            '"constraints": ["发现的约束/限制"], '
            '"artifacts": {"文件路径或名称": "简要说明"}, '
            '"lessons": ["失败教训或需注意的点"]}\n'
            "只提取对后续任务有用的信息，忽略临时细节。没有则返回空数组/对象��"
        )),
        HumanMessage(content=(
            f"任务: {subtask.description[:200]}\n\n"
            f"��行结果:\n{result[:2000]}"
        )),
    ]

    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=_llm_timeout())
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(content)
    except Exception as e:
        logger.debug("Knowledge extraction failed for %s: %s", subtask.id, e)
        return {}


def _merge_knowledge(pm: ProjectMemory, knowledge: dict[str, Any]) -> None:
    """Merge extracted knowledge into ProjectMemory."""
    for d in knowledge.get("decisions", []):
        if d and d not in pm.decisions:
            pm.decisions.append(d)
    for c in knowledge.get("constraints", []):
        if c and c not in pm.constraints:
            pm.constraints.append(c)
    for path, desc in knowledge.get("artifacts", {}).items():
        pm.artifacts[path] = desc
    for lesson in knowledge.get("lessons", []):
        if lesson and lesson not in pm.lessons:
            pm.lessons.append(lesson)


def _format_project_memory(pm: ProjectMemory) -> str:
    """Format ProjectMemory as context text for sub-task injection."""
    parts = []
    if pm.decisions:
        parts.append("**关键决策**: " + "; ".join(pm.decisions[-8:]))
    if pm.constraints:
        parts.append("**已知约束**: " + "; ".join(pm.constraints[-5:]))
    if pm.artifacts:
        items = [f"`{k}`: {v}" for k, v in list(pm.artifacts.items())[-8:]]
        parts.append("**产出物**: " + "; ".join(items))
    if pm.lessons:
        parts.append("**教训**: " + "; ".join(pm.lessons[-5:]))
    return "\n".join(parts)


def build_subtask_context(
    subtask: SubTask,
    all_subtasks: list[SubTask],
    original_input: str,
    project_memory: ProjectMemory | None = None,
) -> str:
    """Build the user_input string for run_react() with inter-task context."""
    parts = [f"## 项目背景\n{original_input[:1000]}"]

    # Inject cross-stage project memory (distilled, not raw results)
    if project_memory:
        mem_text = _format_project_memory(project_memory)
        if mem_text:
            parts.append(f"## 项目记忆（前序阶段的关键��息）\n{mem_text}")

    # Include results from direct dependency tasks (truncated)
    dep_results = []
    for dep_id in subtask.depends_on:
        dep = next((t for t in all_subtasks if t.id == dep_id), None)
        if dep and dep.result:
            result_text = dep.result[:1500]
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
You are reviewing a project's progress. Your ONLY evaluation standard is the acceptance criteria below.

## Clarified Goal
{clarified_goal}

## Acceptance Criteria (your evaluation checklist)
{acceptance_criteria}

## Completed Tasks and Results
{completed_summary}

## Remaining Tasks
{remaining_summary}

## Evaluation Criteria (all measured against the user's original goal)
1. **Criteria coverage** (1-5): How many acceptance criteria are being met by the completed work? \
1=none met, 3=some met, 5=all met or on track to be met.
2. **Correction needed?**: Do any completed tasks need to be redone because their output \
fails to satisfy the acceptance criteria?
3. **Plan adjustment?**: Based on what we've learned, do remaining tasks need to be modified \
to better satisfy the acceptance criteria?

IMPORTANT: Do NOT evaluate tasks in isolation. A task that "completed successfully" but \
produced results that don't advance any acceptance criterion should score LOW and be flagged for correction.

Return JSON only:
{{
  "score": 3,
  "analysis": "brief analysis of how well current progress serves the user's goal",
  "corrections": [
    {{"task_id": "t1", "action": "redo", "reason": "why this diverges from the user's goal"}}
  ],
  "adjustments": [
    {{"task_id": "t3", "new_description": "...", "reason": "how this better serves the goal"}}
  ],
  "new_tasks": [
    {{"id": "t_fix1", "description": "...", "depends_on": ["t1"], "insert_before": "t3"}}
  ]
}}

If everything is well-aligned with the user's goal, return corrections/adjustments/new_tasks as empty arrays.
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

    # Parse acceptance criteria from state
    criteria_text = state.acceptance_criteria or f"完成用户要求: {state.original_input[:500]}"

    prompt = _REVIEW_PROMPT.format(
        clarified_goal=state.acceptance_criteria.split("\n")[0] if state.acceptance_criteria else state.original_input[:500],
        acceptance_criteria=criteria_text,
        completed_summary=completed_summary,
        remaining_summary=remaining_summary,
    )

    messages = [
        SystemMessage(content="You are a project review assistant. Return only valid JSON."),
        HumanMessage(content=prompt),
    ]

    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=_llm_timeout())
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
        "acceptance_criteria": state.acceptance_criteria,
        "subtasks": [asdict(t) for t in state.subtasks],
        "iteration": state.iteration,
        "scores": state.scores,
        "status": state.status,
        "project_memory": asdict(state.project_memory),
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
        # Restore ProjectMemory
        pm_raw = raw.get("project_memory", {})
        pm = ProjectMemory(
            decisions=pm_raw.get("decisions", []),
            constraints=pm_raw.get("constraints", []),
            artifacts=pm_raw.get("artifacts", {}),
            lessons=pm_raw.get("lessons", []),
        )
        return ProjectState(
            project_id=raw["project_id"],
            session_id=raw.get("session_id", ""),
            original_input=raw["original_input"],
            acceptance_criteria=raw.get("acceptance_criteria", ""),
            subtasks=subtasks,
            iteration=raw.get("iteration", 0),
            scores=raw.get("scores", []),
            status=raw.get("status", "running"),
            project_memory=pm,
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

    # ── Step 1: Clarify goal → measurable acceptance criteria ──
    if on_event:
        await on_event("project_start", ProjectState(
            project_id=project_id, session_id=session_id,
            original_input=user_input,
        ), {"phase": "clarifying_goal"})

    clarified_goal, criteria, confidence, ambiguities = await clarify_goal(user_input, llm)
    acceptance_text = f"{clarified_goal}\n\n验收标准:\n" + "\n".join(f"- {c}" for c in criteria)
    logger.info("Goal clarified: %s, %d criteria, confidence=%d",
                clarified_goal[:80], len(criteria), confidence)

    # ── Step 2: Decompose based on clarified goal ──
    if on_event:
        await on_event("project_progress", ProjectState(
            project_id=project_id, session_id=session_id,
            original_input=user_input, acceptance_criteria=acceptance_text,
        ), {"phase": "decomposing"})

    subtasks = await decompose_project(user_input, llm, clarified_goal, criteria)

    state = ProjectState(
        project_id=project_id,
        session_id=session_id,
        original_input=user_input,
        acceptance_criteria=acceptance_text,
        subtasks=subtasks,
    )

    logger.info("Project %s started: %d sub-tasks, max %d iterations",
                project_id, len(subtasks), cfg.max_iterations)

    if on_event:
        await on_event("project_progress", state, {"phase": "decomposed"})

    # ── Step 2.5: Plan review — only when goal is ambiguous ──
    # confidence >= 4 means user intent is clear → skip review, execute directly.
    # confidence < 4 means ambiguity detected → pause for user confirmation.
    _review_threshold = getattr(cfg, "plan_review_confidence", 4)
    if confidence < _review_threshold:
        state.status = "plan_review"
        plan_summary = _format_plan_for_review(state)
        ambiguity_text = ""
        if ambiguities:
            ambiguity_text = "\n\n需要澄清的点:\n" + "\n".join(f"- {a}" for a in ambiguities)

        if on_event:
            await on_event("plan_review", state, {
                "plan_summary": plan_summary + ambiguity_text,
                "confidence": confidence,
                "message": "以下是我理解的目标和执行计划，请确认或修改后继续：",
            })
        await save_project_checkpoint(state)
        logger.info("Project %s paused for plan review (confidence=%d < %d)",
                     project_id, confidence, _review_threshold)
        return state
    else:
        logger.info("Project %s skipping plan review (confidence=%d >= %d)",
                     project_id, confidence, _review_threshold)

    # ── Step 3: Iterative execute + review loop ──
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
                    # Extract cross-stage knowledge into ProjectMemory
                    try:
                        knowledge = await _extract_subtask_knowledge(output, task, llm)
                        if knowledge:
                            _merge_knowledge(state.project_memory, knowledge)
                            logger.info("Extracted knowledge from %s: %d decisions, %d constraints",
                                        task.id, len(knowledge.get("decisions", [])),
                                        len(knowledge.get("constraints", [])))
                    except Exception as _ke:
                        logger.debug("Knowledge extraction skipped for %s: %s", task.id, _ke)
                else:
                    task.status = "failed"
                    task.result = output

            made_progress = True

        # ── Failure fast-feedback: propagate errors to upstream producers ──
        _failed_in_batch = [t for t in batch if t.status == "failed"]
        if _failed_in_batch:
            from common.config import get_settings as _gsc
            _max_retries = _gsc().project_pilot.max_subtask_retries
            for failed_task in _failed_in_batch:
                # Find upstream tasks that this task depends on
                for dep_id in failed_task.depends_on:
                    upstream = next((t for t in state.subtasks if t.id == dep_id), None)
                    if upstream and upstream.status == "completed" and upstream.retry_count < _max_retries:
                        upstream.status = "pending"
                        upstream.retry_count += 1
                        upstream.decision_response = (
                            f"你上次的产出导致下游任务 [{failed_task.id}] 失败。"
                            f"错误信息: {(failed_task.result or '')[:800]}\n"
                            f"请修复后重新输出。"
                        )
                        logger.info("Fast-feedback: upstream %s marked for redo "
                                    "(downstream %s failed, attempt %d)",
                                    upstream.id, failed_task.id, upstream.retry_count)
                        # Also record lesson in project memory
                        state.project_memory.lessons.append(
                            f"[{upstream.id}] 的产出导致 [{failed_task.id}] 失败: "
                            f"{(failed_task.result or '')[:200]}"
                        )
                # Reset the failed task itself for retry after upstream fixes
                if failed_task.depends_on and failed_task.retry_count < _max_retries:
                    failed_task.status = "pending"
                    failed_task.retry_count += 1

        if on_event:
            await on_event("project_progress", state, {"phase": "batch_done"})

        # If paused for decision, stop executing
        if state.status == "paused_for_decision":
            break

        await save_project_checkpoint(state)

    return made_progress


# Complexity → timeout/rounds multiplier (relative to config base values)
_COMPLEXITY_MULTIPLIER = {
    "light":  (0.3, 0.3),   # 30% of base timeout/rounds
    "medium": (0.6, 0.5),   # 60% timeout, 50% rounds
    "heavy":  (1.0, 0.8),   # 100% timeout, 80% rounds
}


async def _execute_subtask(
    subtask: SubTask,
    state: ProjectState,
    llm: Any,
    react_params: dict[str, Any],
) -> dict[str, Any]:
    """Execute a single sub-task via run_react()."""
    from graph.orchestrator import run_react

    user_input = build_subtask_context(
        subtask, state.subtasks, state.original_input,
        project_memory=state.project_memory,
    )

    # Complexity-aware timeout: scale from config base
    timeout_mult, rounds_mult = _COMPLEXITY_MULTIPLIER.get(
        subtask.complexity, (0.6, 0.5))
    try:
        from common.config import get_settings
        _cfg = get_settings().react
        sub_timeout = max(int(_cfg.timeout * timeout_mult), 120)
        sub_rounds = max(int(_cfg.max_rounds * rounds_mult), 8)
    except Exception:
        sub_timeout = react_params.get("timeout", 0)
        sub_rounds = 0  # let run_react use default

    logger.info("Sub-task %s [%s] → timeout=%ds, max_rounds=%d",
                subtask.id, subtask.complexity, sub_timeout, sub_rounds)

    # Build kwargs — only override if we got valid values
    kwargs: dict[str, Any] = {
        "user_input": user_input,
        "llm": react_params.get("llm", llm),
        "qdrant": react_params.get("qdrant"),
        "collection_name": react_params.get("collection_name", "case_library"),
        "timeout": sub_timeout,
    }

    result = await run_react(**kwargs)

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

    # Apply plan review confirmation or decision response
    if state.status == "plan_review":
        # User confirmed (or modified) the plan → proceed to execution
        if decision_response:
            # User may have provided modifications — re-clarify if non-trivial
            logger.info("Plan review confirmed with response: %s", decision_response[:80])
        state.status = "running"

    elif decision_response and state.status == "paused_for_decision":
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
    if state.acceptance_criteria:
        parts.append(f"\n## 验收标准\n{state.acceptance_criteria}")

    # Project memory summary
    pm = state.project_memory
    if pm.decisions or pm.constraints or pm.lessons:
        parts.append("\n## 项目记忆\n")
        if pm.decisions:
            parts.append("**关键决策**: " + "; ".join(pm.decisions[-5:]))
        if pm.constraints:
            parts.append("**发现的约束**: " + "; ".join(pm.constraints[-5:]))
        if pm.artifacts:
            items = [f"`{k}`: {v}" for k, v in list(pm.artifacts.items())[-5:]]
            parts.append("**产出物**: " + "; ".join(items))
        if pm.lessons:
            parts.append("**教训**: " + "; ".join(pm.lessons[-3:]))

    parts.append("\n## 各任务结果\n")
    for t in state.subtasks:
        icon = {"completed": "✅", "failed": "❌", "pending": "⬜", "running": "🔄"}.get(t.status, "❓")
        parts.append(f"### {icon} [{t.id}] {t.description[:80]} `[{t.complexity}]`")
        if t.result:
            parts.append(t.result[:1500])
        if t.retry_count > 0:
            parts.append(f"_(重试 {t.retry_count} 次)_")
        parts.append("")

    return "\n".join(parts)
