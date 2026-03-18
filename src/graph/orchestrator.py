"""Main orchestrator: assembles the LangGraph that ties everything together.

Flow: dispatch → plan → execute (loop) → quality check → done
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from agents.adapter import AdapterFactory
from agents.registry import AgentRegistry
from agents.skill_acquirer import SkillAcquirer
from common.timing import timed_node
from graph.dag_builder import all_subtasks_done, get_ready_subtasks, plan_node
from graph.dispatcher import dispatch_node
from graph.quality_gate import quality_check_node
from graph.state import AgentState, INTENT_AGENT_MAP, IntentCategory

logger = logging.getLogger(__name__)


async def execute_node(
    state: AgentState,
    registry: AgentRegistry | None = None,
    adapter_factory: AdapterFactory | None = None,
    skill_acquirer: SkillAcquirer | None = None,
    on_subtask_progress=None,
) -> dict[str, Any]:
    """Execute the next ready subtask(s) in the plan.

    Args:
        on_subtask_progress: Optional async callback(subtask_name, status, index, total).
            Called when each subtask starts/completes. Used by feishu_bot for live updates.
    """
    subtasks = list(state.get("subtasks", []))
    results = dict(state.get("subtask_results", {}))
    index = state.get("current_subtask_index", 0)

    if not subtasks:
        return {
            "final_output": "No subtasks to execute.",
            "status": "failed",
        }

    ready = get_ready_subtasks(subtasks, results)
    if not ready:
        # All done or stuck
        return _finalize(subtasks, results)

    total = len(subtasks)
    for i, subtask in enumerate(ready):
        agent_type = subtask.get("agent_type", "general")
        task_desc = subtask.get("description", subtask.get("name", ""))
        subtask_name = subtask.get("name", subtask.get("id", ""))

        # Notify: subtask starting
        if on_subtask_progress:
            try:
                await on_subtask_progress(subtask_name, "running", i + 1, total)
            except Exception:
                pass

        # Gather context from dependencies
        context = {}
        for dep_id in subtask.get("dependencies", []):
            if dep_id in results:
                context[dep_id] = results[dep_id]

        # Try skill acquisition to enrich context
        if skill_acquirer is not None:
            acquisition = await skill_acquirer.acquire(task_desc, [agent_type])
            if acquisition.found:
                context["_skill_context"] = acquisition.content
                logger.info("Skill acquired via %s for subtask %s", acquisition.level.value, subtask.get("id"))

        # Execute via adapter
        output = await _run_agent(agent_type, task_desc, context, registry, adapter_factory)
        subtask["status"] = "completed"
        subtask["result"] = output
        results[subtask["id"]] = output

        # Notify: subtask done
        if on_subtask_progress:
            try:
                await on_subtask_progress(subtask_name, "done", i + 1, total)
            except Exception:
                pass

    if all_subtasks_done(subtasks):
        return _finalize(subtasks, results)

    return {
        "subtasks": subtasks,
        "subtask_results": results,
        "current_subtask_index": index + 1,
        "status": "executing",
    }


async def _run_agent(
    agent_type: str,
    task: str,
    context: dict[str, str],
    registry: AgentRegistry | None,
    factory: AdapterFactory | None,
) -> str:
    """Select and run the best agent for a subtask.

    Routing: intent → agent_type (thinker/retriever/executor) → adapter.
    The adapter dynamically generates the system prompt using task + context + intent.
    """
    if registry is None or factory is None:
        return f"[mock] Completed: {task}"

    # Map fine-grained intent to agent type
    intent = IntentCategory(agent_type) if agent_type in IntentCategory.__members__.values() else IntentCategory.GENERAL
    target_type = INTENT_AGENT_MAP.get(intent, "thinker")

    # Select agent by type
    agent_meta = registry.select_by_type(target_type)

    # Fallback: pick highest-priority agent
    if agent_meta is None:
        all_agents = registry.list_all()
        if all_agents:
            agent_meta = min(all_agents, key=lambda a: a.priority)
        else:
            return f"[no agent available] Attempted: {task}"

    adapter = factory.get_adapter(agent_meta)
    result = await adapter.execute(task, context or None, intent=intent.value)

    # Update stats
    registry.update_stats(agent_meta.id, result.success)

    return result.output


def _finalize(subtasks: list[dict], results: dict[str, str]) -> dict[str, Any]:
    """Combine subtask results into final output."""
    if len(results) == 1:
        final = list(results.values())[0]
    else:
        parts = []
        for st in subtasks:
            sid = st["id"]
            if sid in results:
                parts.append(f"## {st.get('name', sid)}\n{results[sid]}")
        final = "\n\n".join(parts)

    return {
        "subtasks": subtasks,
        "subtask_results": results,
        "final_output": final,
        "status": "quality_check",
    }


def should_continue_executing(state: AgentState) -> str:
    """Conditional edge: continue executing or move to quality check."""
    status = state.get("status", "")
    if status == "executing":
        return "execute"
    return "quality_check"


def should_retry_or_finish(state: AgentState) -> str:
    """Conditional edge after quality check: finish or retry."""
    if state.get("quality_passed", False):
        return "end"
    # Phase 1: no retry, just finish
    return "end"


def build_graph(
    registry: AgentRegistry | None = None,
    adapter_factory: AdapterFactory | None = None,
    llm=None,
    llm_light=None,
    qdrant=None,
    collection_name: str = "case_library",
    on_subtask_progress=None,
) -> StateGraph:
    """Build the main orchestration graph.

    Args:
        registry: Agent registry for agent selection.
        adapter_factory: Factory for creating agent adapters.
        llm: LLM for agent execution (full max_tokens).
        llm_light: Lightweight LLM for control-plane calls (dispatch/plan/quality).
                   Falls back to llm if not provided.
        qdrant: Qdrant client for experience retrieval.
        collection_name: Qdrant collection for the case library.
        on_subtask_progress: Optional async callback for subtask-level progress.

    Returns:
        A compiled LangGraph StateGraph.
    """
    ctrl_llm = llm_light or llm  # control-plane uses light version

    # Create skill acquirer with available resources
    skill_acq = SkillAcquirer(qdrant=qdrant, collection_name=collection_name, llm=ctrl_llm)

    graph = StateGraph(AgentState)

    @timed_node("dispatch")
    async def _dispatch(state):
        return await dispatch_node(state, llm=ctrl_llm, qdrant=qdrant, collection_name=collection_name)

    @timed_node("plan")
    async def _plan(state):
        return await plan_node(state, llm=ctrl_llm)

    @timed_node("execute")
    async def _execute(state):
        return await execute_node(
            state, registry=registry, adapter_factory=adapter_factory,
            skill_acquirer=skill_acq, on_subtask_progress=on_subtask_progress,
        )

    @timed_node("quality_check")
    async def _quality(state):
        return await quality_check_node(state, llm=ctrl_llm)

    graph.add_node("dispatch", _dispatch)
    graph.add_node("plan", _plan)
    graph.add_node("execute", _execute)
    graph.add_node("quality_check", _quality)

    # Edges
    graph.set_entry_point("dispatch")
    graph.add_edge("dispatch", "plan")
    graph.add_edge("plan", "execute")
    graph.add_conditional_edges("execute", should_continue_executing, {
        "execute": "execute",
        "quality_check": "quality_check",
    })
    graph.add_conditional_edges("quality_check", should_retry_or_finish, {
        "end": END,
    })

    return graph
