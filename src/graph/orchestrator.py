"""Main orchestrator: assembles the LangGraph that ties everything together.

Supports two modes:
  - ReAct mode (default): Single LLM with tool_use in a reasoning loop.
    No intent classification or DAG planning needed.
  - Legacy mode: dispatch → plan → execute (loop) → quality check → done.
    Kept for backward compatibility and fallback.
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


# ── ReAct mode entry point ────────────────────────────────────────

async def run_react(
    user_input: str,
    llm: Any = None,
    qdrant: Any = None,
    collection_name: str = "case_library",
    on_progress: Any = None,
    timeout: int = 0,  # 0 = use config default
    conversation_history: str = "",
    session_directives: list[str] | None = None,
    # Legacy fallback params
    registry: AgentRegistry | None = None,
    adapter_factory: AdapterFactory | None = None,
) -> dict[str, Any]:
    """Run a task using the ReAct orchestrator (new architecture).

    Falls back to legacy pipeline if no LLM is provided (mock/test mode).

    Returns a result dict:
        { "final_output": str, "status": str, "intent": str, ... }
    """
    # No LLM → fall back to legacy pipeline
    if llm is None:
        return await _run_legacy(
            user_input, registry=registry, adapter_factory=adapter_factory,
            llm=llm, qdrant=qdrant, collection_name=collection_name,
        )

    from common.config import get_settings
    from tools.registry import get_default_tools
    from graph.react_orchestrator import react_loop

    tool_registry = get_default_tools()
    tools = tool_registry.as_dict()
    react_cfg = get_settings().react

    deps = {
        "qdrant": qdrant,
        "collection_name": collection_name,
    }

    try:
        meta: dict[str, Any] = {}
        output = await react_loop(
            user_input=user_input,
            tools=tools,
            llm=llm,
            deps=deps,
            max_rounds=react_cfg.max_rounds,
            timeout=timeout or react_cfg.timeout,
            tool_timeout=react_cfg.tool_timeout,
            max_parallel_tools=react_cfg.max_parallel_tools,
            max_conversation_pairs=react_cfg.max_conversation_pairs,
            parent_directives=session_directives,
            conversation_history=conversation_history,
            on_progress=on_progress,
            result_meta=meta,
        )
        # ── Auto self-evaluation ──
        self_eval = None
        try:
            self_eval = await _self_evaluate(user_input, output, llm)
        except Exception as e:
            logger.debug("Self-evaluation skipped: %s", e)

        return {
            "final_output": output,
            "status": "completed",
            "intent": "react",
            "directives": meta.get("directives", []),
            "tools_used": meta.get("tools_used", []),
            "strategies_tried": meta.get("strategies_tried", []),
            "self_eval": self_eval,
        }
    except Exception as e:
        logger.exception("ReAct orchestrator failed")
        return {
            "final_output": f"Error: {e}",
            "status": "failed",
            "intent": "react",
            "error": str(e),
        }


async def _self_evaluate(
    user_input: str, output: str, llm: Any
) -> dict[str, Any] | None:
    """LLM self-evaluates the quality of its own answer.

    Returns a dict with scores and reasoning, or None on failure.
    Used to calibrate quality over time and improve without user feedback.
    """
    import asyncio
    import json
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "你是一个回答质量评估器。请评估以下任务回答的质量。\n"
            "返回 JSON 格式：\n"
            "{\n"
            '  "score": 1-5,           // 1=很差, 5=很好\n'
            '  "completeness": 1-5,    // 是否完整回答了问题\n'
            '  "accuracy_confidence": 1-5, // 你对答案准确性的信心\n'
            '  "has_sources": true/false,  // 是否包含来源引用\n'
            '  "improvement": "..."    // 一句话说明如何改进（如果需要）\n'
            "}\n"
            "只返回 JSON，不要解释。"
        )),
        HumanMessage(content=(
            f"用户任务: {user_input[:300]}\n\n"
            f"回答: {output[:800]}"
        )),
    ]

    try:
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=10)
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        result = json.loads(content)
        if isinstance(result, dict) and "score" in result:
            logger.info("Self-eval: score=%s, completeness=%s, confidence=%s",
                       result.get("score"), result.get("completeness"),
                       result.get("accuracy_confidence"))
            return result
    except Exception as e:
        logger.debug("Self-evaluation parse failed: %s", e)

    return None


async def _run_legacy(
    user_input: str,
    registry: AgentRegistry | None = None,
    adapter_factory: AdapterFactory | None = None,
    llm: Any = None,
    qdrant: Any = None,
    collection_name: str = "case_library",
) -> dict[str, Any]:
    """Legacy pipeline fallback for mock/test mode."""
    graph = build_graph(
        registry=registry,
        adapter_factory=adapter_factory,
        llm=llm,
        qdrant=qdrant,
        collection_name=collection_name,
    )
    compiled = graph.compile()
    result = await compiled.ainvoke({
        "user_input": user_input,
        "session_id": "legacy",
    })
    return result


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
