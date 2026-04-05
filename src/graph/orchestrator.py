"""Main orchestrator: ReAct-based task execution + ProjectPilot routing.

Single entry point for all task processing. Small tasks go through the
ReAct loop directly; large multi-step projects are routed to ProjectPilot
(iterative DAG with feedback loop).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

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
) -> dict[str, Any]:
    """Run a task using the ReAct orchestrator.

    Returns a result dict:
        { "final_output": str, "status": str, "intent": str, ... }
    """
    if llm is None:
        return {
            "final_output": "Error: no LLM configured",
            "status": "failed",
            "intent": "react",
            "error": "no_llm",
        }

    from common.config import get_settings
    from tools.registry import get_default_tools
    from graph.react_orchestrator import react_loop, estimate_timeout

    tool_registry = get_default_tools()
    tools = tool_registry.as_dict()
    react_cfg = get_settings().react

    # Dynamic timeout: explicit > task-type estimate > config default
    effective_timeout = timeout or estimate_timeout(user_input, default=react_cfg.timeout)

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
            timeout=effective_timeout,
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

        # ── Plan mode: if LLM submitted a plan, return it for user approval ──
        if meta.get("plan_pending"):
            return {
                "final_output": output,
                "status": "plan_pending",
                "intent": "react",
                "directives": meta.get("directives", []),
                "tools_used": meta.get("tools_used", []),
                "strategies_tried": meta.get("strategies_tried", []),
                "self_eval": None,
            }

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
    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "你是一个回答质量评估器。请评估以下任务回答的质量。\n\n"
            "评分标准：\n"
            "- score: 1=完全没回答 2=尝试了但明显不完整 3=基本回答了核心问题 4=较完整，有细节 5=全面完整\n"
            "- completeness: 1=没开始 2=做了一部分 3=核心部分完成 4=大部分完成 5=全部完成\n"
            "- accuracy_confidence: 1=纯猜测 2=不太确定 3=基于部分事实 4=有依据 5=有工具验证\n\n"
            "重要：\n"
            "- 如果回答包含具体数据、代码、文件操作结果，completeness 应 ≥3\n"
            "- 如果回答以'请确认'/'是否继续'结尾但已完成主要工作，score 仍应 ≥3\n"
            "- 如果回答用了工具并展示了结果，accuracy_confidence 应 ≥4\n\n"
            "返回 JSON 格式：\n"
            "{\n"
            '  "score": 1-5,\n'
            '  "completeness": 1-5,\n'
            '  "accuracy_confidence": 1-5,\n'
            '  "has_sources": true/false,\n'
            '  "improvement": "..."\n'
            "}\n"
            "只返回 JSON，不要解释。"
        )),
        HumanMessage(content=(
            f"用户任务: {user_input[:300]}\n\n"
            f"回答: {output[:1200]}"
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


# ── Task Mode Classification ────────────────────────────────────────

_PROJECT_SIGNALS_ZH = [
    "项目", "方案", "工程", "系统设计", "架构设计", "整体规划",
    "分阶段", "多步骤", "分步", "第一步", "第二步",
    "全面", "完整方案", "端到端",
]
_PROJECT_SIGNALS_EN = [
    "project", "plan", "phase", "multi-step", "end-to-end",
    "step 1", "step 2", "comprehensive", "full plan",
    "architecture", "system design", "roadmap",
]


def _classify_task_mode(user_input: str) -> str:
    """Classify whether user_input is a single task or a multi-step project.

    Uses keyword heuristics — no LLM call needed.
    Returns 'project' or 'single'.
    """
    text = user_input.lower()
    signal_count = 0

    for kw in _PROJECT_SIGNALS_ZH + _PROJECT_SIGNALS_EN:
        if kw in text:
            signal_count += 1

    # Detect numbered lists (1. xxx  2. xxx) as project signals
    import re
    numbered_items = re.findall(r'(?:^|\n)\s*\d+[.、)\s]', user_input)
    if len(numbered_items) >= 3:
        signal_count += 2

    # Long inputs with multiple lines are more likely projects
    line_count = len(user_input.strip().split("\n"))
    if line_count >= 8:
        signal_count += 1

    if signal_count >= 2:
        logger.info("Task classified as 'project' (signal_count=%d)", signal_count)
        return "project"

    return "single"


# ── ProjectPilot Entry Point ────────────────────────────────────────

async def run_project_pilot(
    user_input: str,
    llm: Any = None,
    qdrant: Any = None,
    collection_name: str = "case_library",
    on_progress: Any = None,
    on_event: Any = None,
    timeout: int = 0,
    conversation_history: str = "",
    session_directives: list[str] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Run a task using ProjectPilot (iterative DAG orchestrator).

    Returns a result dict matching the run_react() format.
    """
    if llm is None:
        return {
            "final_output": "Error: no LLM configured",
            "status": "failed",
            "intent": "project",
            "error": "no_llm",
        }

    from graph.project_pilot import run_project, format_project_result

    react_params = {
        "llm": llm,
        "qdrant": qdrant,
        "collection_name": collection_name,
    }

    try:
        # No project-level timeout — subtasks inherit react.timeout individually.
        # A large project may run for hours; as long as subtasks keep progressing, let it run.
        state = await run_project(
            user_input=user_input,
            llm=llm,
            on_event=on_event,
            session_id=session_id,
            **react_params,
        )

        output = format_project_result(state)

        return {
            "final_output": output,
            "status": state.status,
            "intent": "project",
            "project_id": state.project_id,
            "iteration": state.iteration,
            "scores": state.scores,
            "subtask_count": state.total_count(),
            "completed_count": state.completed_count(),
        }

    except Exception as e:
        logger.exception("ProjectPilot failed")
        return {
            "final_output": f"项目执行失败: {e}",
            "status": "failed",
            "intent": "project",
            "error": str(e),
        }


# ── Unified Entry Point ─────────────────────────────────────────────

async def run_auto(
    user_input: str,
    llm: Any = None,
    qdrant: Any = None,
    collection_name: str = "case_library",
    on_progress: Any = None,
    on_event: Any = None,
    timeout: int = 0,
    conversation_history: str = "",
    session_directives: list[str] | None = None,
    session_id: str = "",
) -> dict[str, Any]:
    """Unified entry point: classify task mode and route accordingly.

    'single' tasks → run_react() (existing path)
    'project' tasks → run_project_pilot() (iterative DAG)
    """
    mode = _classify_task_mode(user_input)

    if mode == "project":
        return await run_project_pilot(
            user_input=user_input,
            llm=llm,
            qdrant=qdrant,
            collection_name=collection_name,
            on_progress=on_progress,
            on_event=on_event,
            timeout=timeout,
            conversation_history=conversation_history,
            session_directives=session_directives,
            session_id=session_id,
        )

    return await run_react(
        user_input=user_input,
        llm=llm,
        qdrant=qdrant,
        collection_name=collection_name,
        on_progress=on_progress,
        timeout=timeout,
        conversation_history=conversation_history,
        session_directives=session_directives,
    )
