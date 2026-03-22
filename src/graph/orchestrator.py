"""Main orchestrator: ReAct-based task execution.

Single entry point for all task processing. The LLM decides what tools
to call in a reasoning loop — no intent classification or DAG planning needed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from common.llm import LLMManager

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
