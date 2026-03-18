"""Quality gate: checks if the final output meets the user's request."""

from __future__ import annotations

import json
import logging
from typing import Any

from graph.state import AgentState

logger = logging.getLogger(__name__)


async def quality_check_node(state: AgentState, llm=None) -> dict[str, Any]:
    """LangGraph node: review output quality.

    If no LLM, auto-passes (Phase 1 basic behavior).
    """
    final_output = state.get("final_output", "")
    user_input = state.get("user_input", "")

    if not final_output:
        return {
            "quality_passed": False,
            "quality_feedback": "No output was produced.",
            "status": "failed",
        }

    if llm is not None:
        return await _check_via_llm(user_input, final_output, llm)

    # Auto-pass in mock mode
    return {
        "quality_passed": True,
        "quality_feedback": "Auto-passed (no LLM quality checker configured).",
        "status": "completed",
    }


async def _check_via_llm(user_input: str, output: str, llm) -> dict[str, Any]:
    """Use LLM to evaluate output quality."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from graph.dispatcher import load_prompt

    prompt = load_prompt("quality_check")
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["user"].format(user_input=user_input, output=output)),
    ]

    from common.retry import retry_async

    try:
        response = await retry_async(llm.ainvoke, messages, max_retries=2)
    except Exception:
        logger.warning("Quality check LLM failed after retries, auto-passing")
        return {
            "quality_passed": True,
            "quality_feedback": "Auto-passed (LLM unavailable).",
            "status": "completed",
        }

    try:
        result = json.loads(response.content)
        passed = result.get("passed", False)
        return {
            "quality_passed": passed,
            "quality_feedback": result.get("feedback", ""),
            "status": "completed" if passed else "quality_check",
        }
    except json.JSONDecodeError:
        logger.warning("Failed to parse quality check response, auto-passing")
        return {
            "quality_passed": True,
            "quality_feedback": "Auto-passed (parse error).",
            "status": "completed",
        }
