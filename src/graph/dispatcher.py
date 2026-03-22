"""Dispatcher: intent recognition + classification + routing.

This is a LangGraph node that classifies the user's input and decides
which agent type should handle it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from graph.state import AgentState, IntentCategory

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "config" / "prompts"


def load_prompt(name: str) -> dict[str, str]:
    path = PROMPTS_DIR / f"{name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


async def dispatch_node(state: AgentState, llm=None, qdrant=None, collection_name: str = "case_library") -> dict[str, Any]:
    """LangGraph node: classify user input and determine intent.

    If Qdrant is available, retrieves similar past experiences to inform dispatch.
    If no LLM is provided, falls back to keyword-based classification.
    """
    user_input = state.get("user_input", "")

    # Retrieve similar experiences if available
    experiences = []
    if qdrant is not None:
        try:
            from common.vector import text_to_embedding_async
            from evolution.experience import search_similar_experiences

            query_vec = await text_to_embedding_async(user_input)
            experiences = await search_similar_experiences(qdrant, collection_name, query_vec, top_k=3)
            if experiences:
                logger.info("Found %d similar experiences for dispatch", len(experiences))
        except Exception as e:
            logger.debug("Experience retrieval skipped: %s", e)

    if llm is not None:
        return await _dispatch_via_llm(user_input, llm, experiences=experiences)

    return _dispatch_by_keywords(user_input)


def _dispatch_by_keywords(user_input: str) -> dict[str, Any]:
    """Keyword-based fallback dispatcher with scoring.

    Each intent gets a score based on how many keywords matched.
    The intent with the highest score wins. Ties are broken by priority
    (research > execute, since "search for X using command Y" is research).
    """
    text = user_input.lower()

    keyword_map = {
        IntentCategory.CODE: ["code", "function", "bug", "debug", "program", "script",
                              "class", "代码", "编程", "函数", "调试", "重构"],
        IntentCategory.RESEARCH: ["search", "find", "research", "what is", "who is",
                                  "搜索", "查找", "调研", "是什么", "获取", "列出",
                                  "热门", "排名", "排序", "top", "trending", "最新"],
        IntentCategory.DATA: ["data", "chart", "graph", "statistics", "analyze", "csv",
                              "数据", "图表", "统计", "分析"],
        IntentCategory.WRITING: ["write", "article", "translate", "blog", "email", "copy",
                                 "写", "文章", "翻译", "邮件", "文案"],
        IntentCategory.REASONING: ["calculate", "math", "solve", "logic", "prove",
                                   "数学", "解", "逻辑", "证明"],
        IntentCategory.EXECUTE: ["deploy", "restart", "install", "shell",
                                 "curl", "部署", "重启", "安装",
                                 "查看状态", "检查服务", "启动服务", "停止服务"],
    }

    # Score each intent by keyword match count
    scores: dict[IntentCategory, int] = {}
    for intent, keywords in keyword_map.items():
        count = sum(1 for kw in keywords if kw in text)
        if count > 0:
            scores[intent] = count

    if scores:
        best = max(scores, key=lambda k: scores[k])
        return {
            "intent": best.value,
            "complexity": "complex" if len(text) > 100 else "simple",
            "status": "dispatching",
        }

    return {
        "intent": IntentCategory.GENERAL.value,
        "complexity": "simple",
        "status": "dispatching",
    }


async def _dispatch_via_llm(user_input: str, llm, experiences: list[dict] | None = None) -> dict[str, Any]:
    """Classify using LLM, optionally enriched with similar past experiences."""
    from langchain_core.messages import HumanMessage, SystemMessage

    prompt = load_prompt("dispatcher")

    # Build experience context if available
    experience_hint = ""
    if experiences:
        positive_hints = []
        negative_hints = []
        for exp in experiences:
            if exp.get("score", 0) < 0.5:
                continue
            line = (
                f"- Pattern: {exp.get('problem_pattern', '')}, "
                f"Strategy: {exp.get('recommended_strategy', '')}, "
                f"Agents: {exp.get('recommended_agents', [])}"
            )
            if exp.get("is_negative"):
                negative_hints.append(f"- [WARNING] {exp.get('tips', '')}")
            else:
                positive_hints.append(line)

        parts = []
        if positive_hints:
            parts.append("Similar past tasks for reference:\n" + "\n".join(positive_hints))
        if negative_hints:
            parts.append("Past failures to avoid:\n" + "\n".join(negative_hints))
        if parts:
            experience_hint = "\n\n" + "\n\n".join(parts)

    user_content = prompt["user"].format(user_input=user_input) + experience_hint
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=user_content),
    ]

    from common.retry import retry_async

    try:
        response = await retry_async(llm.ainvoke, messages, max_retries=2)
    except Exception as e:
        logger.warning("Dispatcher LLM failed after retries: %s, falling back to keywords", e)
        return _dispatch_by_keywords(user_input)

    try:
        result = json.loads(response.content)
        return {
            "intent": result.get("intent", "general"),
            "complexity": result.get("complexity", "simple"),
            "status": "dispatching",
        }
    except json.JSONDecodeError:
        logger.warning("Failed to parse dispatcher LLM response, falling back to keywords")
        return _dispatch_by_keywords(user_input)
