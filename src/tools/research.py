"""Deep Research tool: multi-round iterative research with cross-referencing.

Unlike simple web_search (single query → results), deep_research performs:
1. Initial multi-angle search (2-3 queries)
2. Deep dive on best sources (web_fetch)
3. Cross-reference verification
4. Structured synthesis with citations
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)


async def _deep_research(params: dict[str, Any], **deps: Any) -> str:
    """Execute multi-round deep research on a topic."""
    topic = params.get("topic", "")
    if not topic:
        return "Error: topic is required"

    depth = params.get("depth", "standard")  # "quick", "standard", "thorough"
    max_searches = {"quick": 2, "standard": 3, "thorough": 5}.get(depth, 3)

    # Get tool functions from deps
    tools = deps.get("tools", {})
    web_search_tool = tools.get("web_search")
    web_fetch_tool = tools.get("web_fetch")
    llm = deps.get("llm")

    if web_search_tool is None:
        return "Error: web_search tool required for deep research"

    all_results: list[dict] = []
    sources: list[str] = []

    # ── Phase 1: Multi-angle initial search ──
    from datetime import date
    today = date.today().isoformat()

    # Generate search queries from different angles
    queries = [topic]
    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            response = await asyncio.wait_for(
                llm.ainvoke([
                    SystemMessage(content=(
                        "为以下研究主题生成 2-3 个不同角度的搜索查询。"
                        "每个查询应该覆盖主题的不同方面。"
                        "返回 JSON 数组。只返回数组，不要解释。"
                    )),
                    HumanMessage(content=f"主题: {topic}\n当前日期: {today}"),
                ]),
                timeout=10,
            )
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            parsed = json.loads(content)
            if isinstance(parsed, list):
                queries = [str(q) for q in parsed[:max_searches]]
        except Exception as e:
            logger.debug("Query expansion failed, using original: %s", e)

    # Execute searches
    search_tasks = []
    for query in queries[:max_searches]:
        search_tasks.append(web_search_tool.fn({"query": query}, **deps))

    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for i, result in enumerate(search_results):
        if isinstance(result, Exception):
            continue
        if isinstance(result, str) and result and "Error" not in result[:20]:
            all_results.append({
                "query": queries[i] if i < len(queries) else "unknown",
                "content": result,
            })

    if not all_results:
        return f"Research on '{topic}' returned no results from {len(queries)} searches."

    # ── Phase 2: Deep dive on best URLs (if web_fetch available) ──
    deep_content = []
    if web_fetch_tool and depth in ("standard", "thorough"):
        # Extract URLs from search results
        urls = []
        for r in all_results:
            for line in r["content"].split("\n"):
                if line.startswith("URL: "):
                    urls.append(line[5:].strip())

        # Fetch top 2-3 URLs for detail
        fetch_tasks = []
        for url in urls[:min(3, len(urls))]:
            fetch_tasks.append(web_fetch_tool.fn({"url": url}, **deps))

        if fetch_tasks:
            fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
            for i, result in enumerate(fetch_results):
                if isinstance(result, str) and result and "Error" not in result[:20]:
                    deep_content.append({
                        "url": urls[i] if i < len(urls) else "",
                        "content": result[:2000],
                    })
                    sources.append(urls[i] if i < len(urls) else "")

    # ── Phase 3: Synthesize with LLM ──
    if llm:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            # Build context from all gathered info
            context_parts = []
            for i, r in enumerate(all_results):
                context_parts.append(f"[Search {i+1}: {r['query']}]\n{r['content'][:1500]}")
            for i, d in enumerate(deep_content):
                context_parts.append(f"[Deep dive: {d['url']}]\n{d['content'][:1500]}")

            context = "\n\n---\n\n".join(context_parts)

            response = await asyncio.wait_for(
                llm.ainvoke([
                    SystemMessage(content=(
                        "你是一个研究分析师。根据以下搜索结果，生成一份结构化的研究报告。\n"
                        "要求：\n"
                        "1. 提取关键发现，按主题分组\n"
                        "2. 每个关键声明标注来源编号 [N]\n"
                        "3. 标注互相矛盾的信息\n"
                        "4. 最后附上来源列表\n"
                        "5. 用和用户相同的语言写\n"
                        "6. 不要编造任何信息"
                    )),
                    HumanMessage(content=(
                        f"研究主题: {topic}\n\n"
                        f"收集到的资料:\n{context}"
                    )),
                ]),
                timeout=30,
            )
            return response.content.strip()
        except Exception as e:
            logger.warning("Research synthesis failed: %s", e)

    # Fallback: return raw results
    parts = []
    for i, r in enumerate(all_results, 1):
        parts.append(f"**Search {i}: {r['query']}**\n{r['content'][:800]}")
    return "\n\n---\n\n".join(parts)


DEEP_RESEARCH = ToolDef(
    name="deep_research",
    description=(
        "Conduct deep multi-round research on a topic. Use for complex research tasks "
        "that need multiple search angles, source verification, and synthesis. "
        "Much more thorough than a single web_search call."
    ),
    parameters={
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "The research topic or question",
            },
            "depth": {
                "type": "string",
                "description": "Research depth: 'quick' (2 searches), 'standard' (3), 'thorough' (5)",
                "enum": ["quick", "standard", "thorough"],
            },
        },
        "required": ["topic"],
    },
    fn=_deep_research,
    category="external",
)
