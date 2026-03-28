"""Discovery tools: find information from external sources.

- web_search: Internet search via mcporter
- knowledge_recall: Qdrant case library retrieval
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)


async def _web_search(params: dict[str, Any], **deps: Any) -> str:
    """Execute web search via mcporter bailian_web_search."""
    query = params.get("query", "")
    if not query:
        return "Error: query is required"

    # Auto-inject current date for time-sensitive queries
    from datetime import date
    today = date.today().isoformat()
    if not any(kw in query for kw in [today, "2025", "2026", "今天", "today"]):
        query = f"{query} {today}"

    # Try mcporter first, then DuckDuckGo HTML fallback
    result = None

    if shutil.which("mcporter"):
        try:
            proc = await asyncio.create_subprocess_exec(
                "mcporter", "call", "WebSearch.bailian_web_search",
                f"query={query}", "--output", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode == 0:
                data = json.loads(stdout.decode())
                pages = data.get("pages", [])
                if pages:
                    result = _format_search_results(pages)
        except Exception as e:
            logger.info("mcporter search failed, trying fallback: %s", e)

    # Fallback: DuckDuckGo HTML search
    if result is None:
        result = await _ddg_search_fallback(query)

    return result if result else "No results found."


def _format_search_results(pages: list[dict]) -> str:
    """Format search result pages into a structured string."""
    parts = []
    sources = []
    for i, p in enumerate(pages[:5], 1):
        title = p.get("title", "")
        snippet = p.get("snippet", "")[:400]
        url = p.get("url", "")
        parts.append(f"**[{i}] {title}**\n{snippet}\nURL: {url}")
        sources.append(f"[{i}] {title}: {url}")

    result = "\n\n---\n\n".join(parts)
    result += "\n\n**Sources:**\n" + "\n".join(sources)
    return result


async def _ddg_search_fallback(query: str) -> str | None:
    """Fallback search using DuckDuckGo HTML (no API key needed)."""
    import re
    import urllib.parse

    encoded = urllib.parse.quote_plus(query)
    url = f"https://html.duckduckgo.com/html/?q={encoded}"

    try:
        proc = await asyncio.create_subprocess_exec(
            "curl", "-sL", "--max-time", "15",
            "-H", "User-Agent: Mozilla/5.0 (compatible; MulAgent/1.0)",
            url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=20)

        if proc.returncode != 0:
            return None

        html = stdout.decode(errors="replace")

        # Parse DuckDuckGo HTML results
        results = []
        # Extract result blocks
        for m in re.finditer(
            r'class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>.*?'
            r'class="result__snippet"[^>]*>(.*?)</(?:td|div)',
            html, re.DOTALL,
        ):
            link = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()

            # DuckDuckGo wraps URLs in a redirect — extract actual URL
            if "uddg=" in link:
                actual = re.search(r"uddg=([^&]+)", link)
                if actual:
                    link = urllib.parse.unquote(actual.group(1))

            if title and link:
                results.append({"title": title, "snippet": snippet, "url": link})

        if not results:
            return None

        return _format_search_results(results[:5])

    except Exception as e:
        logger.debug("DuckDuckGo fallback failed: %s", e)
        return None


async def _knowledge_recall(params: dict[str, Any], **deps: Any) -> str:
    """Search Qdrant case library for similar past experiences."""
    query = params.get("query", "")
    if not query:
        return "Error: query is required"

    qdrant = deps.get("qdrant")
    collection_name = deps.get("collection_name", "case_library")

    if qdrant is None:
        return "Knowledge base not available."

    try:
        from common.vector import text_to_embedding_async
        from evolution.experience import search_similar_experiences

        query_vec = await text_to_embedding_async(query)
        experiences = await search_similar_experiences(
            qdrant, collection_name, query_vec, top_k=5,
        )

        relevant = [e for e in experiences if e.get("score", 0) > 0.5]
        if not relevant:
            return "No relevant experiences found."

        parts = []
        for i, exp in enumerate(relevant, 1):
            tier_name = exp.get("tier_name", "strategy")
            tier_label = {"atomic": "L1", "strategy": "L2", "domain": "L3"}.get(tier_name, "L2")
            use_info = ""
            use_count = exp.get("use_count", 0)
            success_rate = exp.get("success_rate")
            if use_count > 0:
                rate_str = f", success={success_rate:.0%}" if success_rate is not None else ""
                use_info = f", used={use_count}{rate_str}"

            lines = [
                f"**[{tier_label} Experience {i}]** (Score: {exp.get('effective_score', exp.get('score', 0)):.2f}, "
                f"Complexity: {exp.get('complexity', '?')}/5{use_info})",
                f"Pattern: {exp.get('problem_pattern', '')}",
                f"Strategy: {exp.get('recommended_strategy', '')}",
                f"Tools: {', '.join(exp.get('recommended_agents', []))}",
                f"Tips: {exp.get('tips', '')}",
            ]
            tags = exp.get("domain_tags", [])
            if tags:
                lines.append(f"Domain: {', '.join(tags)}")
            failures = exp.get("failure_patterns", [])
            if failures:
                lines.append(f"⚠️ Avoid: {'; '.join(failures[:3])}")
            est_rounds = exp.get("estimated_rounds")
            if est_rounds:
                lines.append(f"Estimated rounds: {est_rounds}")
            parts.append("\n".join(lines))
        return "\n\n---\n\n".join(parts)
    except Exception as e:
        logger.warning("knowledge_recall error: %s", e)
        return f"Knowledge recall error: {e}"


WEB_SEARCH = ToolDef(
    name="web_search",
    description="Search the internet for information. Use when you need current/external knowledge.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query",
            },
        },
        "required": ["query"],
    },
    fn=_web_search,
)

KNOWLEDGE_RECALL = ToolDef(
    name="knowledge_recall",
    description="Search the internal knowledge base for similar past tasks and solutions.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Description of what you're looking for",
            },
        },
        "required": ["query"],
    },
    fn=_knowledge_recall,
)
