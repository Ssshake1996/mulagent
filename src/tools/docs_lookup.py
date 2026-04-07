"""Documentation lookup tool via Context7 REST API.

Fetches version-specific official documentation for libraries/frameworks
using the Context7 REST API (v2).

Two-step process:
1. resolve-library-id: library name → Context7 library ID
2. get-library-docs: library ID + topic → relevant documentation

REST API: https://context7.com/api/v2/
Auth: Optional API key via CONTEXT7_API_KEY env var (higher rate limits)
Free tier: 1,000 calls/month with key, shared pool without key.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

_API_BASE = "https://context7.com/api/v2"
_REQUEST_TIMEOUT = 15

# Cache: library_name → (library_id, timestamp)
_lib_id_cache: dict[str, tuple[str, float]] = {}
_LIB_CACHE_TTL = 3600  # 1 hour

# Cache: (library_id, topic) → (docs, timestamp)
_docs_cache: dict[str, tuple[str, float]] = {}
_DOCS_CACHE_TTL = 600  # 10 minutes


def _get_auth_headers() -> list[str]:
    """Build curl auth headers if API key is available."""
    api_key = os.environ.get("CONTEXT7_API_KEY", "")
    if api_key:
        return ["-H", f"Authorization: Bearer {api_key}"]
    return []


async def _api_get(path: str, params: dict) -> dict | list | str:
    """Make a GET request to Context7 REST API."""
    import urllib.parse

    query = urllib.parse.urlencode(params)
    url = f"{_API_BASE}{path}?{query}"

    cmd = [
        "curl", "-sL", "--max-time", str(_REQUEST_TIMEOUT),
        "-H", "Accept: application/json",
    ]
    cmd.extend(_get_auth_headers())
    cmd.append(url)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_REQUEST_TIMEOUT + 5,
        )

        if proc.returncode != 0:
            return f"Error: Context7 request failed: {stderr.decode(errors='replace')[:200]}"

        raw = stdout.decode(errors="replace").strip()
        if not raw:
            return "Error: empty response from Context7"

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # May be HTML error page
            if "<html" in raw[:200].lower():
                return "Error: Context7 returned HTML (possible rate limit or maintenance)"
            return raw[:2000]

    except asyncio.TimeoutError:
        return "Error: Context7 request timed out"
    except Exception as e:
        return f"Error: Context7 request failed: {e}"


async def _resolve_library_id(library_name: str) -> str | None:
    """Resolve a library name to a Context7 library ID."""
    # Check cache
    cached = _lib_id_cache.get(library_name.lower())
    if cached and (time.monotonic() - cached[1]) < _LIB_CACHE_TTL:
        return cached[0]

    result = await _api_get("/libs/search", {"libraryName": library_name})

    if isinstance(result, str) and result.startswith("Error"):
        logger.warning("Library resolution failed for '%s': %s", library_name, result[:100])
        return None

    # REST API returns a list of library matches
    if isinstance(result, list) and result:
        # Pick the best match (first result, highest trust score)
        best = result[0]
        lib_id = best.get("id", "")
        if lib_id:
            _lib_id_cache[library_name.lower()] = (lib_id, time.monotonic())
            logger.info("Resolved '%s' → '%s' (%d snippets, trust: %s)",
                       library_name, lib_id,
                       best.get("totalSnippets", 0),
                       best.get("trustScore", "?"))
            return lib_id

    # Fallback: if result is a dict with id field
    if isinstance(result, dict) and "id" in result:
        lib_id = result["id"]
        _lib_id_cache[library_name.lower()] = (lib_id, time.monotonic())
        return lib_id

    return None


async def _docs_lookup(params: dict[str, Any], **deps: Any) -> str:
    """Look up official documentation for a library."""
    library = params.get("library", "")
    topic = params.get("topic", "")

    if not library:
        return "Error: library name is required"

    # Check docs cache
    cache_key = f"{library.lower()}:{topic.lower()}"
    cached = _docs_cache.get(cache_key)
    if cached and (time.monotonic() - cached[1]) < _DOCS_CACHE_TTL:
        return cached[0]

    # Step 1: Resolve library ID
    lib_id = await _resolve_library_id(library)
    if not lib_id:
        return await _fallback_docs_fetch(library, topic, deps)

    # Step 2: Get documentation
    query_params = {"libraryId": lib_id}
    if topic:
        query_params["query"] = topic
    else:
        query_params["query"] = library  # Use library name as default query

    result = await _api_get("/context", query_params)

    if isinstance(result, str) and result.startswith("Error"):
        return await _fallback_docs_fetch(library, topic, deps)

    # Format documentation results
    if isinstance(result, list) and result:
        parts = [f"# {library} Documentation\n"]
        for i, doc in enumerate(result[:8], 1):
            title = doc.get("title", f"Section {i}")
            content = doc.get("content", "")
            source = doc.get("source", "")
            parts.append(f"## {title}")
            if content:
                # Limit each section
                if len(content) > 1500:
                    content = content[:1500] + "\n..."
                parts.append(content)
            if source:
                parts.append(f"*Source: {source}*")
            parts.append("")

        text = "\n".join(parts)
        if len(text) > 6000:
            text = text[:6000] + "\n... (documentation truncated)"
        _docs_cache[cache_key] = (text, time.monotonic())
        return text

    # Single result or dict format
    if isinstance(result, dict):
        text = json.dumps(result, ensure_ascii=False, indent=1)
        if len(text) > 6000:
            text = text[:6000] + "\n... (truncated)"
        _docs_cache[cache_key] = (text, time.monotonic())
        return text

    return await _fallback_docs_fetch(library, topic, deps)


async def _fallback_docs_fetch(library: str, topic: str, deps: dict) -> str:
    """Fallback: fetch docs via web_search + web_fetch."""
    tools = deps.get("tools", {})
    web_search = tools.get("web_search")

    if web_search is None:
        return f"Could not find documentation for '{library}'. Context7 unavailable and no search fallback."

    query = f"{library} official documentation"
    if topic:
        query += f" {topic}"

    try:
        result = await web_search.fn({"query": query}, **deps)
        return f"[Fallback: web search results for {library} docs]\n\n{result}"
    except Exception as e:
        return f"Documentation lookup failed for '{library}': {e}"


DOCS_LOOKUP = ToolDef(
    name="docs_lookup",
    description=(
        "Look up official documentation for a library or framework. "
        "Uses Context7 to fetch version-specific docs. "
        "Example: docs_lookup(library='react', topic='useEffect cleanup')"
    ),
    parameters={
        "type": "object",
        "properties": {
            "library": {
                "type": "string",
                "description": "Library or framework name (e.g., 'react', 'fastapi', 'langchain')",
            },
            "topic": {
                "type": "string",
                "description": "Specific topic or API to look up (optional)",
            },
        },
        "required": ["library"],
    },
    fn=_docs_lookup,
    category="external",
)
