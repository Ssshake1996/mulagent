"""Tool registry: manages all available tools and their schemas."""

from __future__ import annotations

import json
import logging
from typing import Any

from tools.base import ToolDef
from tools.discovery import WEB_SEARCH, KNOWLEDGE_RECALL
from tools.injection import WEB_FETCH, READ_FILE, LIST_DIR, GLOB_SEARCH, GREP_SEARCH
from tools.generation import EXECUTE_SHELL, CODE_RUN, WRITE_FILE, EDIT_FILE
from tools.isolation import DELEGATE, CHECK_BACKGROUND
from tools.research import DEEP_RESEARCH
from tools.docs_lookup import DOCS_LOOKUP
from tools.codemap import CODEMAP
from tools.browser import BROWSER_FETCH
from tools.sql_query import SQL_QUERY
from tools.git_tools import GIT_OPS, GITHUB_OPS
from tools.task_manager import TODO_MANAGE, PLAN_SUBMIT

logger = logging.getLogger(__name__)

# ── Core tools: always sent with full schema ──
CORE_TOOLS: list[ToolDef] = [
    WEB_SEARCH,
    KNOWLEDGE_RECALL,
    WEB_FETCH,
    READ_FILE,
    LIST_DIR,
    GLOB_SEARCH,
    GREP_SEARCH,
    EXECUTE_SHELL,
    CODE_RUN,
    WRITE_FILE,
    EDIT_FILE,
    DELEGATE,
    GIT_OPS,
    GITHUB_OPS,
    TODO_MANAGE,
    PLAN_SUBMIT,
]

# ── Deferred tools: name + description only, schema loaded on demand ──
_DEFERRED_TOOL_DEFS: list[ToolDef] = [
    DEEP_RESEARCH,
    DOCS_LOOKUP,
    CODEMAP,
    BROWSER_FETCH,
    SQL_QUERY,
    CHECK_BACKGROUND,
]

# Mark deferred tools
for _t in _DEFERRED_TOOL_DEFS:
    _t.deferred = True

# All built-in tools
ALL_TOOLS: list[ToolDef] = CORE_TOOLS + _DEFERRED_TOOL_DEFS


# ── load_tool meta-tool ──

async def _load_tool_fn(params: dict[str, Any], **deps: Any) -> str:
    """Load the full schema of a deferred tool so it becomes available."""
    tool_name = params.get("name", "")
    if not tool_name:
        return "Error: 'name' parameter is required"

    # Find the tool in the registry passed via deps
    registry: dict[str, ToolDef] = deps.get("tools", {})
    tool = registry.get(tool_name)
    if not tool:
        available = [t.name for t in _DEFERRED_TOOL_DEFS]
        return f"Error: tool '{tool_name}' not found. Available deferred tools: {available}"

    if not tool.deferred:
        return f"Tool '{tool_name}' is already loaded (core tool)."

    # Return full schema so LLM knows the parameters
    schema = tool.to_openai_schema()
    return (
        f"Tool '{tool_name}' loaded successfully. Full schema:\n"
        f"```json\n{json.dumps(schema, indent=2, ensure_ascii=False)}\n```\n"
        f"You can now call {tool_name}() in the next round."
    )


LOAD_TOOL = ToolDef(
    name="load_tool",
    description="Load the full schema of a deferred tool to make it callable. Use this when you need a tool listed as [deferred].",
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Name of the deferred tool to load (e.g., 'deep_research', 'codemap')",
            },
        },
        "required": ["name"],
    },
    fn=_load_tool_fn,
)


class ToolRegistry:
    """Registry of tools available to the orchestrator."""

    def __init__(self, tools: list[ToolDef] | None = None):
        self._tools: dict[str, ToolDef] = {}
        for tool in (tools or ALL_TOOLS):
            self.register(tool)
        # Always include load_tool meta-tool
        self.register(LOAD_TOOL)

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def as_dict(self) -> dict[str, ToolDef]:
        return dict(self._tools)

def get_default_tools() -> ToolRegistry:
    """Create a registry with all default built-in tools + plugins."""
    registry = ToolRegistry(ALL_TOOLS)

    # Load custom plugin tools from config/tools.yaml
    try:
        from tools.plugins import load_plugins
        plugins = load_plugins()
        for plugin in plugins:
            registry.register(plugin)
        if plugins:
            logger.info("Loaded %d plugin tools", len(plugins))
    except Exception as e:
        logger.warning("Plugin loading failed: %s", e)

    return registry
