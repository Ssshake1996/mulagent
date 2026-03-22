"""Tool registry: manages all available tools and their schemas."""

from __future__ import annotations

import logging
from typing import Any

from tools.base import ToolDef
from tools.discovery import WEB_SEARCH, KNOWLEDGE_RECALL
from tools.injection import WEB_FETCH, READ_FILE, LIST_DIR
from tools.generation import EXECUTE_SHELL, CODE_RUN, WRITE_FILE, EDIT_FILE
from tools.isolation import DELEGATE
from tools.research import DEEP_RESEARCH
from tools.docs_lookup import DOCS_LOOKUP
from tools.codemap import CODEMAP
from tools.browser import BROWSER_FETCH
from tools.sql_query import SQL_QUERY
from tools.git_tools import GIT_OPS, GITHUB_OPS

logger = logging.getLogger(__name__)

# All built-in tools
ALL_TOOLS: list[ToolDef] = [
    WEB_SEARCH,
    KNOWLEDGE_RECALL,
    WEB_FETCH,
    READ_FILE,
    LIST_DIR,
    EXECUTE_SHELL,
    CODE_RUN,
    WRITE_FILE,
    EDIT_FILE,
    DELEGATE,
    DEEP_RESEARCH,
    DOCS_LOOKUP,
    CODEMAP,
    BROWSER_FETCH,
    SQL_QUERY,
    GIT_OPS,
    GITHUB_OPS,
]


class ToolRegistry:
    """Registry of tools available to the orchestrator."""

    def __init__(self, tools: list[ToolDef] | None = None):
        self._tools: dict[str, ToolDef] = {}
        for tool in (tools or ALL_TOOLS):
            self.register(tool)

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def as_dict(self) -> dict[str, ToolDef]:
        return dict(self._tools)

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Generate OpenAI-compatible tool schemas for LLM bind_tools."""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def tool_descriptions_text(self) -> str:
        """Generate a human-readable tool list for system prompts."""
        lines = []
        for tool in self._tools.values():
            params = ", ".join(
                f"{k}: {v.get('type', 'any')}"
                for k, v in tool.parameters.get("properties", {}).items()
            )
            lines.append(f"- **{tool.name}**({params}): {tool.description}")
        return "\n".join(lines)


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
