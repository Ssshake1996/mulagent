"""Tool registry: manages all available tools and their schemas."""

from __future__ import annotations

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

# All built-in tools
ALL_TOOLS: list[ToolDef] = [
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
    CHECK_BACKGROUND,
    DEEP_RESEARCH,
    DOCS_LOOKUP,
    CODEMAP,
    BROWSER_FETCH,
    SQL_QUERY,
    GIT_OPS,
    GITHUB_OPS,
    TODO_MANAGE,
    PLAN_SUBMIT,
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
