"""Base class and types for tools."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


@dataclass
class ToolDef:
    """Definition of a tool available to the ReAct orchestrator."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for tool parameters
    fn: Callable[..., Awaitable[str]]  # async (params) -> str
    category: str = "general"  # search, file, execution, vcs, task, delegation
    deferred: bool = False  # True = schema not sent until load_tool is called

    def to_openai_schema(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible function schema for tool_use."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
