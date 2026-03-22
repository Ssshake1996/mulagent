"""Tool definitions for ReAct orchestrator.

Each tool is a callable with a standardized interface:
  - name: str
  - description: str (used in LLM tool_use schema)
  - parameters: dict (JSON Schema)
  - execute(params) -> str

Tools are grouped by context-management purpose:
  Discovery:  web_search, knowledge_recall
  Injection:  web_fetch, read_file
  Generation: execute_shell, code_run, write_file
  Isolation:  delegate
"""

from tools.registry import ToolRegistry, get_default_tools

__all__ = ["ToolRegistry", "get_default_tools"]
