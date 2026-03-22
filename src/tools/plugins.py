"""Plugin system: load custom tools from YAML configuration.

Users can define custom API tools in config/tools.yaml:

```yaml
tools:
  - name: weather
    description: "查询城市天气"
    endpoint: "https://api.example.com/weather"
    method: GET
    params:
      city:
        type: string
        description: "城市名称"
        required: true
    headers:
      Authorization: "Bearer ${WEATHER_API_KEY}"
    response_path: "data.weather"  # jmespath-like extraction

  - name: create_ticket
    description: "创建工单"
    endpoint: "https://jira.internal/api/tickets"
    method: POST
    params:
      title:
        type: string
        description: "工单标题"
        required: true
      body:
        type: string
        description: "工单内容"
    headers:
      Authorization: "Bearer ${JIRA_TOKEN}"
```

Environment variables in headers are resolved at runtime via ${VAR_NAME}.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from tools.base import ToolDef

logger = logging.getLogger(__name__)

_PLUGIN_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "tools.yaml"

# Regex to match ${ENV_VAR} patterns
_ENV_VAR_RE = re.compile(r"\$\{(\w+)\}")


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} with environment variable values."""
    def _replace(match):
        var_name = match.group(1)
        return os.environ.get(var_name, f"${{{var_name}}}")
    return _ENV_VAR_RE.sub(_replace, value)


def _extract_value(data: Any, path: str) -> Any:
    """Extract a value from nested dict using dot-separated path.

    E.g., "data.weather.temp" → data["data"]["weather"]["temp"]
    """
    if not path:
        return data
    parts = path.split(".")
    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, current)
        elif isinstance(current, list) and part.isdigit():
            idx = int(part)
            current = current[idx] if idx < len(current) else current
        else:
            break
    return current


def _build_plugin_fn(config: dict[str, Any]):
    """Build an async tool function from plugin config."""
    endpoint = config["endpoint"]
    method = config.get("method", "GET").upper()
    headers_template = config.get("headers", {})
    response_path = config.get("response_path", "")
    timeout = config.get("timeout", 30)

    async def _plugin_fn(params: dict[str, Any], **deps: Any) -> str:
        import httpx

        # Resolve environment variables in headers
        headers = {k: _resolve_env_vars(v) for k, v in headers_template.items()}
        headers.setdefault("Content-Type", "application/json")

        # Build URL with query params for GET, body for POST
        url = endpoint
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                if method == "GET":
                    resp = await client.get(url, params=params, headers=headers)
                elif method == "POST":
                    resp = await client.post(url, json=params, headers=headers)
                elif method == "PUT":
                    resp = await client.put(url, json=params, headers=headers)
                elif method == "DELETE":
                    resp = await client.delete(url, params=params, headers=headers)
                else:
                    return f"Unsupported HTTP method: {method}"

                if resp.status_code >= 400:
                    return f"API error {resp.status_code}: {resp.text[:500]}"

                # Parse response
                try:
                    data = resp.json()
                    if response_path:
                        data = _extract_value(data, response_path)
                    # Format output
                    if isinstance(data, (dict, list)):
                        output = json.dumps(data, ensure_ascii=False, indent=2)
                    else:
                        output = str(data)
                    # Truncate very long responses
                    if len(output) > 5000:
                        output = output[:5000] + "\n... (truncated)"
                    return output
                except (json.JSONDecodeError, ValueError):
                    text = resp.text
                    if len(text) > 5000:
                        text = text[:5000] + "\n... (truncated)"
                    return text

        except httpx.TimeoutException:
            return f"API call timed out ({timeout}s): {url}"
        except Exception as e:
            return f"API call failed: {e}"

    return _plugin_fn


def _build_tool_schema(config: dict[str, Any]) -> dict[str, Any]:
    """Build JSON Schema parameters from plugin config."""
    params_config = config.get("params", {})
    properties = {}
    required = []

    for param_name, param_def in params_config.items():
        if isinstance(param_def, dict):
            properties[param_name] = {
                "type": param_def.get("type", "string"),
                "description": param_def.get("description", ""),
            }
            if param_def.get("required", False):
                required.append(param_name)
        else:
            # Simple string type
            properties[param_name] = {
                "type": "string",
                "description": str(param_def),
            }

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }


def load_plugins(config_path: Path | None = None) -> list[ToolDef]:
    """Load custom tools from config/tools.yaml.

    Returns:
        List of ToolDef objects ready to register.
    """
    path = config_path or _PLUGIN_CONFIG_PATH
    if not path.exists():
        return []

    try:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to load plugin config: %s", e)
        return []

    tools_config = data.get("tools", [])
    if not isinstance(tools_config, list):
        return []

    plugins = []
    for config in tools_config:
        if not isinstance(config, dict):
            continue
        name = config.get("name", "")
        if not name or not config.get("endpoint"):
            logger.warning("Skipping plugin with missing name/endpoint: %s", config)
            continue

        tool = ToolDef(
            name=name,
            description=config.get("description", f"Custom API tool: {name}"),
            parameters=_build_tool_schema(config),
            fn=_build_plugin_fn(config),
        )
        plugins.append(tool)
        logger.info("Loaded plugin tool: %s → %s %s",
                     name, config.get("method", "GET"), config["endpoint"])

    return plugins
