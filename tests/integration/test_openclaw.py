"""Integration tests for OpenClaw agent execution.

These tests require:
  - `openclaw` CLI on PATH
  - OpenClaw agents registered: thinker, retriever, executor
  - A working LLM provider (qwen3.5-plus)

Run with: PYTHONPATH=src python -m pytest tests/integration/test_openclaw.py -v
"""

import pytest

from agents.adapter import (
    AdapterFactory,
    OpenClawAdapter,
    OPENCLAW_AGENT_MAP,
    _openclaw_available,
)
from agents.registry import AgentMeta


def _openclaw_agent_exists(agent_id: str) -> bool:
    """Check if a specific openclaw agent is registered."""
    import subprocess
    try:
        result = subprocess.run(
            ["openclaw", "agents", "list"],
            capture_output=True, text=True, timeout=5,
        )
        return agent_id in result.stdout
    except Exception:
        return False

pytestmark = pytest.mark.skipif(
    not _openclaw_available() or not _openclaw_agent_exists("thinker"),
    reason="openclaw CLI not found or 'thinker' agent not registered",
)


def _make_meta(agent_id: str, agent_type: str = "thinker") -> AgentMeta:
    return AgentMeta(
        id=agent_id,
        name=agent_id.replace("_", " ").title(),
        description=f"Test {agent_id}",
        agent_type=agent_type,
    )


@pytest.mark.asyncio
async def test_openclaw_thinker():
    """Test thinker via OpenClaw CLI produces valid code."""
    adapter = OpenClawAdapter(
        agent_meta=_make_meta("thinker", "thinker"),
        use_openclaw=True,
        openclaw_timeout=60,
    )
    result = await adapter.execute("Write a Python function that returns the factorial of n.")
    assert result.success, f"Agent failed: {result.output}"
    assert len(result.output) > 10


@pytest.mark.asyncio
async def test_adapter_factory_with_openclaw():
    """Test AdapterFactory creates OpenClaw-enabled adapters."""
    factory = AdapterFactory(use_openclaw=True, openclaw_timeout=60)
    meta = _make_meta("thinker")
    adapter = factory.get_adapter(meta)
    assert adapter._use_openclaw is True
    assert adapter._openclaw_timeout == 60


@pytest.mark.asyncio
async def test_openclaw_agent_map_completeness():
    """Verify all 3 agent types are in the mapping."""
    expected = {"thinker", "retriever", "executor"}
    assert set(OPENCLAW_AGENT_MAP.keys()) == expected
