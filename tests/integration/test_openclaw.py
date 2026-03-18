"""Integration tests for OpenClaw agent execution.

These tests require:
  - `openclaw` CLI on PATH
  - OpenClaw agents registered: code_agent, research_agent, writing_agent, reasoning_agent, data_agent
  - A working LLM provider (bailian/qwen3.5-plus)

Run with: PYTHONPATH=src python -m pytest tests/integration/test_openclaw.py -v
"""

import json
import pytest
import shutil

from agents.adapter import (
    AdapterFactory,
    OpenClawAdapter,
    OPENCLAW_AGENT_MAP,
    _openclaw_available,
)
from agents.registry import AgentMeta


pytestmark = pytest.mark.skipif(
    not _openclaw_available(),
    reason="openclaw CLI not found on PATH",
)


def _make_meta(agent_id: str) -> AgentMeta:
    return AgentMeta(
        id=agent_id,
        name=agent_id.replace("_", " ").title(),
        description=f"Test {agent_id}",
        skills=["test"],
    )


@pytest.mark.asyncio
async def test_openclaw_code_agent():
    """Test code_agent via OpenClaw CLI produces valid code."""
    adapter = OpenClawAdapter(
        agent_meta=_make_meta("code_agent"),
        use_openclaw=True,
        openclaw_timeout=60,
    )
    result = await adapter.execute("Write a Python function that returns the factorial of n. Return only the function.")
    assert result.success, f"Agent failed: {result.output}"
    assert "def " in result.output
    assert "factorial" in result.output.lower() or "fact" in result.output.lower()


@pytest.mark.asyncio
async def test_openclaw_writing_agent():
    """Test writing_agent via OpenClaw CLI."""
    adapter = OpenClawAdapter(
        agent_meta=_make_meta("writing_agent"),
        use_openclaw=True,
        openclaw_timeout=60,
    )
    result = await adapter.execute("Translate the following to English: 今天天气很好")
    assert result.success, f"Agent failed: {result.output}"
    assert len(result.output) > 5  # should have some translation


@pytest.mark.asyncio
async def test_openclaw_reasoning_agent():
    """Test reasoning_agent via OpenClaw CLI."""
    adapter = OpenClawAdapter(
        agent_meta=_make_meta("reasoning_agent"),
        use_openclaw=True,
        openclaw_timeout=60,
    )
    result = await adapter.execute("What is 17 * 23? Show your work.")
    assert result.success, f"Agent failed: {result.output}"
    assert "391" in result.output


@pytest.mark.asyncio
async def test_adapter_factory_with_openclaw():
    """Test AdapterFactory creates OpenClaw-enabled adapters."""
    factory = AdapterFactory(use_openclaw=True, openclaw_timeout=60)
    meta = _make_meta("code_agent")
    adapter = factory.get_adapter(meta)
    assert adapter._use_openclaw is True
    assert adapter._openclaw_timeout == 60


@pytest.mark.asyncio
async def test_openclaw_with_context():
    """Test OpenClaw agent receives context from prior steps."""
    adapter = OpenClawAdapter(
        agent_meta=_make_meta("code_agent"),
        use_openclaw=True,
        openclaw_timeout=60,
    )
    result = await adapter.execute(
        "Add error handling to this function",
        context={"step1": "def divide(a, b): return a / b"},
    )
    assert result.success, f"Agent failed: {result.output}"
    # Should reference the divide function or error handling
    output_lower = result.output.lower()
    assert "def " in result.output or "error" in output_lower or "except" in output_lower


@pytest.mark.asyncio
async def test_openclaw_agent_map_completeness():
    """Verify all 5 expected agents are in the mapping."""
    expected = {"code_agent", "research_agent", "data_agent", "writing_agent", "reasoning_agent"}
    assert set(OPENCLAW_AGENT_MAP.keys()) == expected
