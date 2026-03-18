"""Tests for OpenClaw adapter."""

import pytest
from agents.adapter import OpenClawAdapter, AdapterFactory, AgentResult
from agents.registry import AgentMeta


def _make_meta(id: str = "test_agent") -> AgentMeta:
    return AgentMeta(id=id, name="Test Agent", description="A test agent", skills=["test"])


@pytest.mark.asyncio
async def test_mock_execution():
    adapter = OpenClawAdapter(_make_meta())
    result = await adapter.execute("write hello world")
    assert result.success is True
    assert "Test Agent" in result.output
    assert "write hello world" in result.output


@pytest.mark.asyncio
async def test_mock_execution_with_context():
    adapter = OpenClawAdapter(_make_meta())
    result = await adapter.execute("analyze data", context={"data": "sample"})
    assert result.success is True
    assert "context" in result.output


@pytest.mark.asyncio
async def test_result_structure():
    adapter = OpenClawAdapter(_make_meta("my_agent"))
    result = await adapter.execute("do something")
    assert isinstance(result, AgentResult)
    assert result.agent_id == "my_agent"


def test_adapter_factory():
    factory = AdapterFactory()
    meta = _make_meta()
    a1 = factory.get_adapter(meta)
    a2 = factory.get_adapter(meta)
    assert a1 is a2  # cached


def test_adapter_factory_different_agents():
    factory = AdapterFactory()
    m1 = _make_meta("a1")
    m2 = _make_meta("a2")
    assert factory.get_adapter(m1) is not factory.get_adapter(m2)
