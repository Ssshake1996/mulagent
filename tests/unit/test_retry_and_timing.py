"""Tests for retry utility and timing decorator."""

import pytest
from unittest.mock import AsyncMock

from common.retry import retry_async
from common.timing import timed_node


# --- Retry tests ---

@pytest.mark.asyncio
async def test_retry_succeeds_first_try():
    fn = AsyncMock(return_value="ok")
    result = await retry_async(fn, max_retries=2)
    assert result == "ok"
    assert fn.call_count == 1


@pytest.mark.asyncio
async def test_retry_succeeds_after_failure():
    fn = AsyncMock(side_effect=[ValueError("fail"), "ok"])
    result = await retry_async(fn, max_retries=1, base_delay=0.01)
    assert result == "ok"
    assert fn.call_count == 2


@pytest.mark.asyncio
async def test_retry_exhausted():
    fn = AsyncMock(side_effect=ValueError("always fail"))
    with pytest.raises(ValueError, match="always fail"):
        await retry_async(fn, max_retries=1, base_delay=0.01)
    assert fn.call_count == 2


@pytest.mark.asyncio
async def test_retry_no_retries():
    fn = AsyncMock(side_effect=ValueError("fail"))
    with pytest.raises(ValueError):
        await retry_async(fn, max_retries=0)
    assert fn.call_count == 1


@pytest.mark.asyncio
async def test_retry_selective_exceptions():
    fn = AsyncMock(side_effect=TypeError("wrong type"))
    with pytest.raises(TypeError):
        await retry_async(fn, max_retries=2, base_delay=0.01, retryable_exceptions=(ValueError,))
    assert fn.call_count == 1  # not retried because TypeError not in retryable


# --- Timing tests ---

@pytest.mark.asyncio
async def test_timed_node_returns_result():
    @timed_node("test_node")
    async def my_node(state):
        return {"status": "ok"}

    result = await my_node({})
    assert result["status"] == "ok"
    assert "_timing" in result
    assert "test_node" in result["_timing"]
    assert result["_timing"]["test_node"] >= 0


@pytest.mark.asyncio
async def test_timed_node_on_error():
    @timed_node("fail_node")
    async def my_node(state):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await my_node({})
