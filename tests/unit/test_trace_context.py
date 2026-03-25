"""Tests for trace_id context propagation."""

import pytest
from common.trace_context import TraceContext, trace_ctx, get_trace_id


def test_new_trace_generates_id():
    ctx = TraceContext()
    tid = ctx.new_trace()
    assert len(tid) == 16
    assert tid.isalnum()


def test_new_trace_with_explicit_id():
    ctx = TraceContext()
    tid = ctx.new_trace("custom_trace_123")
    assert tid == "custom_trace_123"


def test_get_returns_current_trace():
    trace_ctx.new_trace("abc123def456gh")
    assert get_trace_id() == "abc123def456gh"


def test_clear_resets_trace():
    trace_ctx.new_trace("to_clear")
    assert get_trace_id() == "to_clear"
    trace_ctx.clear()
    assert get_trace_id() == ""


def test_set_overrides_trace():
    trace_ctx.new_trace("original")
    trace_ctx.set("overridden")
    assert get_trace_id() == "overridden"


def test_default_is_empty():
    trace_ctx.clear()
    assert get_trace_id() == ""


def test_json_formatter_includes_trace_id():
    """JSONFormatter should inject trace_id when active."""
    import logging
    from common.logging_config import JSONFormatter

    trace_ctx.new_trace("fmt_test_trace")

    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="test message", args=(), exc_info=None,
    )
    output = formatter.format(record)

    import json
    data = json.loads(output)
    assert data.get("trace_id") == "fmt_test_trace"

    trace_ctx.clear()


def test_json_formatter_no_trace_when_empty():
    """JSONFormatter should not include trace_id when no trace is active."""
    import logging
    import json
    from common.logging_config import JSONFormatter

    trace_ctx.clear()

    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname="", lineno=0,
        msg="no trace", args=(), exc_info=None,
    )
    output = formatter.format(record)
    data = json.loads(output)
    assert "trace_id" not in data
