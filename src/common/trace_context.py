"""Request-scoped trace context using contextvars.

Provides a trace_id that propagates through the entire request lifecycle
without explicit parameter passing. Works with both sync and async code.

Usage:
    from common.trace_context import trace_ctx, get_trace_id

    # At request entry point (middleware or handler):
    trace_ctx.new_trace()

    # Anywhere in the call chain:
    tid = get_trace_id()  # returns current trace_id or ""
"""

from __future__ import annotations

import contextvars
import uuid

# Context variable holding the current trace_id
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


class TraceContext:
    """Manage request-scoped trace IDs."""

    def new_trace(self, trace_id: str = "") -> str:
        """Start a new trace. Returns the trace_id."""
        tid = trace_id or uuid.uuid4().hex[:16]
        _trace_id_var.set(tid)
        return tid

    def get(self) -> str:
        """Get the current trace_id (empty string if none active)."""
        return _trace_id_var.get()

    def set(self, trace_id: str) -> None:
        """Manually set trace_id (e.g., from incoming header)."""
        _trace_id_var.set(trace_id)

    def clear(self) -> None:
        """Clear the trace context."""
        _trace_id_var.set("")


# Global singleton
trace_ctx = TraceContext()


def get_trace_id() -> str:
    """Convenience shortcut for trace_ctx.get()."""
    return _trace_id_var.get()
