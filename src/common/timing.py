"""Timing utilities for performance monitoring."""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


def timed_node(node_name: str):
    """Decorator that logs execution time of a LangGraph node function."""
    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(state: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any]:
            start = time.perf_counter()
            try:
                result = await fn(state, *args, **kwargs)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info("[%s] completed in %.0fms", node_name, elapsed_ms)
                # Inject timing into result for trace recording
                if isinstance(result, dict):
                    result.setdefault("_timing", {})[node_name] = round(elapsed_ms)
                return result
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.error("[%s] failed after %.0fms: %s", node_name, elapsed_ms, e)
                raise
        return wrapper
    return decorator
