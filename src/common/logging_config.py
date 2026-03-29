"""Structured logging configuration.

Provides JSON-formatted logging for production and human-readable format for dev.
Also includes a metrics collector for key performance indicators.

Usage:
    from common.logging_config import setup_logging, metrics

    setup_logging(json_format=True)  # production
    setup_logging(json_format=False)  # development

    metrics.record_task("react", 2.5, "completed", tools_used=["web_search"])
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from typing import Any


# ── JSON Log Formatter ────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    """Outputs log records as single-line JSON objects.

    Fields: timestamp, level, logger, message, plus any extras.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Inject trace_id from context (always, if active)
        try:
            from common.trace_context import get_trace_id
            tid = get_trace_id()
            if tid:
                log_entry["trace_id"] = tid
        except Exception:
            pass

        # Add extra fields (from logger.info("msg", extra={...}))
        for key in ("trace_id", "task_type", "duration_s", "status", "tools_used",
                     "session_id", "user_id", "model", "tokens",
                     "tool_name", "error"):
            val = getattr(record, key, None)
            if val is not None:
                log_entry[key] = val

        return json.dumps(log_entry, ensure_ascii=False, default=str)


# ── Metrics Collector ────────────────────────────────────────────

@dataclass
class TaskMetric:
    """A single task execution metric."""
    task_type: str
    duration_s: float
    status: str
    tools_used: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """In-process metrics collector for observability.

    Tracks:
    - Task count, latency, success rate
    - Tool usage frequency
    - Error counts by type
    """

    def __init__(self):
        self._tasks: list[TaskMetric] = []
        self._tool_counts: dict[str, int] = {}
        self._error_counts: dict[str, int] = {}
        self._max_history = 1000

    def record_task(
        self,
        task_type: str,
        duration_s: float,
        status: str,
        *,
        tools_used: list[str] | None = None,
    ) -> None:
        """Record a completed task."""
        metric = TaskMetric(
            task_type=task_type,
            duration_s=duration_s,
            status=status,
            tools_used=tools_used or [],
        )
        self._tasks.append(metric)
        if len(self._tasks) > self._max_history:
            self._tasks = self._tasks[-self._max_history:]

        for tool in metric.tools_used:
            self._tool_counts[tool] = self._tool_counts.get(tool, 0) + 1

        logger = logging.getLogger("metrics")
        logger.info(
            "task_completed",
            extra={
                "task_type": task_type,
                "duration_s": round(duration_s, 2),
                "status": status,
                "tools_used": tools_used or [],
            },
        )

    def get_summary(self, last_n_minutes: int = 60) -> dict[str, Any]:
        """Get a summary of metrics for the last N minutes."""
        cutoff = time.time() - (last_n_minutes * 60)
        recent = [t for t in self._tasks if t.timestamp > cutoff]

        if not recent:
            return {
                "period_minutes": last_n_minutes,
                "total_tasks": 0,
                "success_rate": 0.0,
                "avg_duration_s": 0.0,
                "tool_usage": {},
                "errors": dict(self._error_counts),
            }

        completed = [t for t in recent if t.status == "completed"]
        durations = [t.duration_s for t in recent]

        return {
            "period_minutes": last_n_minutes,
            "total_tasks": len(recent),
            "completed": len(completed),
            "failed": len(recent) - len(completed),
            "success_rate": round(len(completed) / len(recent), 3),
            "avg_duration_s": round(sum(durations) / len(durations), 2),
            "p95_duration_s": round(sorted(durations)[int(len(durations) * 0.95)], 2) if durations else 0,
            "tool_usage": dict(self._tool_counts),
            "errors": dict(self._error_counts),
        }


# Global metrics instance
metrics = MetricsCollector()


# ── Setup ────────────────────────────────────────────────────────

def setup_logging(
    *,
    json_format: bool = False,
    level: int = logging.INFO,
) -> None:
    """Configure logging for the application.

    Args:
        json_format: Use JSON formatter (production) vs human-readable (dev).
        level: Root logging level.
    """
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(level)

    if json_format:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
        ))

    root.addHandler(handler)

    # Suppress noisy libraries
    for lib in ("httpx", "httpcore", "urllib3", "websockets", "qdrant_client"):
        logging.getLogger(lib).setLevel(logging.WARNING)
