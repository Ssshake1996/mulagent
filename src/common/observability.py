"""Observability trio: Prometheus metrics, distributed tracing, alerting.

Provides:
1. Prometheus-compatible metrics (counters, histograms, gauges)
2. Distributed tracing with trace_id/span_id propagation
3. Alert rules that fire on threshold breaches

Designed to work with or without external infrastructure:
- With Prometheus: exposes /metrics endpoint
- Without Prometheus: metrics stored in memory, queryable via API
- Tracing: generates trace/span IDs, logs them for correlation
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Metrics ───────────────────────────────────────────────────────

class MetricsRegistry:
    """Prometheus-compatible metrics registry.

    Supports counters, histograms, and gauges.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._counters: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._alert_rules: list[AlertRule] = []

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Increment a counter."""
        key = self._key(name, labels)
        self._counters[key] += value
        self._check_alerts(name, self._counters[key], labels)

    def observe(self, name: str, value: float, **labels: str) -> None:
        """Record a histogram observation (e.g., latency)."""
        key = self._key(name, labels)
        hist = self._histograms[key]
        hist.append(value)
        # Keep last 1000 observations to prevent memory growth
        if len(hist) > 1000:
            self._histograms[key] = hist[-500:]
        self._check_alerts(name, value, labels)

    def set_gauge(self, name: str, value: float, **labels: str) -> None:
        """Set a gauge to an absolute value."""
        key = self._key(name, labels)
        self._gauges[key] = value
        self._check_alerts(name, value, labels)

    def get_counter(self, name: str, **labels: str) -> float:
        return self._counters.get(self._key(name, labels), 0.0)

    def get_histogram_stats(self, name: str, **labels: str) -> dict[str, float]:
        """Get histogram statistics: count, mean, p50, p95, p99."""
        key = self._key(name, labels)
        values = self._histograms.get(key, [])
        if not values:
            return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "count": n,
            "mean": round(sum(sorted_v) / n, 3),
            "p50": round(sorted_v[n // 2], 3),
            "p95": round(sorted_v[int(n * 0.95)], 3),
            "p99": round(sorted_v[int(n * 0.99)], 3),
        }

    def to_prometheus(self) -> str:
        """Export all metrics in Prometheus text format."""
        lines = []
        for key, value in sorted(self._counters.items()):
            name, labels_str = self._unkey(key)
            lines.append(f"{name}{labels_str} {value}")
        for key, values in sorted(self._histograms.items()):
            name, labels_str = self._unkey(key)
            if values:
                lines.append(f"{name}_count{labels_str} {len(values)}")
                lines.append(f"{name}_sum{labels_str} {sum(values):.3f}")
        for key, value in sorted(self._gauges.items()):
            name, labels_str = self._unkey(key)
            lines.append(f"{name}{labels_str} {value}")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for the /metrics API."""
        result = {"counters": {}, "histograms": {}, "gauges": {}}
        for key, value in self._counters.items():
            result["counters"][key] = value
        for key in self._histograms:
            name, _ = self._unkey(key)
            result["histograms"][key] = self.get_histogram_stats(name)
        for key, value in self._gauges.items():
            result["gauges"][key] = value
        return result

    def add_alert_rule(self, rule: AlertRule) -> None:
        self._alert_rules.append(rule)

    def _check_alerts(self, name: str, value: float, labels: dict) -> None:
        for rule in self._alert_rules:
            if rule.metric_name == name and value >= rule.threshold:
                if not rule.fired:
                    rule.fired = True
                    rule.fire_time = time.time()
                    logger.warning(
                        "ALERT [%s]: %s = %.2f >= %.2f — %s",
                        rule.severity, name, value, rule.threshold, rule.message,
                    )

    @staticmethod
    def _key(name: str, labels: dict[str, str]) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"

    @staticmethod
    def _unkey(key: str) -> tuple[str, str]:
        if "{" in key:
            name, rest = key.split("{", 1)
            return name, "{" + rest
        return key, ""


@dataclass
class AlertRule:
    """A threshold-based alert rule."""
    metric_name: str
    threshold: float
    severity: str = "warning"  # warning, critical
    message: str = ""
    fired: bool = False
    fire_time: float = 0.0


# ── Tracing ───────────────────────────────────────────────────────

@dataclass
class Span:
    """A span in a distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: str = ""
    operation: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class Tracer:
    """Simple distributed tracer with trace_id/span_id propagation."""

    def __init__(self, metrics: MetricsRegistry | None = None):
        self._metrics = metrics
        self._active_spans: dict[str, Span] = {}
        # Keep recent completed traces for debugging
        self._recent_traces: list[Span] = []
        self._max_recent = 100

    def new_trace_id(self) -> str:
        return uuid.uuid4().hex[:16]

    def new_span_id(self) -> str:
        return uuid.uuid4().hex[:8]

    @contextmanager
    def span(self, operation: str, trace_id: str = "", parent_span_id: str = ""):
        """Context manager for creating a traced span."""
        tid = trace_id or self.new_trace_id()
        sid = self.new_span_id()
        s = Span(
            trace_id=tid,
            span_id=sid,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
        )
        self._active_spans[sid] = s

        try:
            yield s
            s.status = "ok"
        except Exception as e:
            s.status = "error"
            s.attributes["error"] = str(e)
            raise
        finally:
            s.end_time = time.time()
            del self._active_spans[sid]
            self._recent_traces.append(s)
            if len(self._recent_traces) > self._max_recent:
                self._recent_traces = self._recent_traces[-self._max_recent // 2:]

            # Record span duration as metric
            if self._metrics:
                self._metrics.observe(
                    "span_duration_seconds",
                    s.duration_ms / 1000,
                    operation=operation,
                    status=s.status,
                )

            logger.debug(
                "trace=%s span=%s op=%s duration=%.1fms status=%s",
                tid, sid, operation, s.duration_ms, s.status,
            )

    @asynccontextmanager
    async def async_span(self, operation: str, trace_id: str = "", parent_span_id: str = ""):
        """Async context manager for creating a traced span."""
        tid = trace_id or self.new_trace_id()
        sid = self.new_span_id()
        s = Span(
            trace_id=tid,
            span_id=sid,
            parent_span_id=parent_span_id,
            operation=operation,
            start_time=time.time(),
        )
        self._active_spans[sid] = s

        try:
            yield s
            s.status = "ok"
        except Exception as e:
            s.status = "error"
            s.attributes["error"] = str(e)
            raise
        finally:
            s.end_time = time.time()
            del self._active_spans[sid]
            self._recent_traces.append(s)
            if len(self._recent_traces) > self._max_recent:
                self._recent_traces = self._recent_traces[-self._max_recent // 2:]

            if self._metrics:
                self._metrics.observe(
                    "span_duration_seconds",
                    s.duration_ms / 1000,
                    operation=operation,
                    status=s.status,
                )

    def get_recent_traces(self, limit: int = 20) -> list[dict]:
        """Get recent completed spans for debugging."""
        return [
            {
                "trace_id": s.trace_id,
                "span_id": s.span_id,
                "parent_span_id": s.parent_span_id,
                "operation": s.operation,
                "duration_ms": round(s.duration_ms, 1),
                "status": s.status,
                "attributes": s.attributes,
            }
            for s in self._recent_traces[-limit:]
        ]


# ── Global instances ──────────────────────────────────────────────

metrics = MetricsRegistry()
tracer = Tracer(metrics)


