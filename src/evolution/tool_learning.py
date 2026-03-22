"""Tool Learning: parameter optimization and recommendation with anti-Matthew effect.

Uses UCB1 (Upper Confidence Bound) algorithm to balance exploitation (use tools
that worked well) with exploration (try underused tools to discover better strategies).

This prevents the Matthew Effect where popular tools get all the traffic while
potentially better tools are never tried.

Key algorithm: UCB1 score = success_rate + C * sqrt(ln(total_trials) / tool_trials)
- High success_rate → exploitation (use what works)
- Low tool_trials → high exploration bonus (try underused tools)
- C parameter controls exploration vs exploitation tradeoff
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolStats:
    """Statistics for a single tool."""
    name: str
    total_calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_s: float = 0.0
    # Parameter patterns: {param_hash: {args, successes, failures}}
    param_patterns: dict[str, dict] = field(default_factory=dict)
    # Task type affinity: {task_pattern: success_count}
    task_affinity: dict[str, int] = field(default_factory=dict)
    last_used: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_calls == 0:
            return 0.5  # Optimistic prior for untried tools
        return self.successes / self.total_calls

    @property
    def avg_latency_s(self) -> float:
        if self.total_calls == 0:
            return 0.0
        return self.total_latency_s / self.total_calls


class ToolLearner:
    """Learns optimal tool usage patterns with UCB1-based exploration.

    Anti-Matthew Effect: UCB1 gives exploration bonus to underused tools,
    ensuring the system continues to learn rather than converging on a
    local optimum.
    """

    def __init__(self, exploration_c: float = 1.4, decay_factor: float = 0.995):
        """
        Args:
            exploration_c: UCB1 exploration parameter. Higher = more exploration.
                          1.4 (sqrt(2)) is the theoretical optimum for UCB1.
            decay_factor: Applied to old stats each day to weight recent data more.
        """
        self._stats: dict[str, ToolStats] = {}
        self._exploration_c = exploration_c
        self._decay_factor = decay_factor
        self._total_trials = 0
        self._last_decay_time = time.time()

    def record_outcome(
        self,
        tool_name: str,
        args: dict[str, Any],
        success: bool,
        latency_s: float = 0.0,
        task_pattern: str = "",
    ) -> None:
        """Record the outcome of a tool call."""
        stats = self._stats.setdefault(tool_name, ToolStats(name=tool_name))
        stats.total_calls += 1
        stats.last_used = time.time()
        stats.total_latency_s += latency_s
        self._total_trials += 1

        if success:
            stats.successes += 1
        else:
            stats.failures += 1

        # Record parameter pattern
        param_key = self._param_hash(args)
        if param_key:
            pattern = stats.param_patterns.setdefault(param_key, {
                "args_sample": {k: str(v)[:50] for k, v in list(args.items())[:3]},
                "successes": 0, "failures": 0,
            })
            if success:
                pattern["successes"] += 1
            else:
                pattern["failures"] += 1

        # Record task affinity
        if task_pattern and success:
            stats.task_affinity[task_pattern] = stats.task_affinity.get(task_pattern, 0) + 1

        # Periodic decay (every ~24h)
        if time.time() - self._last_decay_time > 86400:
            self._apply_decay()

    def recommend_tools(
        self,
        task_description: str,
        available_tools: list[str],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Recommend tools for a task using UCB1 algorithm.

        Returns ranked list of tools with UCB1 scores and reasoning.
        """
        if not available_tools:
            return []

        task_pattern = self._extract_task_pattern(task_description)
        recommendations = []

        for tool_name in available_tools:
            stats = self._stats.get(tool_name)

            if stats is None or stats.total_calls == 0:
                # Untried tool gets maximum exploration bonus
                ucb_score = 1.0 + self._exploration_c  # Optimistic prior
                recommendations.append({
                    "tool": tool_name,
                    "ucb_score": round(ucb_score, 3),
                    "success_rate": 0.5,
                    "exploration_bonus": round(self._exploration_c, 3),
                    "total_calls": 0,
                    "reason": "未使用过，探索奖励最高",
                })
                continue

            # UCB1 formula: success_rate + C * sqrt(ln(total) / tool_trials)
            exploitation = stats.success_rate
            exploration = self._exploration_c * math.sqrt(
                math.log(max(self._total_trials, 1)) / max(stats.total_calls, 1)
            )

            # Task affinity bonus: if this tool worked well for similar tasks
            affinity_bonus = 0.0
            if task_pattern and task_pattern in stats.task_affinity:
                affinity_count = stats.task_affinity[task_pattern]
                affinity_bonus = min(0.2, affinity_count * 0.05)  # Cap at 0.2

            ucb_score = exploitation + exploration + affinity_bonus

            # Latency penalty for very slow tools (soft penalty)
            if stats.avg_latency_s > 30:
                ucb_score -= 0.05

            reason_parts = []
            if exploitation > 0.7:
                reason_parts.append(f"成功率高({exploitation:.0%})")
            if exploration > 0.5:
                reason_parts.append(f"探索价值高(使用{stats.total_calls}次)")
            if affinity_bonus > 0:
                reason_parts.append(f"同类任务匹配")

            recommendations.append({
                "tool": tool_name,
                "ucb_score": round(ucb_score, 3),
                "success_rate": round(exploitation, 3),
                "exploration_bonus": round(exploration, 3),
                "affinity_bonus": round(affinity_bonus, 3),
                "total_calls": stats.total_calls,
                "avg_latency_s": round(stats.avg_latency_s, 1),
                "reason": "；".join(reason_parts) if reason_parts else "综合评分",
            })

        # Sort by UCB score descending
        recommendations.sort(key=lambda r: r["ucb_score"], reverse=True)
        return recommendations[:top_k]

    def suggest_params(self, tool_name: str, current_args: dict) -> dict[str, Any] | None:
        """Suggest optimized parameters based on historical patterns.

        Finds the most successful parameter pattern for this tool and suggests
        adjustments to the current args.
        """
        stats = self._stats.get(tool_name)
        if stats is None or not stats.param_patterns:
            return None

        # Find best parameter pattern
        best_pattern = None
        best_success_rate = 0.0

        for param_key, pattern in stats.param_patterns.items():
            total = pattern["successes"] + pattern["failures"]
            if total < 3:  # Minimum sample size
                continue
            rate = pattern["successes"] / total
            if rate > best_success_rate:
                best_success_rate = rate
                best_pattern = pattern

        if best_pattern is None or best_success_rate <= 0.5:
            return None

        return {
            "suggested_args": best_pattern.get("args_sample", {}),
            "success_rate": round(best_success_rate, 3),
            "sample_size": best_pattern["successes"] + best_pattern["failures"],
        }

    def get_stats_summary(self) -> dict[str, Any]:
        """Get a summary of all tool learning statistics."""
        tools = []
        for name, stats in sorted(
            self._stats.items(), key=lambda x: x[1].total_calls, reverse=True
        ):
            tools.append({
                "name": name,
                "total_calls": stats.total_calls,
                "success_rate": round(stats.success_rate, 3),
                "avg_latency_s": round(stats.avg_latency_s, 1),
                "top_tasks": sorted(
                    stats.task_affinity.items(),
                    key=lambda x: x[1], reverse=True,
                )[:3],
            })
        return {
            "total_trials": self._total_trials,
            "exploration_c": self._exploration_c,
            "tools": tools,
        }

    def _apply_decay(self) -> None:
        """Decay old statistics to weight recent data more heavily."""
        for stats in self._stats.values():
            stats.successes = int(stats.successes * self._decay_factor)
            stats.failures = int(stats.failures * self._decay_factor)
            stats.total_calls = stats.successes + stats.failures
        self._total_trials = sum(s.total_calls for s in self._stats.values())
        self._last_decay_time = time.time()
        logger.info("Tool stats decayed (factor=%.3f)", self._decay_factor)

    def _param_hash(self, args: dict) -> str:
        """Create a stable hash for parameter patterns."""
        if not args:
            return ""
        # Use sorted keys for stability, truncate values
        parts = []
        for k in sorted(args.keys()):
            v = str(args[k])[:30]
            parts.append(f"{k}={v}")
        return "|".join(parts)

    def _extract_task_pattern(self, task: str) -> str:
        """Extract a coarse task pattern for affinity matching."""
        task_lower = task.lower()
        patterns = [
            ("search", ["搜索", "查找", "search", "find", "lookup"]),
            ("code", ["代码", "编程", "code", "program", "implement", "debug"]),
            ("analysis", ["分析", "统计", "analyze", "data", "compare"]),
            ("writing", ["写", "文档", "write", "document", "translate"]),
            ("research", ["调研", "研究", "research", "investigate"]),
        ]
        for pattern_name, keywords in patterns:
            if any(kw in task_lower for kw in keywords):
                return pattern_name
        return "general"

    # ── Serialization for persistence ──

    def to_dict(self) -> dict:
        """Serialize to dict for Redis/file storage."""
        return {
            "total_trials": self._total_trials,
            "exploration_c": self._exploration_c,
            "stats": {
                name: {
                    "total_calls": s.total_calls,
                    "successes": s.successes,
                    "failures": s.failures,
                    "total_latency_s": s.total_latency_s,
                    "param_patterns": s.param_patterns,
                    "task_affinity": s.task_affinity,
                }
                for name, s in self._stats.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolLearner:
        """Deserialize from dict."""
        learner = cls(exploration_c=data.get("exploration_c", 1.4))
        learner._total_trials = data.get("total_trials", 0)
        for name, s in data.get("stats", {}).items():
            stats = ToolStats(name=name)
            stats.total_calls = s.get("total_calls", 0)
            stats.successes = s.get("successes", 0)
            stats.failures = s.get("failures", 0)
            stats.total_latency_s = s.get("total_latency_s", 0)
            stats.param_patterns = s.get("param_patterns", {})
            stats.task_affinity = s.get("task_affinity", {})
            learner._stats[name] = stats
        return learner

    async def save_to_redis(self) -> bool:
        """Persist learning state to Redis."""
        from common.redis_client import cache_set
        data = json.dumps(self.to_dict(), ensure_ascii=False)
        return await cache_set("tool_learning:state", data, ttl=86400 * 30)  # 30 days

    @classmethod
    async def load_from_redis(cls) -> ToolLearner:
        """Load learning state from Redis."""
        from common.redis_client import cache_get
        raw = await cache_get("tool_learning:state")
        if raw:
            try:
                return cls.from_dict(json.loads(raw))
            except Exception as e:
                logger.warning("Failed to load tool learning state: %s", e)
        return cls()


# Global instance
_learner: ToolLearner | None = None


async def get_tool_learner() -> ToolLearner:
    """Get or initialize the global tool learner."""
    global _learner
    if _learner is None:
        _learner = await ToolLearner.load_from_redis()
    return _learner
