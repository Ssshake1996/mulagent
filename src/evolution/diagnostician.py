"""Evolution Diagnostician — aggregates historical data to find system weaknesses.

Analyzes traces, feedback, tool learning stats, and conversation patterns
to produce a structured diagnostic report of what needs improvement.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DiagnosticReport:
    """Structured output from system self-diagnosis."""
    # Overall stats
    total_tasks: int = 0
    success_rate: float = 0.0
    avg_duration_s: float = 0.0
    timeout_rate: float = 0.0

    # Weak areas: [{category, failure_rate, sample_inputs, common_errors}]
    weak_areas: list[dict[str, Any]] = field(default_factory=list)

    # Recurring failure patterns: [{pattern, count, last_seen}]
    failing_patterns: list[dict[str, Any]] = field(default_factory=list)

    # Tool effectiveness: [{tool, calls, success_rate, avg_latency}]
    tool_stats: list[dict[str, Any]] = field(default_factory=list)

    # Underused but available tools
    underused_tools: list[str] = field(default_factory=list)

    # Capabilities users asked for but system couldn't deliver
    missing_capabilities: list[dict[str, Any]] = field(default_factory=list)

    # Role effectiveness: [{role, delegations, success_rate}]
    role_effectiveness: list[dict[str, Any]] = field(default_factory=list)

    # Repeated questions (user asked same thing → previous answer was bad)
    repeated_questions: list[dict[str, Any]] = field(default_factory=list)

    # Knowledge gaps: areas where knowledge base was insufficient
    knowledge_gaps: list[str] = field(default_factory=list)

    # Timestamp
    diagnosed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_tasks": self.total_tasks,
            "success_rate": round(self.success_rate, 3),
            "avg_duration_s": round(self.avg_duration_s, 1),
            "timeout_rate": round(self.timeout_rate, 3),
            "weak_areas": self.weak_areas,
            "failing_patterns": self.failing_patterns,
            "tool_stats": self.tool_stats,
            "underused_tools": self.underused_tools,
            "missing_capabilities": self.missing_capabilities,
            "role_effectiveness": self.role_effectiveness,
            "repeated_questions": self.repeated_questions,
            "knowledge_gaps": self.knowledge_gaps,
            "diagnosed_at": self.diagnosed_at,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Diagnostic Report ({self.diagnosed_at}) ===",
            f"Tasks: {self.total_tasks}  Success: {self.success_rate:.0%}  "
            f"Avg: {self.avg_duration_s:.0f}s  Timeout: {self.timeout_rate:.0%}",
        ]
        if self.weak_areas:
            lines.append(f"\nWeak Areas ({len(self.weak_areas)}):")
            for w in self.weak_areas[:5]:
                lines.append(
                    f"  - {w['category']}: fail {w['failure_rate']:.0%} "
                    f"({w.get('count', '?')} tasks)"
                )
        if self.failing_patterns:
            lines.append(f"\nRecurring Failures ({len(self.failing_patterns)}):")
            for p in self.failing_patterns[:5]:
                lines.append(f"  - {p['pattern']} (x{p['count']})")
        if self.underused_tools:
            lines.append(f"\nUnderused Tools: {', '.join(self.underused_tools[:8])}")
        if self.missing_capabilities:
            lines.append(f"\nMissing Capabilities ({len(self.missing_capabilities)}):")
            for m in self.missing_capabilities[:5]:
                lines.append(f"  - {m['description']}")
        if self.knowledge_gaps:
            lines.append(f"\nKnowledge Gaps: {', '.join(self.knowledge_gaps[:5])}")
        return "\n".join(lines)


class Diagnostician:
    """Analyzes system performance data to find improvement opportunities."""

    def __init__(self, data_dir: Path | None = None):
        from common.config import DATA_DIR
        self._data_dir = data_dir or DATA_DIR

    async def diagnose(self, days: int = 7) -> DiagnosticReport:
        """Run full system diagnosis over the past N days."""
        report = DiagnosticReport(
            diagnosed_at=datetime.now(timezone.utc).isoformat()[:19],
        )

        # Gather data from all sources
        conv_stats = self._analyze_conversations(days)
        tool_data = await self._analyze_tool_learning()
        trace_data = await self._analyze_traces(days)

        # Merge into report
        report.total_tasks = conv_stats.get("total_tasks", 0) + trace_data.get("total_tasks", 0)
        report.success_rate = trace_data.get("success_rate", conv_stats.get("success_rate", 0))
        report.avg_duration_s = trace_data.get("avg_duration_s", 0)
        report.timeout_rate = trace_data.get("timeout_rate", 0)
        report.tool_stats = tool_data.get("tool_stats", [])
        report.underused_tools = tool_data.get("underused_tools", [])
        report.weak_areas = trace_data.get("weak_areas", []) + conv_stats.get("weak_areas", [])
        report.failing_patterns = trace_data.get("failing_patterns", [])
        report.missing_capabilities = conv_stats.get("missing_capabilities", [])
        report.repeated_questions = conv_stats.get("repeated_questions", [])
        report.knowledge_gaps = self._detect_knowledge_gaps(report)

        # Deduplicate weak areas
        seen = set()
        unique = []
        for w in report.weak_areas:
            key = w.get("category", "")
            if key not in seen:
                seen.add(key)
                unique.append(w)
        report.weak_areas = unique

        return report

    def _analyze_conversations(self, days: int) -> dict[str, Any]:
        """Analyze conversation files for patterns."""
        from graph.conversation import ConversationStore
        conv = ConversationStore()

        result: dict[str, Any] = {
            "total_tasks": 0,
            "success_rate": 0,
            "weak_areas": [],
            "missing_capabilities": [],
            "repeated_questions": [],
        }

        cutoff = time.time() - days * 86400
        conv_dir = conv.data_dir

        total_turns = 0
        failed_patterns: dict[str, int] = {}
        unknown_cmd_inputs: list[str] = []
        question_history: list[str] = []

        try:
            for json_path in conv_dir.rglob("*.json"):
                if json_path.name.startswith("_"):
                    continue
                try:
                    if json_path.stat().st_mtime < cutoff:
                        continue
                    data = json.loads(json_path.read_text())
                    turns = data.get("turns", [])
                    total_turns += len(turns)

                    for i, turn in enumerate(turns):
                        if turn.get("role") != "user":
                            continue
                        content = turn.get("content", "")
                        result["total_tasks"] += 1

                        # Detect unknown commands
                        if content.startswith("/") and i + 1 < len(turns):
                            next_turn = turns[i + 1]
                            if "unknown command" in next_turn.get("content", "").lower():
                                unknown_cmd_inputs.append(content[:80])

                        # Detect error responses
                        if i + 1 < len(turns):
                            resp = turns[i + 1].get("content", "")
                            if resp.startswith("Error:") or "failed" in resp.lower()[:50]:
                                # Categorize failure
                                cat = self._categorize_input(content)
                                failed_patterns[cat] = failed_patterns.get(cat, 0) + 1

                        # Track questions for repeat detection
                        question_history.append(content[:100].lower())

                except Exception:
                    continue
        except Exception as e:
            logger.debug("Conversation analysis error: %s", e)

        # Compute weak areas from failures
        if result["total_tasks"] > 0:
            total_fails = sum(failed_patterns.values())
            result["success_rate"] = 1.0 - total_fails / max(result["total_tasks"], 1)

            for cat, count in sorted(failed_patterns.items(), key=lambda x: -x[1]):
                if count >= 2:
                    result["weak_areas"].append({
                        "category": cat,
                        "failure_rate": count / result["total_tasks"],
                        "count": count,
                    })

        # Detect repeated questions (>50% word overlap)
        if len(question_history) > 1:
            for i in range(1, len(question_history)):
                words_a = set(question_history[i - 1].split())
                words_b = set(question_history[i].split())
                if words_a and words_b:
                    overlap = len(words_a & words_b) / max(len(words_a), len(words_b))
                    if overlap > 0.6 and len(words_a) > 3:
                        result["repeated_questions"].append({
                            "question": question_history[i][:80],
                            "overlap": round(overlap, 2),
                        })

        # Missing capabilities from unknown commands
        if unknown_cmd_inputs:
            for inp in unknown_cmd_inputs[:5]:
                result["missing_capabilities"].append({
                    "description": f"User tried: {inp}",
                    "source": "unknown_command",
                })

        return result

    async def _analyze_tool_learning(self) -> dict[str, Any]:
        """Get tool stats from the tool learner."""
        result: dict[str, Any] = {"tool_stats": [], "underused_tools": []}
        try:
            from evolution.tool_learning import get_tool_learner
            learner = await get_tool_learner()
            summary = learner.get_stats_summary()

            all_tool_names = set()
            try:
                from tools.registry import get_default_tools
                registry = get_default_tools()
                all_tool_names = set(registry.as_dict().keys())
            except Exception:
                pass

            used_tools = set()
            for ts in summary.get("tools", []):
                used_tools.add(ts["name"])
                result["tool_stats"].append({
                    "tool": ts["name"],
                    "calls": ts["total_calls"],
                    "success_rate": ts["success_rate"],
                    "avg_latency": ts["avg_latency_s"],
                })

            result["underused_tools"] = sorted(all_tool_names - used_tools)
        except Exception as e:
            logger.debug("Tool learning analysis error: %s", e)
        return result

    async def _analyze_traces(self, days: int) -> dict[str, Any]:
        """Analyze task traces from the database."""
        result: dict[str, Any] = {
            "total_tasks": 0,
            "success_rate": 0,
            "avg_duration_s": 0,
            "timeout_rate": 0,
            "weak_areas": [],
            "failing_patterns": [],
        }
        try:
            from common.db import get_db_session_factory
            factory = get_db_session_factory()
            if factory is None:
                return result

            from sqlalchemy import select, func
            from models.trace import TaskTrace

            async with factory() as db:
                cutoff = datetime.now(timezone.utc) - timedelta(days=days)
                stmt = select(TaskTrace).where(TaskTrace.created_at >= cutoff)
                rows = (await db.execute(stmt)).scalars().all()

                if not rows:
                    return result

                result["total_tasks"] = len(rows)
                completed = [r for r in rows if r.status == "completed"]
                failed = [r for r in rows if r.status == "failed"]
                timed_out = [r for r in rows if "timeout" in (r.status or "").lower()
                             or "timed out" in (r.final_output or "").lower()]

                result["success_rate"] = len(completed) / len(rows) if rows else 0
                result["timeout_rate"] = len(timed_out) / len(rows) if rows else 0

                # Average duration
                durations = []
                for r in rows:
                    if r.completed_at and r.created_at:
                        d = (r.completed_at - r.created_at).total_seconds()
                        if 0 < d < 3600:
                            durations.append(d)
                if durations:
                    result["avg_duration_s"] = sum(durations) / len(durations)

                # Failure pattern analysis
                fail_categories: dict[str, list[str]] = {}
                for r in failed:
                    cat = self._categorize_input(r.user_input or "")
                    fail_categories.setdefault(cat, []).append(
                        (r.final_output or "")[:100]
                    )

                for cat, errors in sorted(fail_categories.items(), key=lambda x: -len(x[1])):
                    if len(errors) >= 2:
                        result["failing_patterns"].append({
                            "pattern": cat,
                            "count": len(errors),
                            "sample_error": errors[0][:100],
                        })
                        result["weak_areas"].append({
                            "category": cat,
                            "failure_rate": len(errors) / result["total_tasks"],
                            "count": len(errors),
                        })

        except Exception as e:
            logger.debug("Trace analysis error: %s", e)
        return result

    def _detect_knowledge_gaps(self, report: DiagnosticReport) -> list[str]:
        """Infer knowledge gaps from failure patterns."""
        gaps = []
        for w in report.weak_areas:
            cat = w.get("category", "")
            if w.get("failure_rate", 0) > 0.3:
                gaps.append(cat)
        return gaps

    def _categorize_input(self, text: str) -> str:
        """Coarse categorization of user input."""
        t = text.lower()[:200]
        categories = [
            ("coding", ["代码", "code", "实现", "implement", "debug", "编程", "bug"]),
            ("writing", ["写", "write", "文章", "报告", "翻译", "translate"]),
            ("search", ["搜索", "search", "查找", "查询", "find"]),
            ("analysis", ["分析", "analyze", "对比", "compare", "统计"]),
            ("data", ["数据", "data", "csv", "excel", "sql", "database"]),
            ("devops", ["部署", "deploy", "docker", "k8s", "ci/cd", "运维"]),
            ("design", ["设计", "design", "架构", "architecture"]),
        ]
        for name, keywords in categories:
            if any(kw in t for kw in keywords):
                return name
        return "general"
