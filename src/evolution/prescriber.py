"""Evolution Prescriber — converts diagnostic reports into actionable improvements.

Uses LLM to analyze system weaknesses and generate concrete evolution actions
(prompt refinements, new skills, tool additions, config tuning).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Evolution:
    """A single evolution action to apply."""
    type: str          # prompt_refine | add_skill | add_tool | update_knowledge | tune_params
    target: str        # file path or config key
    reason: str        # why this change is needed
    patch: Any         # content to apply (str for files, dict for config)
    priority: int = 2  # 1=critical, 2=important, 3=nice-to-have
    confidence: float = 0.5  # 0-1, how confident the prescriber is

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "target": self.target,
            "reason": self.reason,
            "patch": self.patch if isinstance(self.patch, (str, dict, list)) else str(self.patch),
            "priority": self.priority,
            "confidence": self.confidence,
        }

    def summary(self) -> str:
        return f"[P{self.priority}|{self.confidence:.0%}] {self.type}: {self.target} — {self.reason[:80]}"


class Prescriber:
    """Generates evolution actions from diagnostic data."""

    async def prescribe(
        self,
        report_dict: dict[str, Any],
        llm=None,
    ) -> list[Evolution]:
        """Generate evolution actions from a diagnostic report.

        Uses LLM for intelligent analysis if available,
        falls back to rule-based prescriptions otherwise.
        """
        evolutions: list[Evolution] = []

        # Rule-based prescriptions (always run)
        evolutions.extend(self._prescribe_timeout_tuning(report_dict))
        evolutions.extend(self._prescribe_underused_tools(report_dict))

        # LLM-powered prescriptions
        if llm is not None:
            llm_evolutions = await self._prescribe_with_llm(report_dict, llm)
            evolutions.extend(llm_evolutions)

        # Sort by priority then confidence
        evolutions.sort(key=lambda e: (e.priority, -e.confidence))
        return evolutions

    def _prescribe_timeout_tuning(self, report: dict) -> list[Evolution]:
        """Auto-tune timeout if timeout rate is high."""
        evolutions = []
        timeout_rate = report.get("timeout_rate", 0)
        if timeout_rate > 0.15:
            evolutions.append(Evolution(
                type="tune_params",
                target="react.timeout",
                reason=f"Timeout rate is {timeout_rate:.0%} — increase default timeout",
                patch={"timeout": 900, "tool_timeout": 180},
                priority=1,
                confidence=0.9,
            ))
        return evolutions

    def _prescribe_underused_tools(self, report: dict) -> list[Evolution]:
        """Suggest knowledge updates for underused tools."""
        evolutions = []
        underused = report.get("underused_tools", [])
        if len(underused) > 5:
            evolutions.append(Evolution(
                type="update_knowledge",
                target="config/knowledge/tools_guide.md",
                reason=f"{len(underused)} tools never used: {', '.join(underused[:5])}...",
                patch=self._build_tools_guide(underused),
                priority=3,
                confidence=0.6,
            ))
        return evolutions

    def _build_tools_guide(self, tools: list[str]) -> str:
        lines = ["# Available Tools Guide\n",
                 "The following tools are available but rarely used. "
                 "Consider using them when appropriate:\n"]
        for t in tools:
            lines.append(f"- **{t}**: (add description)")
        return "\n".join(lines)

    async def _prescribe_with_llm(
        self, report: dict, llm
    ) -> list[Evolution]:
        """Use LLM to analyze the report and propose improvements."""
        from langchain_core.messages import HumanMessage, SystemMessage
        import asyncio

        # Build a concise report summary for the LLM
        summary_parts = [
            f"Tasks: {report.get('total_tasks', 0)}",
            f"Success rate: {report.get('success_rate', 0):.0%}",
            f"Timeout rate: {report.get('timeout_rate', 0):.0%}",
        ]
        if report.get("weak_areas"):
            summary_parts.append("Weak areas:")
            for w in report["weak_areas"][:5]:
                summary_parts.append(
                    f"  - {w['category']}: fail {w.get('failure_rate', 0):.0%} ({w.get('count', 0)} tasks)"
                )
        if report.get("failing_patterns"):
            summary_parts.append("Recurring failures:")
            for p in report["failing_patterns"][:5]:
                summary_parts.append(f"  - {p['pattern']} (x{p['count']})")
        if report.get("missing_capabilities"):
            summary_parts.append("Missing capabilities:")
            for m in report["missing_capabilities"][:3]:
                summary_parts.append(f"  - {m['description']}")
        if report.get("knowledge_gaps"):
            summary_parts.append(f"Knowledge gaps: {', '.join(report['knowledge_gaps'][:5])}")

        report_text = "\n".join(summary_parts)

        messages = [
            SystemMessage(content="""You are a system evolution advisor for a multi-agent AI framework.

Analyze the diagnostic report and propose CONCRETE improvements. Each improvement must be one of:
1. prompt_refine — improve an agent role's system prompt in config/agents.yaml
2. add_skill — create a new skill directory under config/skills/
3. add_tool — add an API tool entry to config/tools.yaml
4. update_knowledge — update/create a knowledge base file in config/knowledge/
5. tune_params — adjust ReAct parameters (timeout, max_rounds, etc.)

Respond with ONLY a JSON array of improvements:
[
  {
    "type": "prompt_refine",
    "target": "roles.coder",
    "reason": "why this helps (cite data from report)",
    "patch": "the new/modified prompt text or config snippet",
    "priority": 1-3,
    "confidence": 0.0-1.0
  }
]

Rules:
- Only propose changes supported by evidence in the report
- Be specific — include actual prompt text, YAML config, or markdown content
- Prioritize high-impact, low-risk changes
- Set confidence based on how strong the evidence is
- Maximum 5 proposals"""),
            HumanMessage(content=f"Diagnostic Report:\n{report_text}"),
        ]

        try:
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=30)
            content = response.content.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )
            proposals = json.loads(content)
            if not isinstance(proposals, list):
                return []

            evolutions = []
            for p in proposals[:5]:
                if not isinstance(p, dict):
                    continue
                evo_type = p.get("type", "")
                if evo_type not in (
                    "prompt_refine", "add_skill", "add_tool",
                    "update_knowledge", "tune_params",
                ):
                    continue
                evolutions.append(Evolution(
                    type=evo_type,
                    target=p.get("target", ""),
                    reason=p.get("reason", ""),
                    patch=p.get("patch", ""),
                    priority=min(3, max(1, int(p.get("priority", 2)))),
                    confidence=min(1.0, max(0.0, float(p.get("confidence", 0.5)))),
                ))
            return evolutions
        except Exception as e:
            logger.warning("LLM prescription failed: %s", e)
            return []
