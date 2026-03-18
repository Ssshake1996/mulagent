"""Three-level skill acquisition: history → web search → external skill.

When an agent lacks the capability for a task, this module tries to
fill the gap through a priority chain:
  L1: Qdrant case library (fast, trusted)
  L2: Web search via mcporter (medium, public knowledge)
  L3: External skill code (slow, requires security review)
"""

from __future__ import annotations

import asyncio
import json as _json
import logging
import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class AcquisitionLevel(str, Enum):
    HISTORY = "history"
    WEB_SEARCH = "web_search"
    EXTERNAL_SKILL = "external_skill"


@dataclass
class AcquisitionResult:
    level: AcquisitionLevel
    found: bool
    content: str
    metadata: dict[str, Any] | None = None


class SkillAcquirer:
    """Tries to acquire missing capabilities in priority order."""

    def __init__(self, qdrant=None, collection_name: str = "case_library", llm=None):
        self._qdrant = qdrant
        self._collection_name = collection_name
        self._llm = llm

    async def acquire(self, task_description: str, required_skills: list[str]) -> AcquisitionResult:
        """Try to fill a skill gap through the 3-level priority chain."""
        # Level 1: Check case library
        result = await self._try_history(task_description)
        if result.found:
            return result

        # Level 2: Web search
        result = await self._try_web_search(task_description)
        if result.found:
            return result

        # Level 3: External skill
        return await self._try_external_skill(task_description, required_skills)

    async def _try_history(self, task: str) -> AcquisitionResult:
        """L1: Search case library for similar solved tasks."""
        if self._qdrant is None:
            return AcquisitionResult(level=AcquisitionLevel.HISTORY, found=False, content="")

        try:
            from common.vector import text_to_embedding
            from evolution.experience import search_similar_experiences

            query_vec = text_to_embedding(task)
            experiences = await search_similar_experiences(
                self._qdrant, self._collection_name, query_vec, top_k=3,
            )

            # Filter for high-relevance results
            relevant = [e for e in experiences if e.get("score", 0) > 0.6]
            if not relevant:
                return AcquisitionResult(level=AcquisitionLevel.HISTORY, found=False, content="")

            # Format as actionable context
            parts = []
            for exp in relevant:
                parts.append(
                    f"Pattern: {exp['problem_pattern']}\n"
                    f"Strategy: {exp['recommended_strategy']}\n"
                    f"Agents: {exp['recommended_agents']}\n"
                    f"Tips: {exp['tips']}"
                )
            content = "\n---\n".join(parts)

            logger.info("L1 history: found %d relevant experiences", len(relevant))
            return AcquisitionResult(
                level=AcquisitionLevel.HISTORY,
                found=True,
                content=content,
                metadata={"experience_count": len(relevant), "top_score": relevant[0]["score"]},
            )
        except Exception as e:
            logger.warning("L1 history search failed: %s", e)
            return AcquisitionResult(level=AcquisitionLevel.HISTORY, found=False, content="")

    async def _try_web_search(self, task: str) -> AcquisitionResult:
        """L2: Search the web for relevant knowledge via mcporter (bailian_web_search)."""
        try:
            if not shutil.which("mcporter"):
                logger.debug("mcporter not installed, skipping L2")
                return AcquisitionResult(level=AcquisitionLevel.WEB_SEARCH, found=False, content="")

            proc = await asyncio.create_subprocess_exec(
                "mcporter", "call", "WebSearch.bailian_web_search",
                f"query={task}", "--output", "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

            if proc.returncode != 0:
                logger.warning("L2 mcporter search failed: %s", stderr.decode(errors="replace").strip())
                return AcquisitionResult(level=AcquisitionLevel.WEB_SEARCH, found=False, content="")

            data = _json.loads(stdout.decode())
            pages = data.get("pages", [])

            if not pages:
                return AcquisitionResult(level=AcquisitionLevel.WEB_SEARCH, found=False, content="")

            # Format search results as context
            parts = []
            for p in pages[:5]:
                title = p.get("title", "")
                snippet = p.get("snippet", "")[:500]
                url = p.get("url", "")
                parts.append(f"**{title}**\n{snippet}\nSource: {url}")
            content = "\n\n".join(parts)

            logger.info("L2 mcporter search: found %d results", len(pages))
            return AcquisitionResult(
                level=AcquisitionLevel.WEB_SEARCH,
                found=True,
                content=content,
                metadata={"result_count": len(pages)},
            )
        except Exception as e:
            logger.warning("L2 web search failed: %s", e)
            return AcquisitionResult(level=AcquisitionLevel.WEB_SEARCH, found=False, content="")

    async def _try_external_skill(self, task: str, skills: list[str]) -> AcquisitionResult:
        """L3: Try to find and load external skill code.

        Uses LLM to generate a skill snippet, then runs lightweight security review.
        If review is uncertain, marks for human approval.
        """
        if self._llm is None:
            return AcquisitionResult(
                level=AcquisitionLevel.EXTERNAL_SKILL,
                found=False,
                content="No LLM available for skill generation",
            )

        try:
            # Ask LLM to generate a skill snippet
            code_snippet = await self._generate_skill(task, skills)
            if not code_snippet:
                return AcquisitionResult(
                    level=AcquisitionLevel.EXTERNAL_SKILL, found=False, content="",
                )

            # Security review
            from agents.skill_security import review_skill

            verdict = await review_skill(code_snippet, source="llm_generated", llm=self._llm)

            if verdict.approved:
                logger.info("L3 skill approved: %s", verdict.reason)
                return AcquisitionResult(
                    level=AcquisitionLevel.EXTERNAL_SKILL,
                    found=True,
                    content=code_snippet,
                    metadata={"security": verdict.risk_level.value, "reason": verdict.reason},
                )
            elif verdict.needs_human_review:
                logger.info("L3 skill needs human review: %s", verdict.reason)
                return AcquisitionResult(
                    level=AcquisitionLevel.EXTERNAL_SKILL,
                    found=False,
                    content=f"[NEEDS_HUMAN_REVIEW] {verdict.reason}\n\nCode:\n{code_snippet}",
                    metadata={"security": verdict.risk_level.value, "needs_review": True},
                )
            else:
                logger.warning("L3 skill rejected: %s", verdict.reason)
                return AcquisitionResult(
                    level=AcquisitionLevel.EXTERNAL_SKILL,
                    found=False,
                    content=f"Rejected: {verdict.reason}",
                    metadata={"security": verdict.risk_level.value},
                )
        except Exception as e:
            logger.warning("L3 external skill failed: %s", e)
            return AcquisitionResult(
                level=AcquisitionLevel.EXTERNAL_SKILL, found=False, content=str(e),
            )

    async def _generate_skill(self, task: str, skills: list[str]) -> str | None:
        """Ask LLM to generate a helper code snippet for the task."""
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=(
                "You are a Python skill generator. Given a task description, write a short, "
                "self-contained Python function that helps solve it.\n"
                "Rules:\n"
                "- Pure computation only, no file I/O, no network calls, no subprocess\n"
                "- Max 50 lines\n"
                "- Include a docstring\n"
                "- Return ONLY the Python code, no explanation"
            )),
            HumanMessage(content=f"Task: {task}\nRequired skills: {skills}"),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            code = response.content.strip()
            # Strip markdown code fences if present
            if code.startswith("```"):
                lines = code.split("\n")
                code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
            return code if code else None
        except Exception as e:
            logger.warning("Skill generation failed: %s", e)
            return None
