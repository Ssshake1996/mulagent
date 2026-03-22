"""Experience extractor — abstracts patterns from execution traces into the case library.

Uses LLM to extract reusable patterns from successful task executions,
then stores them as vectors in Qdrant for semantic retrieval.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

logger = logging.getLogger(__name__)


async def extract_experience(
    task_trace: dict[str, Any],
    llm=None,
) -> dict[str, Any] | None:
    """Extract a reusable experience pattern from a task trace.

    Returns a dict with: pattern, strategy, agent_combo, tips, or None if extraction fails.
    """
    if llm is None:
        return None

    from langchain_core.messages import HumanMessage, SystemMessage

    # Build richer context for extraction
    tools_used = task_trace.get("tools_used", [])
    strategies = task_trace.get("strategies_tried", [])
    self_eval = task_trace.get("self_eval", {})

    strategies_text = ""
    if strategies:
        strategy_lines = [
            f"  {s.get('tool', '?')}({s.get('args_summary', '')}) → {s.get('outcome', '?')}"
            for s in strategies[-8:]
        ]
        strategies_text = f"\nStrategies tried:\n" + "\n".join(strategy_lines)

    eval_text = ""
    if self_eval:
        eval_text = f"\nSelf-evaluation: score={self_eval.get('score', '?')}, improvement={self_eval.get('improvement', 'N/A')}"

    # ── Extract recovery/correction patterns from strategy transitions ──
    recovery_patterns = []
    correction_patterns = []
    if strategies:
        for i in range(1, len(strategies)):
            prev = strategies[i - 1]
            curr = strategies[i]
            # Recovery: fail → ok with different tool
            if prev.get("outcome") == "fail" and curr.get("outcome") == "ok":
                recovery_patterns.append(
                    f"{prev.get('tool', '?')} failed → switched to {curr.get('tool', '?')} (success)"
                )
            # Correction: same tool, fail → ok (adjusted args)
            if (prev.get("outcome") == "fail" and curr.get("outcome") == "ok"
                    and prev.get("tool") == curr.get("tool")):
                correction_patterns.append(
                    f"{curr.get('tool', '?')}: adjusted args from '{prev.get('args_summary', '')}' "
                    f"to '{curr.get('args_summary', '')}'"
                )

    recovery_text = ""
    if recovery_patterns:
        recovery_text = f"\nRecovery patterns: " + "; ".join(recovery_patterns[:3])
    correction_text = ""
    if correction_patterns:
        correction_text = f"\nCorrection patterns: " + "; ".join(correction_patterns[:3])

    prompt = f"""Analyze this completed task and extract a reusable experience pattern.

Task input: {task_trace.get('user_input', '')[:500]}
Status: {task_trace.get('status', '')}
Tools used: {', '.join(tools_used) if tools_used else 'N/A'}{strategies_text}{eval_text}{recovery_text}{correction_text}

Respond with ONLY a JSON object:
{{
  "problem_pattern": "concise description of the problem type",
  "recommended_strategy": "step-by-step strategy that worked (or should work)",
  "recommended_agents": ["list of tools/agents that work well"],
  "tips": "lessons learned, pitfalls to avoid",
  "complexity": 1-5,
  "estimated_rounds": number of ReAct rounds needed,
  "failure_patterns": ["approaches that failed and should be avoided"],
  "prerequisites": ["skills or knowledge needed for this task type"],
  "recovery_patterns": ["how to recover when initial approach fails"],
  "correction_patterns": ["how to adjust args/approach for the same tool"]
}}"""

    messages = [
        SystemMessage(content=(
            "You are an experience analyst. Extract structured, reusable patterns from task executions. "
            "Focus on what worked, what failed, and how to do better next time. "
            "Include failure patterns so future agents can avoid the same mistakes."
        )),
        HumanMessage(content=prompt),
    ]

    try:
        import asyncio
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=15)
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(content)
    except Exception as e:
        logger.warning("Experience extraction failed: %s", e)
        return None


async def store_experience(
    qdrant: QdrantClient,
    collection_name: str,
    experience: dict[str, Any],
    embedding: list[float],
    task_id: str,
) -> str:
    """Store an extracted experience in the Qdrant case library."""
    point_id = str(uuid.uuid4())
    qdrant.upsert(
        collection_name=collection_name,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "task_id": task_id,
                    "problem_pattern": experience.get("problem_pattern", ""),
                    "recommended_strategy": experience.get("recommended_strategy", ""),
                    "recommended_agents": experience.get("recommended_agents", []),
                    "tips": experience.get("tips", ""),
                    "complexity": experience.get("complexity", 3),
                    "estimated_rounds": experience.get("estimated_rounds", 5),
                    "failure_patterns": experience.get("failure_patterns", []),
                    "prerequisites": experience.get("prerequisites", []),
                    "recovery_patterns": experience.get("recovery_patterns", []),
                    "correction_patterns": experience.get("correction_patterns", []),
                },
            )
        ],
    )
    return point_id


async def search_similar_experiences(
    qdrant: QdrantClient,
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search for similar past experiences in the case library.

    Results are sorted by: vector similarity × quality_score, so high-rated
    experiences rank higher and negative experiences are flagged.
    """
    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k * 2,  # fetch extra to allow quality-based reranking
    )
    experiences = []
    for r in results.points:
        quality = r.payload.get("quality_score", 1.0)
        is_negative = quality < 0
        experiences.append({
            "score": r.score,
            "quality_score": quality,
            "is_negative": is_negative,
            "problem_pattern": r.payload.get("problem_pattern", ""),
            "recommended_strategy": r.payload.get("recommended_strategy", ""),
            "recommended_agents": r.payload.get("recommended_agents", []),
            "tips": r.payload.get("tips", ""),
            "complexity": r.payload.get("complexity", 3),
            "estimated_rounds": r.payload.get("estimated_rounds", 5),
            "failure_patterns": r.payload.get("failure_patterns", []),
            "prerequisites": r.payload.get("prerequisites", []),
            "recovery_patterns": r.payload.get("recovery_patterns", []),
            "correction_patterns": r.payload.get("correction_patterns", []),
        })

    # Sort: positive experiences by score×quality first, then negative as warnings
    positive = [e for e in experiences if not e["is_negative"]]
    negative = [e for e in experiences if e["is_negative"]]
    positive.sort(key=lambda e: e["score"] * max(e["quality_score"], 0.1), reverse=True)

    return (positive[:top_k] + negative[:1])  # top_k positive + at most 1 negative warning
