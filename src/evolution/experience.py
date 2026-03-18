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

    prompt = f"""Analyze this completed task and extract a reusable pattern.

Task input: {task_trace.get('user_input', '')}
Intent: {task_trace.get('intent', '')}
Subtasks: {json.dumps(task_trace.get('subtasks', []), ensure_ascii=False, default=str)}
Status: {task_trace.get('status', '')}

Respond with ONLY a JSON object:
{{
  "problem_pattern": "brief description of what type of problem this is",
  "recommended_strategy": "how to decompose this type of problem",
  "recommended_agents": ["list of agent types that work well"],
  "tips": "any lessons learned or tips for similar problems"
}}"""

    messages = [
        SystemMessage(content="You are an experience analyst. Extract reusable patterns from task executions."),
        HumanMessage(content=prompt),
    ]

    try:
        response = await llm.ainvoke(messages)
        return json.loads(response.content)
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
        })

    # Sort: positive experiences by score×quality first, then negative as warnings
    positive = [e for e in experiences if not e["is_negative"]]
    negative = [e for e in experiences if e["is_negative"]]
    positive.sort(key=lambda e: e["score"] * max(e["quality_score"], 0.1), reverse=True)

    return (positive[:top_k] + negative[:1])  # top_k positive + at most 1 negative warning
