"""Feedback-driven evolution — closes the loop from user feedback to system improvement.

Three mechanisms:
1. Low-rating retrospective: LLM analyzes why a task failed, stores negative experience
2. Agent stats update: feedback adjusts agent success_rate in registry
3. Experience quality: high-rating boosts experience weight, low-rating demotes
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agents.registry import AgentRegistry
from models.trace import SubtaskTrace, TaskTrace

logger = logging.getLogger(__name__)


async def process_feedback(
    rating: int,
    task_id: uuid.UUID,
    comment: str | None,
    *,
    db: AsyncSession,
    registry: AgentRegistry | None = None,
    qdrant: QdrantClient | None = None,
    collection_name: str = "case_library",
    llm=None,
) -> dict[str, Any]:
    """Process user feedback and trigger evolution actions.

    Returns a summary of actions taken.
    """
    actions = []

    # 1. Load the original task trace
    trace = await _load_trace(db, task_id)
    if trace is None:
        logger.warning("Feedback for unknown task %s, skipping evolution", task_id)
        return {"actions": actions, "error": "task_trace_not_found"}

    # 2. Update agent stats based on feedback
    if registry is not None:
        agent_updates = await _update_agent_stats(db, task_id, rating, registry)
        if agent_updates:
            actions.append({"type": "agent_stats_updated", "agents": agent_updates})

    # 3. Low rating → retrospective analysis
    if rating <= 2 and llm is not None and qdrant is not None:
        retro = await _retrospective(trace, comment, llm, qdrant, collection_name)
        if retro:
            actions.append({"type": "retrospective_stored", "pattern": retro.get("failure_pattern", "")})

    # 4. High rating → boost experience quality in Qdrant
    if rating >= 4 and qdrant is not None:
        boosted = _boost_experience(qdrant, collection_name, str(task_id), rating)
        if boosted:
            actions.append({"type": "experience_boosted", "task_id": str(task_id)})

    logger.info("Feedback processed for task %s: %d actions", task_id, len(actions))
    return {"actions": actions}


async def _load_trace(db: AsyncSession, task_id: uuid.UUID) -> TaskTrace | None:
    """Load a task trace by ID."""
    result = await db.execute(
        select(TaskTrace).where(TaskTrace.id == task_id)
    )
    return result.scalar_one_or_none()


async def _update_agent_stats(
    db: AsyncSession,
    task_id: uuid.UUID,
    rating: int,
    registry: AgentRegistry,
) -> list[dict[str, Any]]:
    """Update agent success_rate based on user feedback rating.

    rating 4-5 → success, rating 1-2 → failure, rating 3 → neutral (skip)
    """
    if rating == 3:
        return []

    success = rating >= 4
    result = await db.execute(
        select(SubtaskTrace.agent_id).where(SubtaskTrace.task_id == task_id).distinct()
    )
    agent_ids = [row[0] for row in result.all()]

    updates = []
    for agent_id in agent_ids:
        # Try exact match first, then try with "_agent" suffix
        agent = registry.get(agent_id)
        if agent is None:
            agent = registry.get(f"{agent_id}_agent")
        if agent is None:
            logger.debug("Agent %s not found in registry, skipping stats update", agent_id)
            continue

        registry.update_stats(agent.id, success)
        updates.append({
            "agent_id": agent.id,
            "new_success_rate": round(agent.success_rate, 4),
            "total_runs": agent.total_runs,
        })
        logger.info(
            "Agent %s stats updated: success_rate=%.4f (feedback=%d)",
            agent.id, agent.success_rate, rating,
        )

    return updates


async def _retrospective(
    trace: TaskTrace,
    comment: str | None,
    llm,
    qdrant: QdrantClient,
    collection_name: str,
) -> dict[str, Any] | None:
    """LLM-powered retrospective for low-rating tasks.

    Analyzes what went wrong and stores a negative experience to avoid repeating mistakes.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    feedback_info = f"\nUser comment: {comment}" if comment else ""

    messages = [
        SystemMessage(content=(
            "You are a task failure analyst. Analyze why this task received poor feedback "
            "and extract lessons to avoid similar failures.\n"
            "Respond with ONLY a JSON object:\n"
            '{"failure_pattern": "what type of failure", '
            '"root_cause": "why it failed", '
            '"avoidance_strategy": "how to avoid this in the future", '
            '"affected_agents": ["agent types involved"]}'
        )),
        HumanMessage(content=(
            f"Task input: {trace.user_input}\n"
            f"Intent: {trace.intent_category}\n"
            f"Status: {trace.status}\n"
            f"Output (first 500 chars): {(trace.final_output or '')[:500]}\n"
            f"{feedback_info}"
        )),
    ]

    try:
        response = await llm.ainvoke(messages)
        retro = json.loads(response.content)

        # Store as negative experience in Qdrant
        from common.vector import text_to_embedding

        embedding = text_to_embedding(trace.user_input)
        point_id = str(uuid.uuid4())
        qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "task_id": str(trace.id),
                        "problem_pattern": f"[NEGATIVE] {retro.get('failure_pattern', '')}",
                        "recommended_strategy": retro.get("avoidance_strategy", ""),
                        "recommended_agents": retro.get("affected_agents", []),
                        "tips": f"ROOT CAUSE: {retro.get('root_cause', '')}",
                        "quality_score": -1.0,  # negative marker
                    },
                )
            ],
        )
        logger.info("Retrospective stored for task %s", trace.id)
        return retro
    except Exception as e:
        logger.warning("Retrospective failed: %s", e)
        return None


def _boost_experience(
    qdrant: QdrantClient,
    collection_name: str,
    task_id: str,
    rating: int,
) -> bool:
    """Boost the quality score of an experience based on positive feedback.

    Searches Qdrant for the experience matching this task_id and updates its payload.
    """
    try:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Find the experience point for this task
        results = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="task_id", match=MatchValue(value=task_id))]
            ),
            limit=1,
        )

        points, _ = results
        if not points:
            return False

        point = points[0]
        payload = dict(point.payload)
        # Increase quality score (default 1.0, max 5.0)
        current_score = payload.get("quality_score", 1.0)
        new_score = min(current_score + (rating - 3) * 0.5, 5.0)  # +0.5 for 4, +1.0 for 5
        payload["quality_score"] = new_score

        qdrant.set_payload(
            collection_name=collection_name,
            payload={"quality_score": new_score},
            points=[point.id],
        )
        logger.info("Experience %s quality boosted: %.1f → %.1f", point.id, current_score, new_score)
        return True
    except Exception as e:
        logger.warning("Experience boost failed: %s", e)
        return False
