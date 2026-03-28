"""Experience extractor — tiered & graded experience accumulation system.

Three tiers of experience:
  L1 (Atomic)   — Single tool call patterns: "read_file on large file → use offset"
  L2 (Strategy) — Multi-step task strategies: "deploy task → check status, pull, restart"
  L3 (Domain)   — Cross-task domain knowledge: "this codebase uses X pattern for Y"

Grading dimensions:
  - quality_score: overall quality (boosted by positive feedback, decayed by negative)
  - use_count: how many times this experience was retrieved and used
  - success_rate: ratio of positive outcomes when this experience was applied
  - freshness: timestamp-based, newer experiences rank higher

Lifecycle: L1 auto-extracted → L2 promoted when pattern recurs → L3 consolidated by LLM.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
)

logger = logging.getLogger(__name__)

# ── Tier definitions ────────────────────────────────────────────

TIER_L1 = 1  # Atomic: single tool pattern
TIER_L2 = 2  # Strategy: multi-step task
TIER_L3 = 3  # Domain: cross-task knowledge

TIER_NAMES = {TIER_L1: "atomic", TIER_L2: "strategy", TIER_L3: "domain"}

# ── Freshness decay parameters ──────────────────────────────────

_HALFLIFE_DAYS = 30  # quality halves every 30 days of non-use
_USE_BOOST = 0.1     # each retrieval adds this to quality


# ── Tier classification ─────────────────────────────────────────

def _classify_tier(task_trace: dict[str, Any], experience: dict[str, Any]) -> int:
    """Determine the tier of an extracted experience.

    L1: single tool used, low complexity
    L2: multiple tools or complexity >= 3
    L3: only created via consolidation (never auto-classified)
    """
    tools_used = task_trace.get("tools_used", [])
    strategies = task_trace.get("strategies_tried", [])
    complexity = experience.get("complexity", 3)

    unique_tools = len(set(tools_used)) if tools_used else 0
    total_steps = len(strategies) if strategies else 0

    # L1: simple, single-tool tasks
    if unique_tools <= 1 and total_steps <= 2 and complexity <= 2:
        return TIER_L1

    # L2: multi-step strategies (default)
    return TIER_L2


# ── Experience extraction ───────────────────────────────────────

async def extract_experience(
    task_trace: dict[str, Any],
    llm=None,
) -> dict[str, Any] | None:
    """Extract a reusable experience pattern from a task trace.

    Automatically classifies into L1 (atomic) or L2 (strategy) tier.
    Returns a dict with tier, pattern, strategy, tips, etc., or None.
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
        eval_text = (
            f"\nSelf-evaluation: score={self_eval.get('score', '?')}, "
            f"improvement={self_eval.get('improvement', 'N/A')}"
        )

    # ── Extract recovery/correction patterns from strategy transitions ──
    recovery_patterns = []
    correction_patterns = []
    if strategies:
        for i in range(1, len(strategies)):
            prev = strategies[i - 1]
            curr = strategies[i]
            if prev.get("outcome") == "fail" and curr.get("outcome") == "ok":
                recovery_patterns.append(
                    f"{prev.get('tool', '?')} failed → switched to {curr.get('tool', '?')} (success)"
                )
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
  "domain_tags": ["tag1", "tag2"],
  "recommended_strategy": "step-by-step strategy that worked (or should work)",
  "recommended_agents": ["list of tools/agents that work well"],
  "tips": "lessons learned, pitfalls to avoid",
  "complexity": 1-5,
  "estimated_rounds": number of ReAct rounds needed,
  "failure_patterns": ["approaches that failed and should be avoided"],
  "prerequisites": ["skills or knowledge needed for this task type"],
  "recovery_patterns": ["how to recover when initial approach fails"],
  "correction_patterns": ["how to adjust args/approach for the same tool"]
}}

Rules:
- domain_tags: 2-5 short tags for the problem domain (e.g. ["file_ops", "python", "deploy"])
- complexity: 1=trivial single step, 2=simple, 3=moderate, 4=complex multi-step, 5=very complex"""

    messages = [
        SystemMessage(content=(
            "You are an experience analyst. Extract structured, reusable patterns from task executions. "
            "Focus on what worked, what failed, and how to do better next time. "
            "Include failure patterns so future agents can avoid the same mistakes. "
            "Assign domain_tags to enable cross-task knowledge grouping."
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
        experience = json.loads(content)

        # Auto-classify tier
        experience["tier"] = _classify_tier(task_trace, experience)
        return experience
    except Exception as e:
        logger.warning("Experience extraction failed: %s", e)
        return None


# ── Storage ─────────────────────────────────────────────────────

async def store_experience(
    qdrant: QdrantClient,
    collection_name: str,
    experience: dict[str, Any],
    embedding: list[float],
    task_id: str,
) -> str:
    """Store an extracted experience with tier metadata and grading fields."""
    now = time.time()
    tier = experience.get("tier", TIER_L2)
    point_id = str(uuid.uuid4())

    payload = {
        "task_id": task_id,
        "tier": tier,
        "tier_name": TIER_NAMES.get(tier, "strategy"),
        "problem_pattern": experience.get("problem_pattern", ""),
        "domain_tags": experience.get("domain_tags", []),
        "recommended_strategy": experience.get("recommended_strategy", ""),
        "recommended_agents": experience.get("recommended_agents", []),
        "tips": experience.get("tips", ""),
        "complexity": experience.get("complexity", 3),
        "estimated_rounds": experience.get("estimated_rounds", 5),
        "failure_patterns": experience.get("failure_patterns", []),
        "prerequisites": experience.get("prerequisites", []),
        "recovery_patterns": experience.get("recovery_patterns", []),
        "correction_patterns": experience.get("correction_patterns", []),
        # ── Grading fields ──
        "quality_score": 1.0,
        "use_count": 0,
        "success_count": 0,
        "fail_count": 0,
        "created_at": now,
        "last_used_at": now,
        "promoted_from": None,  # set when L1→L2 or L2→L3
    }

    # ── Deduplication: check for similar existing experience ──
    merged = await _try_merge(qdrant, collection_name, embedding, payload)
    if merged:
        logger.info("Experience merged into existing point (tier=%s)", TIER_NAMES.get(tier))
        return merged

    qdrant.upsert(
        collection_name=collection_name,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
    )
    logger.info("Experience stored: tier=%s, id=%s", TIER_NAMES.get(tier), point_id[:8])
    return point_id


async def _try_merge(
    qdrant: QdrantClient,
    collection_name: str,
    embedding: list[float],
    new_payload: dict[str, Any],
    similarity_threshold: float = 0.85,
) -> str | None:
    """If a very similar experience exists in the same tier, merge instead of duplicating.

    Merge strategy:
    - Increment use_count
    - Keep the higher-quality tips/strategy
    - Union domain_tags, failure_patterns, recovery_patterns
    - Average complexity and estimated_rounds
    Returns the merged point ID, or None if no match found.
    """
    try:
        results = qdrant.query_points(
            collection_name=collection_name,
            query=embedding,
            limit=3,
            query_filter=Filter(
                must=[
                    FieldCondition(key="tier", match=MatchValue(value=new_payload.get("tier", TIER_L2)))
                ]
            ),
        )

        for r in results.points:
            if r.score < similarity_threshold:
                continue

            # Found a very similar experience — merge
            old = dict(r.payload)
            merged_payload = {
                **old,
                "use_count": old.get("use_count", 0) + 1,
                "last_used_at": time.time(),
                # Union list fields
                "domain_tags": list(set(old.get("domain_tags", []) + new_payload.get("domain_tags", []))),
                "failure_patterns": list(set(
                    old.get("failure_patterns", []) + new_payload.get("failure_patterns", [])
                ))[:10],
                "recovery_patterns": list(set(
                    old.get("recovery_patterns", []) + new_payload.get("recovery_patterns", [])
                ))[:5],
                "correction_patterns": list(set(
                    old.get("correction_patterns", []) + new_payload.get("correction_patterns", [])
                ))[:5],
                # Average numeric fields
                "complexity": round(
                    (old.get("complexity", 3) + new_payload.get("complexity", 3)) / 2, 1
                ),
                "estimated_rounds": round(
                    (old.get("estimated_rounds", 5) + new_payload.get("estimated_rounds", 5)) / 2, 1
                ),
            }

            # Keep longer/better strategy and tips (prefer new if longer)
            for field in ("recommended_strategy", "tips"):
                old_val = old.get(field, "")
                new_val = new_payload.get(field, "")
                if len(new_val) > len(old_val):
                    merged_payload[field] = new_val

            qdrant.set_payload(
                collection_name=collection_name,
                payload=merged_payload,
                points=[r.id],
            )
            return str(r.id)

    except Exception as e:
        logger.debug("Merge check failed: %s", e)

    return None


# ── Retrieval with tiered ranking ───────────────────────────────

def _effective_score(
    similarity: float,
    payload: dict[str, Any],
    prefer_tier: int | None = None,
) -> float:
    """Compute effective ranking score with freshness decay and tier preference.

    Score = similarity × quality × freshness_factor × tier_bonus
    """
    quality = max(payload.get("quality_score", 1.0), 0.1)
    use_count = payload.get("use_count", 0)
    tier = payload.get("tier", TIER_L2)

    # Freshness decay: halve score every _HALFLIFE_DAYS of non-use
    last_used = payload.get("last_used_at", payload.get("created_at", time.time()))
    days_since = (time.time() - last_used) / 86400
    freshness = 0.5 ** (days_since / _HALFLIFE_DAYS) if days_since > 0 else 1.0

    # Use count bonus (logarithmic, caps at ~2x)
    use_bonus = 1.0 + min(use_count * _USE_BOOST, 1.0)

    # Success rate bonus
    success_count = payload.get("success_count", 0)
    fail_count = payload.get("fail_count", 0)
    total = success_count + fail_count
    success_bonus = 1.0
    if total >= 3:
        rate = success_count / total
        success_bonus = 0.5 + rate  # 0.5 (all fail) to 1.5 (all success)

    # Tier preference: boost preferred tier
    tier_bonus = 1.0
    if prefer_tier is not None:
        if tier == prefer_tier:
            tier_bonus = 1.3
        elif tier == TIER_L3:
            tier_bonus = 1.1  # L3 domain knowledge always slightly preferred

    return similarity * quality * freshness * use_bonus * success_bonus * tier_bonus


async def search_similar_experiences(
    qdrant: QdrantClient,
    collection_name: str,
    query_embedding: list[float],
    top_k: int = 3,
    tier_filter: int | None = None,
    prefer_tier: int | None = None,
) -> list[dict[str, Any]]:
    """Search for similar past experiences with tiered ranking.

    Args:
        tier_filter: if set, only return experiences of this tier
        prefer_tier: if set, boost this tier in ranking (but still return all tiers)

    Returns experiences sorted by effective score (similarity × quality × freshness × tier_bonus).
    Negative experiences are appended as warnings.
    """
    query_filter = None
    if tier_filter is not None:
        query_filter = Filter(
            must=[FieldCondition(key="tier", match=MatchValue(value=tier_filter))]
        )

    results = qdrant.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=top_k * 3,  # fetch extra for quality-based reranking
        query_filter=query_filter,
    )

    experiences = []
    for r in results.points:
        payload = dict(r.payload)
        quality = payload.get("quality_score", 1.0)
        is_negative = quality < 0

        exp = {
            "point_id": str(r.id),
            "score": r.score,
            "effective_score": _effective_score(r.score, payload, prefer_tier),
            "quality_score": quality,
            "is_negative": is_negative,
            "tier": payload.get("tier", TIER_L2),
            "tier_name": payload.get("tier_name", "strategy"),
            "domain_tags": payload.get("domain_tags", []),
            "problem_pattern": payload.get("problem_pattern", ""),
            "recommended_strategy": payload.get("recommended_strategy", ""),
            "recommended_agents": payload.get("recommended_agents", []),
            "tips": payload.get("tips", ""),
            "complexity": payload.get("complexity", 3),
            "estimated_rounds": payload.get("estimated_rounds", 5),
            "failure_patterns": payload.get("failure_patterns", []),
            "prerequisites": payload.get("prerequisites", []),
            "recovery_patterns": payload.get("recovery_patterns", []),
            "correction_patterns": payload.get("correction_patterns", []),
            "use_count": payload.get("use_count", 0),
            "success_rate": _success_rate(payload),
        }
        experiences.append(exp)

    # Sort: positive by effective score, negative as warnings
    positive = [e for e in experiences if not e["is_negative"]]
    negative = [e for e in experiences if e["is_negative"]]
    positive.sort(key=lambda e: e["effective_score"], reverse=True)

    return positive[:top_k] + negative[:1]


def _success_rate(payload: dict[str, Any]) -> float | None:
    """Compute success rate if enough data."""
    s = payload.get("success_count", 0)
    f = payload.get("fail_count", 0)
    total = s + f
    if total < 2:
        return None
    return round(s / total, 2)


# ── Usage tracking ──────────────────────────────────────────────

def record_experience_used(
    qdrant: QdrantClient,
    collection_name: str,
    point_id: str,
    outcome: str = "unknown",
) -> None:
    """Record that an experience was retrieved and used.

    Call after the task completes to track effectiveness.
    outcome: "success", "fail", or "unknown"
    """
    try:
        results = qdrant.retrieve(collection_name=collection_name, ids=[point_id])
        if not results:
            return

        point = results[0]
        payload = dict(point.payload)
        payload["use_count"] = payload.get("use_count", 0) + 1
        payload["last_used_at"] = time.time()

        if outcome == "success":
            payload["success_count"] = payload.get("success_count", 0) + 1
        elif outcome == "fail":
            payload["fail_count"] = payload.get("fail_count", 0) + 1

        qdrant.set_payload(
            collection_name=collection_name,
            payload=payload,
            points=[point_id],
        )
    except Exception as e:
        logger.debug("record_experience_used failed: %s", e)


# ── Tier promotion ──────────────────────────────────────────────

async def maybe_promote(
    qdrant: QdrantClient,
    collection_name: str,
    llm=None,
) -> list[str]:
    """Check for L1 experiences that should be promoted to L2.

    Promotion criteria:
    - L1 with use_count >= 5 and success_rate >= 0.6
    - Multiple L1s share the same domain_tags → consolidate into L2

    Returns list of promoted point IDs.
    """
    promoted = []

    try:
        # Find high-use L1 experiences
        results = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="tier", match=MatchValue(value=TIER_L1)),
                    FieldCondition(key="use_count", range=Range(gte=5)),
                ]
            ),
            limit=20,
        )

        points, _ = results
        if not points:
            return promoted

        for point in points:
            payload = dict(point.payload)
            success = payload.get("success_count", 0)
            total = success + payload.get("fail_count", 0)

            if total >= 3 and (success / total) >= 0.6:
                # Promote to L2
                payload["tier"] = TIER_L2
                payload["tier_name"] = TIER_NAMES[TIER_L2]
                payload["promoted_from"] = TIER_L1

                qdrant.set_payload(
                    collection_name=collection_name,
                    payload=payload,
                    points=[point.id],
                )
                promoted.append(str(point.id))
                logger.info(
                    "Promoted L1→L2: %s (use=%d, success_rate=%.0f%%)",
                    payload.get("problem_pattern", "")[:50],
                    payload.get("use_count", 0),
                    (success / total) * 100,
                )

    except Exception as e:
        logger.warning("Promotion check failed: %s", e)

    return promoted


async def consolidate_domain_knowledge(
    qdrant: QdrantClient,
    collection_name: str,
    llm=None,
    min_experiences: int = 5,
) -> str | None:
    """Consolidate multiple L2 experiences with shared domain tags into L3 domain knowledge.

    Uses LLM to synthesize a higher-level pattern from related experiences.
    Returns the new L3 point ID, or None.
    """
    if llm is None:
        return None

    try:
        # Find all L2 experiences
        results = qdrant.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="tier", match=MatchValue(value=TIER_L2))]
            ),
            limit=100,
        )

        points, _ = results
        if len(points) < min_experiences:
            return None

        # Group by domain tags
        tag_groups: dict[str, list] = {}
        for p in points:
            for tag in p.payload.get("domain_tags", []):
                tag_groups.setdefault(tag, []).append(p)

        # Find the largest group that hasn't been consolidated yet
        best_tag = None
        best_points = []
        for tag, pts in sorted(tag_groups.items(), key=lambda x: -len(x[1])):
            if len(pts) >= min_experiences:
                best_tag = tag
                best_points = pts[:10]  # cap for LLM context
                break

        if not best_tag:
            return None

        # LLM synthesis
        from langchain_core.messages import HumanMessage, SystemMessage

        summaries = []
        for p in best_points:
            pl = p.payload
            summaries.append(
                f"- Pattern: {pl.get('problem_pattern', '')}\n"
                f"  Strategy: {pl.get('recommended_strategy', '')}\n"
                f"  Tips: {pl.get('tips', '')}\n"
                f"  Tools: {', '.join(pl.get('recommended_agents', []))}"
            )

        messages = [
            SystemMessage(content=(
                "You are a knowledge consolidation expert. Synthesize multiple related experiences "
                "into one high-level domain knowledge entry. Focus on the common patterns, "
                "universal strategies, and key principles that apply across all these experiences.\n\n"
                "Respond with ONLY a JSON object:\n"
                '{"problem_pattern": "high-level domain pattern", '
                '"domain_tags": ["tags"], '
                '"recommended_strategy": "universal strategy for this domain", '
                '"tips": "key principles and common pitfalls", '
                '"recommended_agents": ["commonly needed tools"]}'
            )),
            HumanMessage(content=(
                f"Domain: {best_tag}\n"
                f"Related experiences ({len(best_points)}):\n\n"
                + "\n\n".join(summaries)
            )),
        ]

        import asyncio
        response = await asyncio.wait_for(llm.ainvoke(messages), timeout=20)
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        knowledge = json.loads(content)

        # Store as L3
        from common.vector import text_to_embedding_async

        text_for_embed = f"{knowledge.get('problem_pattern', '')} {knowledge.get('tips', '')}"
        embedding = await text_to_embedding_async(text_for_embed)

        point_id = str(uuid.uuid4())
        payload = {
            "task_id": f"consolidated_{best_tag}_{point_id[:8]}",
            "tier": TIER_L3,
            "tier_name": TIER_NAMES[TIER_L3],
            "problem_pattern": knowledge.get("problem_pattern", ""),
            "domain_tags": knowledge.get("domain_tags", [best_tag]),
            "recommended_strategy": knowledge.get("recommended_strategy", ""),
            "recommended_agents": knowledge.get("recommended_agents", []),
            "tips": knowledge.get("tips", ""),
            "complexity": 0,  # N/A for domain knowledge
            "estimated_rounds": 0,
            "failure_patterns": [],
            "prerequisites": [],
            "recovery_patterns": [],
            "correction_patterns": [],
            "quality_score": 2.0,  # start higher than default (consolidated = validated)
            "use_count": 0,
            "success_count": 0,
            "fail_count": 0,
            "created_at": time.time(),
            "last_used_at": time.time(),
            "promoted_from": TIER_L2,
            "source_count": len(best_points),
        }

        qdrant.upsert(
            collection_name=collection_name,
            points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
        )

        logger.info(
            "Consolidated L3 domain knowledge: tag=%s, sources=%d, id=%s",
            best_tag, len(best_points), point_id[:8],
        )
        return point_id

    except Exception as e:
        logger.warning("Domain consolidation failed: %s", e)
        return None


# ── Stats / introspection ───────────────────────────────────────

def get_experience_stats(
    qdrant: QdrantClient,
    collection_name: str,
) -> dict[str, Any]:
    """Get summary statistics of the experience library.

    Returns counts per tier, top domain tags, average quality, etc.
    """
    stats: dict[str, Any] = {"total": 0, "by_tier": {}, "top_tags": {}, "avg_quality": 0}

    try:
        results = qdrant.scroll(
            collection_name=collection_name,
            limit=500,
            with_payload=True,
            with_vectors=False,
        )

        points, _ = results
        stats["total"] = len(points)

        quality_sum = 0.0
        for p in points:
            pl = p.payload
            tier_name = pl.get("tier_name", "unknown")
            stats["by_tier"][tier_name] = stats["by_tier"].get(tier_name, 0) + 1

            for tag in pl.get("domain_tags", []):
                stats["top_tags"][tag] = stats["top_tags"].get(tag, 0) + 1

            quality_sum += pl.get("quality_score", 1.0)

        if points:
            stats["avg_quality"] = round(quality_sum / len(points), 2)

        # Sort tags by frequency
        stats["top_tags"] = dict(
            sorted(stats["top_tags"].items(), key=lambda x: -x[1])[:15]
        )

    except Exception as e:
        logger.debug("get_experience_stats failed: %s", e)

    return stats
