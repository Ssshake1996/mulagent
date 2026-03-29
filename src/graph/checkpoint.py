"""Task checkpoint and recovery: persist ReAct loop state for resumption.

When a task is interrupted (timeout, crash, user disconnect), this module
saves the intermediate state to Redis. When the user reconnects or retries,
the state can be restored and the task resumed from the last checkpoint.

Checkpoint data:
- Working memory (directives, state, facts)
- Conversation history (AI + Tool messages)
- Round number and strategies tried
- Tool results cache

Storage: Redis with configurable TTL (default 24h).
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_CHECKPOINT_TTL = 86400  # 24 hours
_CHECKPOINT_PREFIX = "checkpoint:"


def _serialize_memory(memory: Any) -> dict:
    """Serialize WorkingMemory to a dict."""
    return {
        "directives": list(memory.directives),
        "state": dict(memory.state),
        "facts": [
            {
                "source": f.source,
                "content": f.content[:2000],  # Truncate large facts
                "round_num": f.round_num,
                "relevance": f.relevance,
                "pinned": f.pinned,
            }
            for f in memory.facts
        ],
    }


def _deserialize_memory(data: dict) -> Any:
    """Deserialize a dict back to WorkingMemory."""
    from graph.memory import WorkingMemory, Fact
    memory = WorkingMemory()
    for d in data.get("directives", []):
        memory.add_directive(d)
    memory.state = data.get("state", {})
    for f_data in data.get("facts", []):
        memory.facts.append(Fact(
            source=f_data["source"],
            content=f_data["content"],
            round_num=f_data["round_num"],
            relevance=f_data.get("relevance", 1.0),
            pinned=f_data.get("pinned", False),
        ))
    return memory


async def save_checkpoint(
    task_id: str,
    user_input: str,
    memory: Any,
    round_num: int,
    strategies_tried: list[dict],
    tool_cache: dict[str, str],
    session_id: str = "",
) -> bool:
    """Save a ReAct loop checkpoint to Redis.

    Returns True if saved successfully.
    """
    from common.redis_client import cache_set

    checkpoint = {
        "task_id": task_id,
        "user_input": user_input,
        "session_id": session_id,
        "memory": _serialize_memory(memory),
        "round_num": round_num,
        "strategies_tried": strategies_tried[-20:],  # Keep last 20
        "tool_cache": {k: v[:1000] for k, v in list(tool_cache.items())[:30]},
        "saved_at": time.time(),
    }

    key = f"{_CHECKPOINT_PREFIX}{task_id}"
    data = json.dumps(checkpoint, ensure_ascii=False)

    result = await cache_set(key, data, ttl=_CHECKPOINT_TTL)
    if result:
        logger.info("Checkpoint saved for task %s (round %d)", task_id, round_num)
    else:
        logger.warning("Failed to save checkpoint for task %s", task_id)
    return result


async def delete_checkpoint(task_id: str) -> None:
    """Delete a checkpoint after successful task completion."""
    from common.redis_client import get_redis
    r = await get_redis()
    if r is not None:
        try:
            await r.delete(f"{_CHECKPOINT_PREFIX}{task_id}")
            logger.debug("Checkpoint deleted for task %s", task_id)
        except Exception:
            pass


def build_resume_context(checkpoint: dict) -> str:
    """Build a context string for resuming a task.

    This provides the LLM with information about what was already done
    so it can continue from where it left off.
    """
    strategies = checkpoint.get("strategies_tried", [])
    round_num = checkpoint.get("round_num", 0)

    parts = [
        f"[System] This task is being resumed from round {round_num}.",
        f"Original task: {checkpoint.get('user_input', '')[:500]}",
    ]

    if strategies:
        strategy_lines = []
        for s in strategies[-8:]:
            emoji = "✅" if s.get("outcome") == "ok" else "❌"
            strategy_lines.append(f"  {emoji} {s.get('tool', '')}({s.get('args_summary', '')})")
        parts.append("Previous attempts:\n" + "\n".join(strategy_lines))

    memory_data = checkpoint.get("memory", {})
    facts = memory_data.get("facts", [])
    if facts:
        fact_lines = [f"  - [{f['source']}] {f['content'][:150]}" for f in facts[-5:]]
        parts.append("Key findings so far:\n" + "\n".join(fact_lines))

    parts.append("Continue from where you left off. Do not repeat successful steps.")
    return "\n\n".join(parts)


async def list_checkpoints(session_id: str = "") -> list[dict]:
    """List available checkpoints, optionally filtered by session.

    Returns summary info (no large data) for each checkpoint.
    """
    from common.redis_client import get_redis
    r = await get_redis()
    if r is None:
        return []

    try:
        keys = []
        async for key in r.scan_iter(f"{_CHECKPOINT_PREFIX}*"):
            keys.append(key)

        results = []
        for key in keys[:50]:  # Limit scan
            raw = await r.get(key)
            if raw:
                try:
                    cp = json.loads(raw)
                    if session_id and cp.get("session_id") != session_id:
                        continue
                    results.append({
                        "task_id": cp.get("task_id", ""),
                        "user_input": cp.get("user_input", "")[:100],
                        "round_num": cp.get("round_num", 0),
                        "saved_at": cp.get("saved_at", 0),
                        "session_id": cp.get("session_id", ""),
                    })
                except Exception:
                    pass

        return sorted(results, key=lambda x: x.get("saved_at", 0), reverse=True)
    except Exception as e:
        logger.warning("Failed to list checkpoints: %s", e)
        return []
