"""Redis client with rate limiting and caching utilities.

Provides:
- Connection management with lazy initialization
- Rate limiting (per-user, sliding window)
- Key-value cache with TTL
- Distributed message dedup
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Lazy-initialized global client
_redis = None
_available: bool | None = None


async def get_redis():
    """Get or create the Redis async client. Returns None if unavailable."""
    global _redis, _available
    if _available is False:
        return None
    if _redis is not None:
        return _redis

    try:
        import redis.asyncio as aioredis
        from common.config import get_settings
        settings = get_settings()
        _redis = aioredis.from_url(
            settings.redis.url,
            decode_responses=True,
            socket_connect_timeout=3,
        )
        await _redis.ping()
        _available = True
        logger.info("Redis connected: %s", settings.redis.url)
        return _redis
    except Exception as e:
        logger.info("Redis unavailable: %s", e)
        _available = False
        _redis = None
        return None




# ── Cache ────────────────────────────────────────────────────────

async def cache_get(key: str) -> str | None:
    """Get a cached value."""
    r = await get_redis()
    if r is None:
        return None
    try:
        return await r.get(key)
    except Exception:
        return None


async def cache_set(key: str, value: str, ttl: int = 300) -> bool:
    """Set a cached value with TTL."""
    r = await get_redis()
    if r is None:
        return False
    try:
        await r.setex(key, ttl, value)
        return True
    except Exception:
        return False


# ── Distributed Dedup ────────────────────────────────────────────

async def is_duplicate(key: str, ttl: int = 300) -> bool:
    """Check if a message/event is a duplicate using Redis SET NX.

    Returns True if the key already exists (duplicate).
    """
    r = await get_redis()
    if r is None:
        return False  # No Redis → can't dedup, assume not duplicate
    try:
        result = await r.set(key, "1", nx=True, ex=ttl)
        return result is None  # None means key already existed
    except Exception:
        return False
