"""Redis client with rate limiting and caching utilities.

Provides:
- Connection management with lazy initialization
- Rate limiting (per-user, sliding window)
- Key-value cache with TTL
- Distributed message dedup
"""

from __future__ import annotations

import logging
import time
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


# ── Rate Limiting ────────────────────────────────────────────────

async def check_rate_limit(
    key: str,
    max_requests: int = 10,
    window_seconds: int = 60,
) -> tuple[bool, int]:
    """Sliding window rate limiter.

    Args:
        key: Rate limit key (e.g., f"rate:{user_id}")
        max_requests: Maximum requests per window
        window_seconds: Window size in seconds

    Returns:
        (allowed, remaining) — whether the request is allowed and remaining quota
    """
    r = await get_redis()
    if r is None:
        return True, max_requests  # No Redis → no rate limiting

    now = time.time()
    window_start = now - window_seconds
    pipe = r.pipeline()

    # Remove expired entries
    pipe.zremrangebyscore(key, 0, window_start)
    # Count current entries
    pipe.zcard(key)
    # Add current request
    pipe.zadd(key, {str(now): now})
    # Set key expiry
    pipe.expire(key, window_seconds + 1)

    results = await pipe.execute()
    current_count = results[1]

    if current_count >= max_requests:
        return False, 0

    return True, max_requests - current_count - 1


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
