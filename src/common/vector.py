"""Vector store client and embedding generation.

Three-tier embedding strategy:
  1. API embedding — real semantic vectors via OpenAI-compatible endpoint
  2. LLM keyword extraction — extract keywords, then hash for deterministic vectors
  3. Hash fallback — SHA-512 content hash (no semantic capability, last resort)

Tier 1 requires embedding config in settings.yaml. If unavailable,
automatically falls back to tier 2 (if LLM available) or tier 3.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
from functools import lru_cache
from typing import Any

import httpx
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from common.config import get_settings

logger = logging.getLogger(__name__)

# ── Qdrant client ────────────────────────────────────────────────


def get_qdrant_client(in_memory: bool = False) -> QdrantClient:
    """Get a Qdrant client. Uses in-memory mode if requested or if remote is unavailable."""
    if in_memory:
        logger.info("Using in-memory Qdrant")
        return QdrantClient(location=":memory:")

    settings = get_settings()
    try:
        client = QdrantClient(url=settings.qdrant.url, timeout=3)
        client.get_collections()  # connectivity check
        logger.info("Connected to Qdrant at %s", settings.qdrant.url)
        return client
    except Exception:
        logger.warning("Qdrant not available at %s, falling back to in-memory", settings.qdrant.url)
        return QdrantClient(location=":memory:")


def get_vector_dim() -> int:
    """Return the configured vector dimension."""
    settings = get_settings()
    if settings.embedding.model and settings.embedding.dimensions:
        return settings.embedding.dimensions
    return 1024  # default for LLM keyword + hash mode


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int | None = None):
    """Create collection if it doesn't exist."""
    if vector_size is None:
        vector_size = get_vector_dim()
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


# ── Embedding: Tier 1 — API ─────────────────────────────────────

@lru_cache(maxsize=1)
def _get_embedding_config() -> tuple[str, str, str, int] | None:
    """Return (model, api_key, base_url, dimensions) if API embedding is configured."""
    settings = get_settings()
    ecfg = settings.embedding
    if not ecfg.model:
        return None

    api_key = ecfg.api_key
    base_url = ecfg.base_url

    # Fallback: reuse LLM provider's credentials
    if not api_key or not base_url:
        llm_cfg = settings.llm.get_model()
        if llm_cfg:
            if not api_key:
                api_key = llm_cfg.api_key
            if not base_url:
                base_url = llm_cfg.base_url

    if not api_key or not base_url:
        return None

    # Normalize base_url: ensure it has compatible embedding path
    # DashScope (both coding.dashscope and dashscope) needs /compatible-mode/v1
    if "dashscope" in base_url and "compatible-mode" not in base_url:
        # Preserve the original host (coding.dashscope vs dashscope)
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        base_url = f"{parsed.scheme}://{parsed.hostname}/compatible-mode/v1"

    return ecfg.model, api_key, base_url, ecfg.dimensions


def _embed_via_api(texts: list[str]) -> list[list[float]] | None:
    """Call OpenAI-compatible embedding endpoint. Returns None on failure."""
    cfg = _get_embedding_config()
    if cfg is None:
        return None

    model, api_key, base_url, dimensions = cfg
    url = f"{base_url.rstrip('/')}/embeddings"

    try:
        resp = httpx.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": texts, "dimensions": dimensions},
            timeout=15,
        )
        if resp.status_code != 200:
            logger.warning("Embedding API returned %d: %s", resp.status_code, resp.text[:200])
            return None

        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        logger.debug("Embedding API success: %d vectors, dim=%d", len(embeddings), len(embeddings[0]))
        return embeddings
    except Exception as e:
        logger.warning("Embedding API call failed: %s", e)
        return None


# ── Embedding: Tier 2 — LLM keyword extraction + hash ───────────

# Shared LLM reference (set by init code to avoid circular imports)
_shared_llm: Any = None


def set_shared_llm(llm: Any) -> None:
    """Set the shared LLM for keyword-based embedding fallback."""
    global _shared_llm
    _shared_llm = llm


_keyword_cache: dict[str, list[str]] = {}


async def _extract_keywords_llm(text: str) -> list[str]:
    """Use LLM to extract semantic keywords from text."""
    if _shared_llm is None:
        return []

    # Check cache
    cache_key = text[:200]
    if cache_key in _keyword_cache:
        return _keyword_cache[cache_key]

    from langchain_core.messages import HumanMessage, SystemMessage

    messages = [
        SystemMessage(content=(
            "从文本中提取5-10个语义关键词，用于检索匹配。"
            "返回 JSON 数组，如 [\"关键词1\", \"关键词2\"]。"
            "只返回 JSON，不要解释。"
        )),
        HumanMessage(content=text[:500]),
    ]

    try:
        response = await _shared_llm.ainvoke(messages)
        content = response.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        keywords = json.loads(content)
        if isinstance(keywords, list):
            keywords = [str(k).strip().lower() for k in keywords if k]
            _keyword_cache[cache_key] = keywords
            # Cap cache size
            if len(_keyword_cache) > 500:
                oldest = next(iter(_keyword_cache))
                del _keyword_cache[oldest]
            return keywords
    except Exception as e:
        logger.debug("LLM keyword extraction failed: %s", e)

    return []


def _keywords_to_vector(keywords: list[str], dim: int = 1024) -> list[float]:
    """Convert keywords to a semantic-aware vector.

    Each keyword contributes a hash-based sub-vector. Because similar texts
    share keywords, their vectors will have partial overlap → non-zero cosine similarity.
    This is far better than full-text hashing where ANY character difference → zero similarity.
    """
    if not keywords:
        return _hash_embedding("", dim)

    # Sort keywords for stability, then combine
    keywords = sorted(set(keywords))

    # Each keyword contributes a component vector
    vector = [0.0] * dim
    for kw in keywords:
        kw_hash = hashlib.sha256(kw.encode()).digest()
        # Expand to dim
        expanded = b""
        for i in range((dim * 4 // 32) + 1):
            expanded += hashlib.sha256(f"{kw}:{i}".encode()).digest()
        for j in range(dim):
            val = struct.unpack_from(">I", expanded, j * 4)[0]
            vector[j] += (val / 2147483647.5) - 1.0

    # L2 normalize
    norm = sum(f * f for f in vector) ** 0.5
    if norm > 0:
        vector = [f / norm for f in vector]

    return vector


# ── Embedding: Tier 3 — Pure hash fallback ───────────────────────

def _hash_embedding(text: str, dim: int = 1024) -> list[float]:
    """Deterministic hash-based embedding. No semantic capability."""
    text = text.strip().lower()
    hash_bytes = b""
    for i in range((dim * 4 // 64) + 1):
        hash_bytes += hashlib.sha512(f"{text}:{i}".encode()).digest()

    floats = []
    for j in range(dim):
        val = struct.unpack_from(">I", hash_bytes, j * 4)[0]
        floats.append((val / 2147483647.5) - 1.0)

    norm = sum(f * f for f in floats) ** 0.5
    if norm > 0:
        floats = [f / norm for f in floats]
    return floats


# ── Public API ───────────────────────────────────────────────────

def text_to_embedding(text: str, dim: int | None = None) -> list[float]:
    """Convert text to embedding vector (synchronous).

    Tries API embedding first, falls back to hash embedding.
    For keyword-based embedding (tier 2), use text_to_embedding_async.
    """
    if dim is None:
        dim = get_vector_dim()

    # Tier 1: API embedding
    result = _embed_via_api([text])
    if result and result[0]:
        return result[0]

    # Tier 3: Hash fallback (tier 2 needs async)
    logger.debug("Using hash fallback for embedding")
    return _hash_embedding(text, dim)


async def text_to_embedding_async(text: str, dim: int | None = None) -> list[float]:
    """Convert text to embedding vector (async, supports all 3 tiers).

    Priority: API embedding → LLM keyword extraction → hash fallback.
    """
    if dim is None:
        dim = get_vector_dim()

    # Tier 1: API embedding
    result = _embed_via_api([text])
    if result and result[0]:
        return result[0]

    # Tier 2: LLM keyword extraction + hash
    keywords = await _extract_keywords_llm(text)
    if keywords:
        logger.debug("Using LLM keyword embedding (%d keywords)", len(keywords))
        return _keywords_to_vector(keywords, dim)

    # Tier 3: Hash fallback
    logger.debug("Using hash fallback for embedding")
    return _hash_embedding(text, dim)


async def batch_embed_async(texts: list[str], dim: int | None = None) -> list[list[float]]:
    """Batch embed multiple texts (async)."""
    if dim is None:
        dim = get_vector_dim()

    # Tier 1: Try API batch
    result = _embed_via_api(texts)
    if result and len(result) == len(texts):
        return result

    # Fallback: individual async embedding
    return [await text_to_embedding_async(t, dim) for t in texts]
