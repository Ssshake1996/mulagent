"""Qdrant vector store client with in-memory support."""

from __future__ import annotations

import hashlib
import logging
import struct

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from common.config import get_settings

logger = logging.getLogger(__name__)

# Default vector dimension for our hash-based embeddings
VECTOR_DIM = 256


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


def ensure_collection(client: QdrantClient, collection_name: str, vector_size: int = VECTOR_DIM):
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def text_to_embedding(text: str, dim: int = VECTOR_DIM) -> list[float]:
    """Convert text to a deterministic embedding vector using content hashing.

    This is a lightweight, dependency-free approach. For production use,
    replace with a real embedding model (e.g., text-embedding-ada-002).
    Uses SHA-512 repeatedly to fill the vector dimensions.
    """
    # Normalize text
    text = text.strip().lower()
    # Generate enough hash bytes to fill dim floats
    hash_bytes = b""
    for i in range((dim * 4 // 64) + 1):
        hash_bytes += hashlib.sha512(f"{text}:{i}".encode()).digest()

    # Convert to floats in [-1, 1]
    floats = []
    for j in range(dim):
        # Unpack 4 bytes as unsigned int, map to [-1, 1]
        val = struct.unpack_from(">I", hash_bytes, j * 4)[0]
        floats.append((val / 2147483647.5) - 1.0)

    # L2 normalize
    norm = sum(f * f for f in floats) ** 0.5
    if norm > 0:
        floats = [f / norm for f in floats]

    return floats
