"""Knowledge RAG: Chunk + Embedding for knowledge bases.

Splits knowledge base Markdown files into semantic chunks, embeds them into
Qdrant, and retrieves the most relevant chunks at query time.

This replaces brute-force full-file injection with precision retrieval:
instead of injecting entire .md files (~7500 chars each), we retrieve only
the 3-10 most relevant chunks (~200-500 chars each).

Chunking strategy:
- Split by Markdown headings (##, ###)
- Fallback: split by paragraph (double newline)
- Minimum chunk size: 50 chars (skip tiny fragments)
- Maximum chunk size: 1500 chars (split oversized paragraphs)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"
_KNOWLEDGE_DIR = _CONFIG_DIR / "knowledge"

# In-memory index: chunk_id → chunk text (for retrieval without re-reading files)
_chunk_index: dict[str, str] = {}
# Track which files have been indexed
_indexed_files: dict[str, float] = {}  # filename → mtime when indexed
_COLLECTION_NAME = "knowledge_chunks"


def _chunk_markdown(text: str, max_chunk: int = 1500, min_chunk: int = 50) -> list[str]:
    """Split Markdown text into semantic chunks.

    Strategy:
    1. Split by ## or ### headings (preserving heading as chunk prefix)
    2. If a section is too large, split by paragraphs
    3. Skip fragments shorter than min_chunk
    """
    # Split by headings (keep heading with content)
    sections = re.split(r'\n(?=#{2,3}\s)', text)

    chunks = []
    for section in sections:
        section = section.strip()
        if len(section) < min_chunk:
            continue
        if len(section) <= max_chunk:
            chunks.append(section)
        else:
            # Split oversized sections by paragraphs
            paragraphs = re.split(r'\n\n+', section)
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                # If a single paragraph exceeds max_chunk, hard-split it
                if len(para) > max_chunk:
                    if current and len(current) >= min_chunk:
                        chunks.append(current)
                        current = ""
                    for i in range(0, len(para), max_chunk):
                        piece = para[i:i + max_chunk]
                        if len(piece) >= min_chunk:
                            chunks.append(piece)
                    continue
                if len(current) + len(para) + 2 <= max_chunk:
                    current = f"{current}\n\n{para}" if current else para
                else:
                    if len(current) >= min_chunk:
                        chunks.append(current)
                    current = para
            if len(current) >= min_chunk:
                chunks.append(current)

    return chunks


def _chunk_id(filename: str, chunk_idx: int, content: str) -> str:
    """Generate a stable chunk ID."""
    h = hashlib.md5(content.encode()).hexdigest()[:8]
    return f"{filename}:{chunk_idx}:{h}"


async def index_knowledge_bases(qdrant: Any = None, force: bool = False) -> int:
    """Index all knowledge base files into Qdrant.

    Skips files that haven't changed since last indexing.
    Returns the number of chunks indexed.
    """
    if qdrant is None:
        return 0

    from common.vector import ensure_collection, batch_embed_async, get_vector_dim

    ensure_collection(qdrant, _COLLECTION_NAME, get_vector_dim())

    if not _KNOWLEDGE_DIR.exists():
        return 0

    total_chunks = 0
    for md_file in sorted(_KNOWLEDGE_DIR.glob("*.md")):
        mtime = md_file.stat().st_mtime
        if not force and md_file.name in _indexed_files:
            if _indexed_files[md_file.name] >= mtime:
                continue  # File unchanged

        text = md_file.read_text(encoding="utf-8")
        chunks = _chunk_markdown(text)
        if not chunks:
            continue

        # Generate embeddings
        embeddings = await batch_embed_async(chunks)

        # Upsert to Qdrant
        from qdrant_client.models import PointStruct
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            cid = _chunk_id(md_file.stem, i, chunk)
            _chunk_index[cid] = chunk
            points.append(PointStruct(
                id=hashlib.md5(cid.encode()).hexdigest(),
                vector=embedding,
                payload={
                    "chunk_id": cid,
                    "source_file": md_file.stem,
                    "chunk_index": i,
                    "content": chunk,
                    "char_count": len(chunk),
                },
            ))

        if points:
            qdrant.upsert(collection_name=_COLLECTION_NAME, points=points)
            total_chunks += len(points)

        _indexed_files[md_file.name] = mtime
        logger.info("Indexed %s: %d chunks", md_file.name, len(chunks))

    if total_chunks:
        logger.info("Knowledge indexing complete: %d total chunks", total_chunks)
    return total_chunks


async def retrieve_knowledge(
    query: str,
    qdrant: Any = None,
    top_k: int = 10,
    source_filter: list[str] | None = None,
    score_threshold: float = 0.3,
) -> list[dict[str, Any]]:
    """Retrieve relevant knowledge chunks for a query.

    Args:
        query: The search query.
        qdrant: Qdrant client.
        top_k: Maximum chunks to return.
        source_filter: Only return chunks from these source files.
        score_threshold: Minimum similarity score.

    Returns:
        List of dicts with 'content', 'source_file', 'score', 'chunk_id'.
    """
    if qdrant is None:
        return []

    from common.vector import text_to_embedding_async

    query_vec = await text_to_embedding_async(query[:500])

    # Build filter
    filter_condition = None
    if source_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        filter_condition = Filter(
            must=[FieldCondition(key="source_file", match=MatchAny(any=source_filter))]
        )

    try:
        results = qdrant.search(
            collection_name=_COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            query_filter=filter_condition,
            score_threshold=score_threshold,
        )
    except Exception as e:
        logger.warning("Knowledge retrieval failed: %s", e)
        return []

    chunks = []
    for hit in results:
        payload = hit.payload or {}
        chunks.append({
            "content": payload.get("content", ""),
            "source_file": payload.get("source_file", ""),
            "score": round(hit.score, 3),
            "chunk_id": payload.get("chunk_id", ""),
            "char_count": payload.get("char_count", 0),
        })

    return chunks


def format_knowledge_context(
    chunks: list[dict],
    max_chars: int = 8000,
) -> tuple[str, list[dict]]:
    """Format retrieved chunks into a context string, respecting a char budget.

    Returns:
        (formatted_text, actually_used_chunks) — the text and which chunks were included.
    """
    if not chunks:
        return "", []

    parts = []
    used = []
    char_count = 0

    for chunk in chunks:
        content = chunk["content"]
        if char_count + len(content) + 50 > max_chars:
            logger.info("Knowledge budget reached: %d/%d chars, trimmed %d chunks",
                       char_count, max_chars, len(chunks) - len(used))
            break
        source = chunk.get("source_file", "unknown")
        score = chunk.get("score", 0)
        parts.append(f"### [{source}] (relevance: {score})\n{content}")
        char_count += len(content) + 50
        used.append(chunk)

    text = "\n\n".join(parts) if parts else ""
    return text, used
