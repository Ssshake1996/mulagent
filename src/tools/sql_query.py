"""SQL query tool for the analyst role.

Safety-first design:
- Read-only by default (wraps queries in read-only transactions)
- Auto-rollback on write attempts
- Schema auto-discovery for context injection
- Query timeout enforcement
- Result size limits

Uses the existing PostgreSQL connection from common.db.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from tools.base import ToolDef

logger = logging.getLogger(__name__)

_MAX_ROWS = 100
_MAX_RESULT_CHARS = 5000
_QUERY_TIMEOUT = 30  # seconds

# Patterns that indicate write operations
_WRITE_PATTERNS = [
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE)\b",
    r"\b(GRANT|REVOKE)\b",
]


def _is_write_query(sql: str) -> bool:
    """Detect if a SQL query attempts to modify data."""
    # Strip comments and string literals for analysis
    cleaned = re.sub(r"--.*$", "", sql, flags=re.MULTILINE)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"'[^']*'", "''", cleaned)
    cleaned = cleaned.upper()

    for pattern in _WRITE_PATTERNS:
        if re.search(pattern, cleaned):
            return True
    return False


async def _discover_schema(engine: Any) -> str:
    """Auto-discover database schema for context injection."""
    from sqlalchemy import text

    schema_parts = []
    try:
        async with engine.connect() as conn:
            # Get table names
            result = await conn.execute(text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' ORDER BY table_name"
            ))
            tables = [row[0] for row in result]

            for table in tables[:20]:  # Limit to 20 tables
                # Get column info
                cols_result = await conn.execute(text(
                    "SELECT column_name, data_type, is_nullable "
                    "FROM information_schema.columns "
                    "WHERE table_schema = 'public' AND table_name = :table "
                    "ORDER BY ordinal_position"
                ), {"table": table})
                columns = [
                    f"  {row[0]}: {row[1]}{' (nullable)' if row[2] == 'YES' else ''}"
                    for row in cols_result
                ]
                schema_parts.append(f"Table: {table}\n" + "\n".join(columns))

    except Exception as e:
        logger.warning("Schema discovery failed: %s", e)
        return f"Schema discovery error: {e}"

    return "\n\n".join(schema_parts) if schema_parts else "No tables found"


async def _sql_query(params: dict[str, Any], **deps: Any) -> str:
    """Execute a read-only SQL query against the database.

    Write queries are detected and blocked. All queries run in
    read-only transactions with auto-rollback on any write attempt.
    """
    query = params.get("query", "").strip()
    if not query:
        return "Error: query is required"

    # Schema discovery mode
    if query.lower() in ("schema", "show tables", "\\dt", "describe"):
        engine = deps.get("db_engine")
        if engine is None:
            return "Error: database not configured"
        return await _discover_schema(engine)

    # Block write queries
    if _is_write_query(query):
        return (
            "BLOCKED: write operations are not allowed. "
            "This tool is read-only for safety. "
            f"Detected write attempt in: {query[:100]}"
        )

    # Enforce LIMIT to prevent huge result sets
    if not re.search(r'\bLIMIT\b', query, re.IGNORECASE):
        query = f"{query.rstrip(';')} LIMIT {_MAX_ROWS}"

    engine = deps.get("db_engine")
    if engine is None:
        # Try to get engine from settings
        try:
            from common.db import get_engine
            engine = get_engine()
        except Exception:
            return "Error: database not configured"

    from sqlalchemy import text
    import asyncio

    start = time.perf_counter()
    try:
        async with engine.connect() as conn:
            # Set statement timeout
            await conn.execute(text(f"SET statement_timeout = '{_QUERY_TIMEOUT}s'"))

            # Execute in a read-only transaction
            await conn.execute(text("SET TRANSACTION READ ONLY"))

            result = await asyncio.wait_for(
                conn.execute(text(query)),
                timeout=_QUERY_TIMEOUT + 5,
            )

            rows = result.fetchall()
            columns = list(result.keys()) if result.keys() else []

            elapsed = time.perf_counter() - start

            # Format results
            if not rows:
                return f"Query returned 0 rows ({elapsed:.2f}s)\n\nSQL: {query[:200]}"

            # Build table
            lines = [f"Columns: {', '.join(columns)}"]
            lines.append("-" * 60)

            char_count = 0
            for i, row in enumerate(rows):
                line = " | ".join(str(v) for v in row)
                if char_count + len(line) > _MAX_RESULT_CHARS:
                    lines.append(f"... ({len(rows) - i} more rows truncated)")
                    break
                lines.append(line)
                char_count += len(line)

            lines.append(f"\n{len(rows)} rows returned ({elapsed:.2f}s)")
            return "\n".join(lines)

    except asyncio.TimeoutError:
        return f"Query timed out ({_QUERY_TIMEOUT}s): {query[:200]}"
    except Exception as e:
        elapsed = time.perf_counter() - start
        error_msg = str(e)
        if "read-only" in error_msg.lower() or "cannot execute" in error_msg.lower():
            return f"BLOCKED: query attempted a write operation (caught by read-only transaction)"
        return f"SQL error ({elapsed:.2f}s): {error_msg[:300]}"


SQL_QUERY = ToolDef(
    name="sql_query",
    description=(
        "Execute a read-only SQL query against the database. For data analysis, "
        "reporting, and schema exploration. Write operations (INSERT, UPDATE, DELETE, etc.) "
        "are blocked for safety. Use 'schema' as the query to discover available tables. "
        "Results are limited to 100 rows by default."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "SQL query to execute (read-only), or 'schema' to discover tables"
                ),
            },
        },
        "required": ["query"],
    },
    fn=_sql_query,
    category="execution",
)
