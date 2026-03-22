# ── Multi-stage Dockerfile for mul-agent ──────────────────────────
#
# Build:   docker build -t mulagent .
# Run API: docker run -p 8000:8000 --env-file .env mulagent api
# Run Bot: docker run --env-file .env mulagent bot
#
# The image includes both API server and Feishu bot entry points.
# Select which to run via the command argument (default: api).

# ── Stage 1: Dependencies ──
FROM python:3.12-slim AS deps

WORKDIR /app

# System deps for asyncpg, httpx, curl (for web_fetch fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]" 2>/dev/null || pip install --no-cache-dir .

# ── Stage 2: Application ──
FROM python:3.12-slim AS app

WORKDIR /app

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/
COPY openclaw_agents/ ./openclaw_agents/
COPY pyproject.toml ./

# Create data directories
RUN mkdir -p data/conversations /tmp/mulagent_files

# Environment
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Health check (for API mode)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

EXPOSE 8000

# Entry point script
COPY <<'ENTRYPOINT' /app/entrypoint.sh
#!/bin/bash
set -e

MODE="${1:-api}"

case "$MODE" in
    api)
        echo "Starting API server..."
        exec uvicorn main:create_app --factory --host 0.0.0.0 --port 8000
        ;;
    bot)
        echo "Starting Feishu bot..."
        exec python -m gateway.feishu_bot
        ;;
    worker)
        echo "Starting background worker..."
        exec python -m workers.main
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: docker run mulagent [api|bot|worker]"
        exit 1
        ;;
esac
ENTRYPOINT

RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["api"]
