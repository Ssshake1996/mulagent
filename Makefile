.PHONY: dev bot up down test test-unit test-int test-e2e lint format install migrate clean logs health

# ── Development ──────────────────────────────────────────
# Run API development server
dev:
	PYTHONPATH=src uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Run Feishu bot (WebSocket long-connection)
bot:
	PYTHONPATH=src python -m gateway.feishu_bot

# ── Infrastructure ───────────────────────────────────────
# Start all infrastructure services (Qdrant, Redis, PostgreSQL)
up:
	docker compose -f docker/docker-compose.yaml up -d

# Stop infrastructure services
down:
	docker compose -f docker/docker-compose.yaml down

# View logs from infrastructure services
logs:
	docker compose -f docker/docker-compose.yaml logs -f --tail=50

# ── Dependencies ─────────────────────────────────────────
# Install all dependencies
install:
	pip install -e ".[dev]"

# ── Database ─────────────────────────────────────────────
# Run database migrations
migrate:
	PYTHONPATH=src alembic upgrade head

# Create a new migration
migrate-new:
	@read -p "Migration message: " msg; \
	PYTHONPATH=src alembic revision --autogenerate -m "$$msg"

# ── Testing ──────────────────────────────────────────────
# Run all tests
test:
	PYTHONPATH=src python -m pytest tests/ -v

# Run unit tests only
test-unit:
	PYTHONPATH=src python -m pytest tests/unit/ -v

# Run integration tests only
test-int:
	PYTHONPATH=src python -m pytest tests/integration/ -v

# Run e2e tests only
test-e2e:
	PYTHONPATH=src python -m pytest tests/e2e/ -v

# ── Code Quality ─────────────────────────────────────────
# Lint check
lint:
	ruff check src/ tests/

# Auto-format code
format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# ── Operations ───────────────────────────────────────────
# Health check
health:
	@curl -s http://localhost:8000/health | python -m json.tool 2>/dev/null || echo "API not running"

# Hot-reload config
reload:
	@curl -s -X POST http://localhost:8000/config/reload | python -m json.tool

# ── Cleanup ──────────────────────────────────────────────
# Clean temporary files and caches
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage
	rm -rf /tmp/mulagent_files

# ── Help ─────────────────────────────────────────────────
help:
	@echo "Available targets:"
	@echo "  dev         - Run API server (hot-reload)"
	@echo "  bot         - Run Feishu bot"
	@echo "  up/down     - Start/stop infrastructure"
	@echo "  logs        - Tail infrastructure logs"
	@echo "  install     - Install dependencies"
	@echo "  migrate     - Run DB migrations"
	@echo "  test        - Run all tests"
	@echo "  test-unit   - Run unit tests"
	@echo "  lint        - Check code quality"
	@echo "  format      - Auto-format code"
	@echo "  health      - Check API health"
	@echo "  reload      - Hot-reload config"
	@echo "  clean       - Remove temp files"
