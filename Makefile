.PHONY: dev up down test lint

# Start infrastructure services
up:
	docker compose -f docker/docker-compose.yaml up -d

# Stop infrastructure services
down:
	docker compose -f docker/docker-compose.yaml down

# Install dependencies
install:
	pip install -e ".[dev]"

# Run development server
dev:
	uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
test:
	python -m pytest tests/ -v

# Run unit tests only
test-unit:
	python -m pytest tests/unit/ -v

# Run integration tests only
test-int:
	python -m pytest tests/integration/ -v

# Run e2e tests only
test-e2e:
	python -m pytest tests/e2e/ -v

# Lint
lint:
	ruff check src/ tests/
	ruff format --check src/ tests/
