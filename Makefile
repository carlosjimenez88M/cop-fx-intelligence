.PHONY: install lint fmt fmt-check type-check test test-unit test-integration \
        test-cov run api docker-build docker-up docker-down \
        pre-commit-install pre-commit clean help

# ── Install ────────────────────────────────────────────────────────────────
install:
	uv sync --all-extras
	uv run pre-commit install

# ── Lint & Format ──────────────────────────────────────────────────────────
lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

fmt-check:
	uv run ruff format --check src tests

type-check:
	uv run mypy src/cop_fx

# ── Pre-commit ───────────────────────────────────────────────────────────────
pre-commit-install:
	uv run pre-commit install

pre-commit:
	uv run pre-commit run --all-files

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	uv run pytest

test-unit:
	uv run pytest -m unit -v

test-integration:
	uv run pytest -m integration -v

test-cov:
	uv run pytest --cov=src/cop_fx --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

# ── Run ────────────────────────────────────────────────────────────────────
run:
	uv run cop-fx run

api:
	uv run cop-fx-api

# ── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker compose build

docker-up:
	docker compose up api dashboard

docker-down:
	docker compose down

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

help:
	@echo "Available targets:"
	@echo "  install            Install deps with uv + register pre-commit hooks"
	@echo "  lint               Run ruff linter"
	@echo "  fmt                Auto-format with ruff"
	@echo "  fmt-check          Check formatting (CI mode)"
	@echo "  type-check         Run mypy type checker"
	@echo "  pre-commit-install Register the git pre-commit hooks"
	@echo "  pre-commit         Run all pre-commit hooks on every file"
	@echo "  test               Run all tests"
	@echo "  test-unit          Run unit tests only"
	@echo "  test-integration   Run integration tests only"
	@echo "  test-cov           Run tests with HTML coverage report"
	@echo "  run                Run daily pipeline"
	@echo "  api                Serve the FastAPI API (uvicorn)"
	@echo "  docker-build       Build all docker images"
	@echo "  docker-up          Run api + dashboard via docker compose"
	@echo "  docker-down        Stop docker compose services"
	@echo "  clean              Remove build artifacts and caches"
