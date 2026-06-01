.PHONY: install lint fmt type-check test test-unit test-integration run clean

# ── Install ────────────────────────────────────────────────────────────────
install:
	uv sync --all-extras

# ── Lint & Format ──────────────────────────────────────────────────────────
lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

fmt-check:
	uv run ruff format --check src tests

type-check:
	uv run mypy src/cop_fx

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

run-publish:
	uv run cop-fx run --publish

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
	@echo "  install         Install all dependencies with uv"
	@echo "  lint            Run ruff linter"
	@echo "  fmt             Auto-format with ruff"
	@echo "  fmt-check       Check formatting (CI mode)"
	@echo "  type-check      Run mypy type checker"
	@echo "  test            Run all tests"
	@echo "  test-unit       Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  test-cov        Run tests with HTML coverage report"
	@echo "  run             Run daily pipeline (dry-run, no tweet)"
	@echo "  run-publish     Run pipeline and publish tweet"
	@echo "  clean           Remove build artifacts and caches"
