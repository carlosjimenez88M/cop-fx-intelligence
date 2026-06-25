# syntax=docker/dockerfile:1
#
# COP/USD Intelligence — imagen multi-stage basada en uv.
#
# Inspirado en el patrón multi-stage de la referencia pc-api (base → production
# → test) pero modernizado:
#   - uv en vez de pip: instalación reproducible desde uv.lock (`--frozen`).
#   - capa de dependencias separada del código (cache de Docker eficiente).
#   - usuario NO privilegiado (defensa en profundidad).
#   - targets: production (API), dashboard (Streamlit) y test (CI local).
#
# Build & run:
#   docker compose up api
#   docker compose up dashboard
#   docker compose run --rm test

ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# base — sistema + dependencias del proyecto (sin el código todavía)
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Bytecode precompilado y copia (no symlink) para imágenes autocontenidas.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Toolchain para las ruedas que compilan (prophet/pmdarima/statsmodels).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Usuario no privilegiado.
RUN groupadd --system app && useradd --system --gid app --home-dir /app appuser

# 1) Capa de dependencias: solo manifiestos → se cachea hasta que cambien.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

# 2) El código + instalación del propio paquete.
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# ---------------------------------------------------------------------------
# production — sirve la API FastAPI con uvicorn
# ---------------------------------------------------------------------------
FROM base AS production
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]
CMD ["uvicorn", "cop_fx.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

# ---------------------------------------------------------------------------
# dashboard — Streamlit
# ---------------------------------------------------------------------------
FROM base AS dashboard
USER appuser
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]

# ---------------------------------------------------------------------------
# test — corre la suite (incluye dev deps)
# ---------------------------------------------------------------------------
FROM base AS test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras
CMD ["uv", "run", "pytest", "-m", "unit"]
