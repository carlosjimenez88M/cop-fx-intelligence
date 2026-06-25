"""Fábrica de la aplicación FastAPI: ensambla config, middleware y routers.

``create_app()`` es una *factory* (no un singleton de módulo) para que los
tests puedan instanciar apps frescas y sobreescribir dependencias sin estado
compartido. El objeto se expone también como ``app`` para uvicorn/gunicorn.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cop_fx.api.config import API_PREFIX, ApiMetadata, CorsPolicy
from cop_fx.api.errors import register_exception_handlers
from cop_fx.api.middleware import RequestContextMiddleware
from cop_fx.api.routers import forecast, health, news, pipeline, predictions, reports
from cop_fx.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("API arrancando: %s", app.title)
    yield
    logger.info("API apagándose")


def create_app(
    *,
    metadata: ApiMetadata | None = None,
    cors: CorsPolicy | None = None,
) -> FastAPI:
    """Construye y devuelve la aplicación FastAPI lista para servir."""
    meta = metadata or ApiMetadata()
    cors_policy = cors or CorsPolicy()

    app = FastAPI(
        title=meta.title,
        version=meta.version,
        description=meta.description,
        contact=meta.contact,
        license_info=meta.license_info,
        openapi_tags=meta.openapi_tags,
        lifespan=_lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_policy.allow_origins,
        allow_methods=cors_policy.allow_methods,
        allow_headers=cors_policy.allow_headers,
        allow_credentials=cors_policy.allow_credentials,
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    # Health en la raíz; el resto de recursos bajo /api/v1.
    app.include_router(health.router)
    for module in (predictions, forecast, news, reports, pipeline):
        app.include_router(module.router, prefix=API_PREFIX)

    return app


app = create_app()
