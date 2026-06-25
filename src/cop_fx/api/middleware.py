"""Middleware de contexto de request: id de correlación + tiempo de respuesta.

A cada request se le asigna un ``X-Request-ID`` (o se respeta el entrante) y se
mide su duración, que se devuelve en ``X-Process-Time-Ms`` y se registra. Es la
base mínima de observabilidad sin atar la API a un proveedor concreto (a
diferencia de la referencia, que dependía de Azure/opencensus).
"""

from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING

from starlette.middleware.base import BaseHTTPMiddleware

from cop_fx.logger import get_logger

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from starlette.requests import Request
    from starlette.responses import Response

logger = get_logger(__name__)

_REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Inyecta request-id, mide latencia y registra cada request."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get(_REQUEST_ID_HEADER, uuid.uuid4().hex)
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000

        response.headers[_REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
        logger.info(
            "%s %s → %d (%.1f ms) [%s]",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            request_id,
        )
        return response
