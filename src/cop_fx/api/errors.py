"""Errores de dominio + handlers HTTP centralizados.

En vez de que cada router arme su propio ``HTTPException`` con strings sueltos,
los servicios lanzan errores de dominio tipados (``ResourceNotFound``,
``DataUnavailable``, ``PipelineError``) y un único handler los traduce a una
respuesta JSON consistente. Esto mantiene los routers limpios (no atrapan
excepciones) y la forma del error es la misma en toda la API (SRP).
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from cop_fx.logger import get_logger

logger = get_logger(__name__)


class ApiError(Exception):
    """Base de los errores de dominio que la API sabe traducir a HTTP."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ResourceNotFound(ApiError):
    """El recurso pedido no existe (404)."""

    status_code = status.HTTP_404_NOT_FOUND
    code = "not_found"


class DataUnavailable(ApiError):
    """Dependencia de datos vacía o no disponible todavía (503)."""

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "data_unavailable"


class PipelineError(ApiError):
    """El pipeline falló durante la ejecución (422)."""

    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT
    code = "pipeline_error"


def _error_body(code: str, message: str) -> dict[str, dict[str, str]]:
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    """Registra los handlers que convierten ApiError → JSON uniforme."""

    @app.exception_handler(ApiError)
    async def _handle_api_error(_: Request, exc: ApiError) -> JSONResponse:
        if exc.status_code >= status.HTTP_500_INTERNAL_SERVER_ERROR:
            logger.error("ApiError %s: %s", exc.code, exc.message)
        else:
            logger.warning("ApiError %s: %s", exc.code, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content=_error_body(exc.code, exc.message),
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Excepción no controlada: %s", exc)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_error_body("internal_error", "Error interno inesperado"),
        )
