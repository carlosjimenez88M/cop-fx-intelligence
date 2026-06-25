"""Metadatos y configuración propios de la capa HTTP.

La configuración OPERATIVA del sistema (modelos, feeds, bandas) ya vive en
``cop_fx.config.settings``. Aquí solo añadimos lo que es exclusivo de la API:
título/versión de OpenAPI y la política de CORS. Reutilizamos el singleton de
settings en vez de duplicar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

API_VERSION = "1.0.0"
API_PREFIX = "/api/v1"


@dataclass(frozen=True)
class ApiMetadata:
    """Metadatos de OpenAPI (lo que se ve en /docs)."""

    title: str = "COP/USD Intelligence API"
    version: str = API_VERSION
    description: str = (
        "API del sistema multiagéntico que estima la **dirección diaria del "
        "USD/COP** (`down`/`up`/`neutral`) cruzando noticias, serie de tiempo "
        "y contexto de mercado bajo confianza acotada.\n\n"
        "La abstención (`neutral`) es una capacidad, no una falla."
    )
    contact: dict[str, str] = field(
        default_factory=lambda: {
            "name": "Carlos Daniel Jiménez",
            "email": "danieljimenez88m@gmail.com",
        }
    )
    license_info: dict[str, str] = field(default_factory=lambda: {"name": "MIT"})
    openapi_tags: list[dict[str, str]] = field(
        default_factory=lambda: [
            {"name": "liveness", "description": "Health y readiness checks."},
            {"name": "predictions", "description": "Calls direccionales y track record."},
            {"name": "forecast", "description": "Forecast Prophet + ARIMA del USD/COP."},
            {"name": "news", "description": "Noticias clasificadas (capa GOLD)."},
            {"name": "reports", "description": "Reportes diarios en Markdown."},
            {"name": "pipeline", "description": "Disparar y monitorear corridas del pipeline."},
        ]
    )


@dataclass(frozen=True)
class CorsPolicy:
    """Política CORS. Por defecto abierta (uso interno); ajustar en producción."""

    allow_origins: list[str] = field(default_factory=lambda: ["*"])
    allow_methods: list[str] = field(default_factory=lambda: ["*"])
    allow_headers: list[str] = field(default_factory=lambda: ["*"])
    allow_credentials: bool = False
