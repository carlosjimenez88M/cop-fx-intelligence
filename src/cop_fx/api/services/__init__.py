"""Capa de servicios: la lógica de la API, desacoplada del transporte HTTP.

Cada servicio expone métodos async y traduce los stores/pipeline del dominio a
los schemas de la API. Los routers dependen de los *protocolos* (no de las
implementaciones concretas), de modo que se pueden sustituir en tests sin tocar
la capa HTTP (Dependency Inversion).
"""

from __future__ import annotations

from cop_fx.api.services.forecast_service import ForecastService
from cop_fx.api.services.news_service import NewsService
from cop_fx.api.services.pipeline_service import PipelineService
from cop_fx.api.services.predictions_service import PredictionsService
from cop_fx.api.services.reports_service import ReportsService

__all__ = [
    "ForecastService",
    "NewsService",
    "PipelineService",
    "PredictionsService",
    "ReportsService",
]
