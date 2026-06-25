"""Proveedores de inyección de dependencias (DI).

Los routers declaran ``svc = Depends(get_predictions_service)`` y reciben una
instancia lista. Centralizar la construcción aquí permite:

  - cablear los stores una sola vez (un ``JobRegistry`` singleton de proceso),
  - sobreescribir cualquier dependencia en tests con
    ``app.dependency_overrides[...]`` sin tocar los routers.
"""

from __future__ import annotations

from functools import lru_cache

from cop_fx.api.services import (
    ForecastService,
    NewsService,
    PipelineService,
    PredictionsService,
    ReportsService,
)
from cop_fx.api.services.pipeline_service import JobRegistry
from cop_fx.config.settings import Settings, get_settings
from cop_fx.tracking.predictions import PredictionStore


@lru_cache(maxsize=1)
def get_job_registry() -> JobRegistry:
    """Registro de jobs único por proceso (estado de las corridas en background)."""
    return JobRegistry()


def get_app_settings() -> Settings:
    return get_settings()


def get_predictions_service() -> PredictionsService:
    return PredictionsService(PredictionStore())


def get_forecast_service() -> ForecastService:
    return ForecastService(get_settings())


def get_news_service() -> NewsService:
    return NewsService()


def get_reports_service() -> ReportsService:
    return ReportsService()


def get_pipeline_service() -> PipelineService:
    return PipelineService(get_job_registry())
