"""Endpoint de forecast Prophet + ARIMA."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from cop_fx.api.dependencies import get_forecast_service
from cop_fx.api.schemas import ForecastResponse
from cop_fx.api.services.protocols import ForecastServiceProtocol

router = APIRouter(prefix="/forecast", tags=["forecast"])

ServiceDep = Annotated[ForecastServiceProtocol, Depends(get_forecast_service)]


@router.get("", response_model=ForecastResponse, summary="Forecast del USD/COP")
async def forecast(
    service: ServiceDep,
    horizon_days: Annotated[int | None, Query(ge=1, le=30)] = None,
) -> ForecastResponse:
    return await service.compute(horizon_days)
