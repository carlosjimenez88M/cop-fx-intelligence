"""Endpoints del track record de calls direccionales."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from cop_fx.api.dependencies import get_predictions_service
from cop_fx.api.schemas import MetricsResponse, PredictionItem, PredictionList
from cop_fx.api.services.protocols import PredictionsServiceProtocol

router = APIRouter(prefix="/predictions", tags=["predictions"])

ServiceDep = Annotated[PredictionsServiceProtocol, Depends(get_predictions_service)]


@router.get("/latest", response_model=PredictionItem, summary="Último call vigente")
async def latest(service: ServiceDep) -> PredictionItem:
    return await service.latest()


@router.get("", response_model=PredictionList, summary="Historial de calls")
async def history(
    service: ServiceDep,
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
) -> PredictionList:
    return await service.history(limit)


@router.get("/metrics", response_model=MetricsResponse, summary="Métricas del track record")
async def metrics(service: ServiceDep) -> MetricsResponse:
    return await service.metrics()
